"""
Tokenizing EV Charging Infrastructure: Bilevel Model
Implementation of the Stackelberg / nested-logit SUE formulation.
"""
 
import numpy as np
from scipy.special import factorial
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")
 
# ============================================================
# 1. Parameters (override these with paper / calibration values)
# ============================================================
class Params:
    def __init__(self):
        # Horizon & discounting
        self.Y = 10                    # contract horizon (years)
        self.r = 0.05                  # annual discount rate
        self.Ay = (1 - (1 + self.r)**(-self.Y)) / self.r   # annuity factor
 
        # Facility
        self.Ho = 365 * 16             # annual operating hours
        self.rc = 50.0                 # effective charging power (kW)
        self.t0 = 0.1                  # non-charging service time (h)
        self.ce = 0.12                 # electricity cost ($/kWh)
        self.cv = 0.03                 # variable O&M ($/kWh)
        self.cm_coeff = 800.0          # annual maintenance per charger ($)
        self.C_per_charger = 25000.0   # capital cost per charger ($)
        self.B0 = 50000.0              # fixed platform / setup cost ($)
 
        # Token / profit sharing
        self.omega = {"L": 1.0, "H": 2.0}   # contractual profit-claim weights
        self.alpha_bar = 0.7           # upper bound on α
        self.eps_mu = 1e-3             # queue stability margin
        self.eps_F = 1000.0            # financing buffer
        self.B_bar = 100000.0          # max extra fundraising
        self.eps_Pi = 1000.0           # min operating profit
 
        # Price bounds
        self.p_min, self.p_max = 0.15, 0.45
        self.tau_min = {"L": 50.0, "H": 200.0}
        self.tau_max = {"L": 500.0, "H": 2000.0}
 
        # Nested logit
        self.theta = 1/50000.0               # token-nest dissimilarity
 
        # User classes  (example: 3 heterogeneous classes)
        # g = 0,1,2
        self.G = [0, 1, 2]                                 # index of the group
        self.Mg = np.array([800.0, 600.0, 400.0])          # population
        self.fg = np.array([120.0, 150.0, 180.0])          # annual sessions
        self.eg = np.array([25.0, 35.0, 45.0])             # kWh per session
        self.beta_VOT = np.array([100.0, 115.0, 120.0])    # $/h should be within 122 to 120
        self.Dg = np.array([0.25, 0.30, 0.35])             # travel/detour (h)
        self.pg_O = np.array([0.20, 0.20, 0.20])           # outside price ($/kWh), use 3 same number
        self.Gamma_O = np.array([5000.0, 8000.0, 12000.0])       # inconvenience of outside ($/yr)
        self.Z = {"L": 30.0, "H": 80.0}                    # net token-adjustment term
 
        # Service time per class (hours)
        self.sg = self.t0 + self.eg / self.rc
 
 
# ============================================================
# 2. Erlang-C waiting time (multi-server queue)
# ============================================================
def erlang_c(A:float, N:int) -> float:
    """
    Compute the Erlang C formula for waiting probability.
    A: traffic intensity (arrival rate / service rate)
    N: number of servers
    Returns the probability that an arriving customer has to wait.
    """
    if N <= 0:
        return 1.0  # All customers wait if no servers
    if A >= N:
        return 1.0  # System is unstable, all customers wait

    # Compute P0 (probability that there are 0 customers in the system)
    sum_terms = sum((A**k) / factorial(k) for k in range(N))
    last_term = (A**N) / (factorial(N) * (1 - A/N))
    P0 = 1.0 / (sum_terms + last_term)

    # Probability that an arriving customer has to wait
    C = last_term * P0
    return C
 
# ============================================================
# 3. Lower-level nested-logit SUE
# ============================================================
def compute_SUE(z: Dict, p: Params, max_iter: int = 200, tol: float = 1e-5
                ) -> Tuple[np.ndarray, Dict]:
    """
    Solve the nested-logit SUE for a given developer decision z.
    z keys: kN, kL, kH, p, tauL, tauH, qL, qH, alpha
    Returns:
        x : shape (n_g, 4)  columns = [O, N, L, H]
        info : diagnostics (waiting times, profits, utilities, ...)
    """
    n_g = len(p.G)
    # indices: 0=O, 1=N, 2=L, 3=H

    # initial population distribution
    x = np.repeat((p.Mg[:, np.newaxis] / 4), 4, axis=1)
 
    k = {"N": z["kN"], "L": z["kL"], "H": z["kH"]}
    K = sum(k.values())
 
    for it in range(max_iter):
        x_old = x.copy()
 
        # --- demand & service rates ---
        S = {}
        E = {}
        lam = {}
        mu = {}
        s_bar = {}
        for j, idx in [("N", 1), ("L", 2), ("H", 3)]:
            S[j] = np.sum(p.fg * x[:, idx])
            E[j] = np.sum(p.fg * p.eg * x[:, idx])
            lam[j] = S[j] / p.Ho if p.Ho > 0 else 0.0
            # average service time
            num = np.sum(p.fg * x[:, idx] * p.sg)
            den = S[j] if S[j] > 1e-9 else 1e-9
            s_bar[j] = num / den
            mu[j] = 1.0 / s_bar[j] if s_bar[j] > 0 else 1e6
 
        # --- waiting times (Erlang-C) ---     
        W = {}
        for j in ["N", "L", "H"]:
            c = max(int(round(k[j])), 0)
            W[j] = erlang_c(lam[j] / mu[j], c)  # waiting time in hours
        
        # stability soft check
        # should be on the upper level
        # for j in ["N", "L", "H"]:
            # if lam[j] >= k[j] * mu[j] - p.eps_mu:
                # W[j] = 1e3   # large penalty
 
        # --- operating profit ---
        Pi_op = (z["p"] - p.ce - p.cv) * sum(E.values()) - p.cm_coeff * K
 
        # --- profit-sharing returns (present value) ---
        qL, qH = max(z["qL"], 1e-6), max(z["qH"], 1e-6)
        denom = p.omega["L"] * qL + p.omega["H"] * qH
        R_hat = {}
        for t in ["L", "H"]:
            R_hat[t] = p.Ay * z["alpha"] * p.omega[t] * Pi_op / denom
 
        # --- deterministic utilities ---
        V = np.zeros((n_g, 4))   # O, N, L, H
        for g in range(n_g):
            # Outside
            V[g, 0] = -p.Ay * (p.fg[g] * p.eg[g] * p.pg_O[g] + p.Gamma_O[g])
 
            # Non-token
            V[g, 1] = -p.Ay * (
                p.fg[g] * p.eg[g] * z["p"]
                + p.beta_VOT[g] * (p.fg[g] * W["N"] + p.fg[g] * p.Dg[g])
            )
 
            # Low / High token
            for t, col in [("L", 2), ("H", 3)]:
                V[g, col] = (
                    R_hat[t]
                    + p.Z[t]
                    - (z[f"tau{t}"] + p.Ay * p.fg[g] * p.eg[g] * z["p"])
                    - p.Ay * p.beta_VOT[g] * (p.fg[g] * W[t] + p.fg[g] * p.Dg[g])
                )
 
        # --- nested logit probabilities ---
        # token nest inclusive value
        I_gT = np.zeros(n_g)  # token nest inclusive value
        for g in range(n_g):
            I_gT[g] = (1.0 / p.theta) * np.log(
                np.exp(p.theta * V[g, 2]) + np.exp(p.theta * V[g, 3]) + 1e-30
            )
            
        # top-level probabilities (O, N, T)
        c = 1/100000.0  # for numerical stability
        I_g = np.stack([V[:, 0], V[:, 1], I_gT], axis=1) * c
        exp_I = np.exp(I_g)
        P_top = exp_I / exp_I.sum(axis=1, keepdims=True)
 
        # conditional token probabilities
        P_t_given_T = np.zeros((n_g, 2))  # L, H
        for g in range(n_g):
            eL = np.exp(p.theta * V[g, 2])
            eH = np.exp(p.theta * V[g, 3])
            P_t_given_T[g, 0] = eL / (eL + eH)
            P_t_given_T[g, 1] = eH / (eL + eH)
 
        # full choice probabilities
        P = np.zeros((n_g, 4))
        P[:, 0] = P_top[:, 0]          # O
        P[:, 1] = P_top[:, 1]          # N
        P[:, 2] = P_top[:, 2] * P_t_given_T[:, 0]  # L
        P[:, 3] = P_top[:, 2] * P_t_given_T[:, 1]  # H
 
        # update flows
        x = p.Mg[:, None] * P
 
        # enforce token quantity consistency (soft projection)
        # (hard enforcement is done via constraint in upper level)
        if abs(x[:, 2].sum() - z["qL"]) > 1e-3 or abs(x[:, 3].sum() - z["qH"]) > 1e-3:
            # optional mild scaling; usually left to upper-level constraint
            pass
 
        if np.max(np.abs(x - x_old)) < tol:
            break
 
    info = {
        "W": W,
        "Pi_op": Pi_op,
        "R_hat": R_hat,
        "V": V,
        "P": P,
        "lam": lam,
        "mu": mu,
        "S": S,
        "E": E,
        "iterations": it + 1,
        "converged": it < max_iter - 1,
    }
    return x, info
 
 
# ============================================================
# 4. Upper-level objective & constraints
# ============================================================
def developer_profit(z: Dict, x: np.ndarray, info: Dict, p: Params) -> float:
    """Equation (1) of the paper."""
    K = z["kN"] + z["kL"] + z["kH"]
    C_K = p.C_per_charger * K
    token_revenue = z["tauL"] * z["qL"] + z["tauH"] * z["qH"]
    Pi_D = (1 - z["alpha"]) * info["Pi_op"] * p.Ay + (token_revenue - C_K - p.B0)
    return Pi_D
 
 
def check_constraints(z: Dict, x: np.ndarray, info: Dict, p: Params) -> Dict[str, bool]:
    """Return a dictionary of constraint satisfaction flags."""
    K = z["kN"] + z["kL"] + z["kH"]
    C_K = p.C_per_charger * K
    token_rev = z["tauL"] * z["qL"] + z["tauH"] * z["qH"]
 
    cons = {}
    # priority structure
    cons["priority"] = (info["W"]["H"] <= info["W"]["L"] + 1e-6 and
                        info["W"]["L"] <= info["W"]["N"] + 1e-6)
    # queue stability
    cons["stability"] = all(
        info["lam"][j] <= z[f"k{j}"] * info["mu"][j] - p.eps_mu
        for j in ["N", "L", "H"]
    )
    # token quantities match participation
    cons["token_qty"] = (abs(x[:, 2].sum() - z["qL"]) < 1.0 and
                         abs(x[:, 3].sum() - z["qH"]) < 1.0)
    # financing
    cons["fin_min"] = token_rev >= C_K + p.B0 + p.eps_F
    cons["fin_max"] = token_rev <= C_K + p.B0 + p.B_bar
    # positive operating profit
    cons["op_profit"] = info["Pi_op"] >= p.eps_Pi
    # bounds already handled by variable ranges
    return cons
 
 
# ============================================================
# 5. Example usage / simple continuous optimisation
# ============================================================
def evaluate(z_vec: np.ndarray, p: Params, return_detail: bool = False):
    """
    Wrapper that turns a flat continuous vector into z, solves SUE,
    returns negative profit (for minimisers) and constraint violations.
    z_vec = [kN, kL, kH, p, tauL, tauH, qL, qH, alpha]
    """
    z = {
        "kN": max(0, z_vec[0]),
        "kL": max(0, z_vec[1]),
        "kH": max(0, z_vec[2]),
        "p":  np.clip(z_vec[3], p.p_min, p.p_max),
        "tauL": np.clip(z_vec[4], p.tau_min["L"], p.tau_max["L"]),
        "tauH": np.clip(z_vec[5], p.tau_min["H"], p.tau_max["H"]),
        "qL": max(0, z_vec[6]),
        "qH": max(0, z_vec[7]),
        "alpha": np.clip(z_vec[8], 0.0, p.alpha_bar),
    }
    # round charger numbers for evaluation
    z["kN"] = int(round(z["kN"]))
    z["kL"] = int(round(z["kL"]))
    z["kH"] = int(round(z["kH"]))
 
    x, info = compute_SUE(z, p)
    profit = developer_profit(z, x, info, p)
    cons = check_constraints(z, x, info, p)
 
    # simple penalty for violated constraints
    penalty = 0.0
    if not cons["priority"]:
        penalty += 1e5
    if not cons["stability"]:
        penalty += 1e5
    if not cons["token_qty"]:
        # soft: encourage q ≈ participation
        penalty += 100 * (abs(x[:, 2].sum() - z["qL"]) + abs(x[:, 3].sum() - z["qH"]))
    if not cons["fin_min"]:
        penalty += 1e4
    if not cons["fin_max"]:
        penalty += 1e4
    if not cons["op_profit"]:
        penalty += 1e4
 
    if return_detail:
        return -profit + penalty, z, x, info, cons
    return -profit + penalty
 
 
if __name__ == "__main__":
    p = Params()
 
    # ----- Quick evaluation of a hand-crafted decision -----
    z0 = {
        "kN": 80, "kL": 80, "kH": 80,
        "p": 0.28,
        "tauL": 180.0, "tauH": 650.0,
        "qL": 250.0, "qH": 80.0,
        "alpha": 0.45,
    }
    x, info = compute_SUE(z0, p)
    profit = developer_profit(z0, x, info, p)
    cons = check_constraints(z0, x, info, p)
 
    print("=" * 60)
    print("Hand-crafted decision evaluation")
    print("=" * 60)
    print(f"Developer profit (present value): ${profit:,.0f}")
    print(f"Annual operating profit:          ${info['Pi_op']:,.0f}")
    print(f"Waiting times  N/L/H:             {info['W']['N']:.3f} / "
          f"{info['W']['L']:.3f} / {info['W']['H']:.3f} h")
    print(f"Token returns (PV) L/H:           ${info['R_hat']['L']:.1f} / "
          f"${info['R_hat']['H']:.1f}")
    print(f"Participation (total) O/N/L/H:    "
          f"{x[:,0].sum():.0f} / {x[:,1].sum():.0f} / "
          f"{x[:,2].sum():.0f} / {x[:,3].sum():.0f}")
    print("Constraint satisfaction:")
    for k, v in cons.items():
        print(f"  {k:12s}: {v}")
 
    # ----- Continuous optimisation example (SciPy) -----
    print("\n" + "=" * 60)
    print("Running continuous optimisation (differential evolution)")
    print("=" * 60)
 
    bounds = [
        (0, 30),          # kN
        (0, 20),          # kL
        (0, 15),          # kH
        (p.p_min, p.p_max),
        (p.tau_min["L"], p.tau_max["L"]),
        (p.tau_min["H"], p.tau_max["H"]),
        (0, 800),         # qL
        (0, 400),         # qH
        (0.0, p.alpha_bar),
    ]
 
    # For speed we use a modest population; increase for better quality
    res = differential_evolution(
        evaluate,
        bounds,
        args=(p,),
        popsize=100,
        mutation=0.7,
        recombination=0.5,
        seed=42,
        workers=1,
        updating="immediate",
        polish=True,
        maxiter=40,          # increase for production runs
    )
 
    obj, z_opt, x_opt, info_opt, cons_opt = evaluate(res.x, p, return_detail=True)
    print(f"Optimised developer profit: ${-obj:,.0f}")
    print("Optimal (rounded) decision:")
    for k, v in z_opt.items():
        print(f"  {k:6s}: {v}")
    print("Constraint satisfaction at optimum:")
    for k, v in cons_opt.items():
        print(f"  {k:12s}: {v}")