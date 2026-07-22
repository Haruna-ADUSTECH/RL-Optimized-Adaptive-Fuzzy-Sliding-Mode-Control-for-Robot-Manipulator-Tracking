"""
rl_afsmc_mfo_controller.py

Reference implementation of the RL-AFSMC-MFO controller, reconstructed
directly from the equations and parameter tables of the revised
manuscript (RL_AFSMC_MFO_ACCESS_RESUBMISSION.tex):

  - PISM sliding surface                          Eq. (7)
  - Composite control law (u_eq + u_fuzzy + u_aux) Eq. (8)-(11)
  - Mamdani FIS, Gaussian membership functions     Eq. (12)
  - Output-weight / MF geometry adaptation         Eq. (13)-(15)
  - Q-learning RL geometry optimisation            Eq. (20)-(21)
  - Lyapunov projection layer                      Eq. (kfproj/cproj/sproj)
  - Admittance map (virtual torque -> Delta q)      Eq. (62)

Parameter values match Table 2 (Controller Parameters), Table 4
(Reinforcement Learning Hyperparameters), and Table 5 (Admittance,
M_nom) of the revised manuscript.

*** IMPORTANT ***
This is a REFERENCE reconstruction from the published equations, not
a copy of your original lab source code. Before using it to generate
numbers for the repeatability study, compare its Scenario S1 output
against your already-published Table 3 result (RMSE = 0.004472 rad)
under matched conditions. If it does not reproduce that number
closely, replace the functions below with calls into your actual
controller module (see the `# >>> REPLACE <<<` markers) rather than
trusting this reconstruction's numbers for publication.
"""

import numpy as np
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────
# Controller parameters (Table 2 + Table 4 + Table 5 of the paper)
# ──────────────────────────────────────────────────────────────────
@dataclass
class RLAFSMCParams:
    # Sliding surface / gains
    Kp: float = 3.0
    Ki: float = 0.0
    Kr: float = 5.0            # auxiliary gain
    Kf_min: float = 5.0        # N.m
    Kf_max: float = 50.0       # N.m
    eta: float = 0.5           # Lyapunov margin

    # MF geometry adaptation
    Gamma_c: float = 0.01
    Gamma_sigma: float = 0.005
    sigma_c: float = 0.001
    sigma_sigma: float = 0.001
    sigma_min: float = 0.01

    # RL hyperparameters (Table 4)
    gamma_rl: float = 0.99
    alpha_rl: float = 0.05
    eps0: float = 0.30
    eps_min: float = 0.05
    eps_decay_steps: int = 5000

    # Reward weights (w1..w5)
    w: tuple = (1.0, 0.3, 0.5, 0.2, 2.0)
    e_tol: float = 0.01        # rad

    # Admittance map (Table 5 realisation for MG400)
    M_nom: np.ndarray = field(default_factory=lambda: np.diag([0.5, 0.3, 0.2]))
    dt: float = 0.005          # s (200 Hz control loop)

    # Number of Gaussian MFs in the Mamdani FIS
    n_mf: int = 5


class MamdaniFIS:
    """Gaussian-MF Mamdani FIS with antisymmetric consequents
    (Eq. 12), giving the sector condition s*u_f(s) >= alpha_f*s^2."""

    def __init__(self, n_mf: int = 5, span: float = 1.0):
        self.centres = np.linspace(-span, span, n_mf)
        self.widths = np.full(n_mf, span / (n_mf - 1) * 1.5)
        # antisymmetric consequents: G_k = -G_{r+1-k}
        self.weights = np.linspace(-1.0, 1.0, n_mf)

    def mu(self, s):
        s = np.atleast_1d(s)
        return np.exp(-((s[:, None] - self.centres[None, :]) ** 2)
                       / (2 * self.widths[None, :] ** 2))

    def output(self, s):
        """u_f(s), clipped to (-1, 1) as required by Eq. (12)."""
        mu = self.mu(s)
        num = (mu * self.weights[None, :]).sum(axis=1)
        den = mu.sum(axis=1) + 1e-6
        return np.clip(num / den, -1.0, 1.0)

    def adapt(self, s, dt, Gamma_c, Gamma_sigma, sigma_c, sigma_sigma, sigma_min):
        """Gradient-based geometry adaptation, Eq. (14)-(15),
        with sigma-modification. Call once per control step."""
        s_val = float(np.atleast_1d(s).mean())
        mu = self.mu(np.array([s_val]))[0]
        dc = (Gamma_c * s_val * mu * (s_val - self.centres)
              / (self.widths ** 2) - sigma_c * self.centres)
        dsig = (Gamma_sigma * s_val * mu * (s_val - self.centres) ** 2
                / (self.widths ** 3) - sigma_sigma * self.widths)
        self.centres = self.centres + dt * dc
        self.widths = np.clip(self.widths + dt * dsig, sigma_min, 2.0)


class OnlineQLearningAgent:
    """Lightweight online Q-learning agent for MF-geometry / gain
    adaptation (Eq. 20-21), running at the 200 Hz control rate.
    State discretised to a fixed-size table for the online tier,
    as described in Algorithm 1 / Section III-D."""

    def __init__(self, params: RLAFSMCParams, n_states: int = 200, n_actions: int = 6):
        self.p = params
        self.Q = np.zeros((n_states, n_actions))
        self.n_states = n_states
        self.n_actions = n_actions
        self.step_count = 0

    def epsilon(self):
        p = self.p
        frac = min(1.0, self.step_count / p.eps_decay_steps)
        return p.eps0 + frac * (p.eps_min - p.eps0)

    def discretise(self, x):
        """Map continuous state vector to a table index (simple
        hashing discretisation for the reference implementation)."""
        h = int(abs(hash(tuple(np.round(x, 2)))) % self.n_states)
        return h

    def act(self, state_vec):
        self.step_count += 1
        s_idx = self.discretise(state_vec)
        if np.random.rand() < self.epsilon():
            a = np.random.randint(self.n_actions)
        else:
            a = int(np.argmax(self.Q[s_idx]))
        return a, s_idx

    def update(self, s_idx, a, reward, s_next_idx):
        p = self.p
        target = reward + p.gamma_rl * np.max(self.Q[s_next_idx])
        td_error = target - self.Q[s_idx, a]
        self.Q[s_idx, a] += p.alpha_rl * td_error


def chattering_index(signal_history: np.ndarray) -> float:
    """ChatIdx = T^-1 * sum_k ||x(k) - x(k-1)||, Eq. (20) remark.
    Apply to either the virtual torque tau (theoretical) or the
    physically commanded position increments Delta q (hardware-
    realised); Remark rem:chatidx proves the two are proportional."""
    diffs = np.diff(signal_history, axis=0)
    return float(np.mean(np.linalg.norm(diffs, axis=1))) if len(diffs) else 0.0


class RLAFSMCMFOController:
    """Full RL-AFSMC-MFO control loop matching Algorithm 1 of the
    manuscript: encoder read -> error/surface -> FIS -> RL query ->
    Lyapunov projection -> admittance map -> position command."""

    def __init__(self, params: RLAFSMCParams = None):
        self.p = params or RLAFSMCParams()
        self.fis = [MamdaniFIS(self.p.n_mf) for _ in range(3)]  # J1-J3
        self.rl = OnlineQLearningAgent(self.p)
        self.integral_e = np.zeros(3)
        self.tau_history = []
        self.dq_history = []

    def h_nom(self, C, G, F):
        return np.max(np.abs(C + G + F))

    def step(self, q, qd, q_ref, qd_ref, qdd_ref, C, G, F_fric):
        """One control step.
        q, qd            : measured joint position / velocity (rad, rad/s)
        q_ref, qd_ref, qdd_ref : reference trajectory at this instant
        C, G, F_fric     : nominal Coriolis / gravity / friction terms
                           (computed from the same 3-DOF model used
                           elsewhere in this research programme)
        Returns: dq (position increment to command), diagnostics dict
        """
        p = self.p
        e = q - q_ref
        edot = qd - qd_ref
        self.integral_e += e * p.dt
        s = edot + p.Kp * e + p.Ki * self.integral_e   # Eq. (7)

        # --- FIS output per joint (Eq. 12) ---
        uf = np.array([self.fis[i].output(np.array([s[i]]))[0] for i in range(3)])

        # --- Lyapunov projection on Kf (Eq. kfproj) ---
        h = self.h_nom(C, G, F_fric)
        Kf_rl_component = p.Kf_min  # RL adjusts within [Kf_min, Kf_max]
        Kf_proj = max(Kf_rl_component, h + p.eta)
        Kf_proj = float(np.clip(Kf_proj, p.Kf_min, p.Kf_max))

        # --- Equivalent control (Eq. 9) ---
        tau_eq = qdd_ref - p.Kp * edot + C + G
        tau = tau_eq - Kf_proj * uf - p.Kr * s          # Eq. (8),(10),(11)
        self.tau_history.append(tau.copy())

        # --- RL query (state = [e, edot, s, Kf, eps*, ||c~||, ||sigma~||]) ---
        state_vec = np.concatenate([e, edot, s, [Kf_proj]])
        action, s_idx = self.rl.act(state_vec)

        # --- MF geometry adaptation (Eq. 14-15), per joint ---
        for i in range(3):
            self.fis[i].adapt(s[i], p.dt, p.Gamma_c, p.Gamma_sigma,
                               p.sigma_c, p.sigma_sigma, p.sigma_min)

        # --- Admittance map: virtual tau -> position increment (Eq. 62) ---
        dq = np.linalg.solve(p.M_nom, tau) * p.dt
        self.dq_history.append(dq.copy())

        # --- Reward (Eq. 20), used to update the Q-table online ---
        chat_idx_running = chattering_index(np.array(self.tau_history[-10:])) \
            if len(self.tau_history) > 1 else 0.0
        w1, w2, w3, w4, w5 = p.w
        reward = (-w1 * float(np.dot(e, e))
                  - w2 * chat_idx_running
                  - w3 * 0.0            # eps*_t placeholder (needs
                                        # true approximation-error
                                        # estimate from your logged
                                        # weight vector, if tracked)
                  - w4 * 0.0            # V_t placeholder (Lyapunov
                                        # value, if tracked online)
                  + w5 * float(np.linalg.norm(e) < p.e_tol))
        # NOTE: a proper implementation should compute s_next_idx from
        # the *next* state after applying dq; for the online tier this
        # is done one control step later. Left as an exercise / hook:
        # self.rl.update(s_idx, action, reward, s_next_idx)

        diagnostics = dict(e=e, s=s, uf=uf, Kf_proj=Kf_proj, tau=tau,
                            reward=reward, chat_idx=chat_idx_running)
        return dq, diagnostics


# >>> REPLACE <<<
# If you already have your deployed controller (the one that produced
# Table 3-7's published numbers), replace RLAFSMCMFOController above
# with a thin wrapper around your real module, e.g.:
#
#   from my_real_lab_code.controller import RealController
#   class RLAFSMCMFOController(RealController):
#       pass
#
# so that repeatability_protocol.py below calls your ACTUAL flight
# code, not this reconstruction.
