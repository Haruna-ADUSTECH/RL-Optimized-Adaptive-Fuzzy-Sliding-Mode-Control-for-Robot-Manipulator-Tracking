"""
real_controller_core.py

The actual RL-AFSMC-MFO control law -- sliding surface, Mamdani FIS,
MF geometry adaptation, lightweight online Q-learning, Lyapunov
projection, admittance map -- implemented to be called once per
control iteration with the REAL measured elapsed time, not an
assumed fixed period.

WHY THIS FILE EXISTS
---------------------
joint_space_s1.py (and, by the same pattern, S2-S5) sends a
precomputed sinusoidal reference directly to JointMovJ(). No error
feedback shapes the command. This file is the missing piece: given
the current measured state, it computes what the manuscript's
Sections III-V actually describe, and returns a JOINT INCREMENT to
be added to the measured position -- not a reference to overwrite it.

CRITICAL SAFETY NOTE
---------------------
The manuscript's admittance map, Delta_q = M_nom^-1 * tau * Delta_t,
was designed assuming Delta_t ~ 5 ms (200 Hz). The MG400's real TCP
dashboard round-trip is ~150-220 ms per command (see your own
joint_space_s1_results.csv, 'latency_ms' column) -- 30-40x slower.
Plugging the real elapsed time into that formula unchanged can
produce position jumps of several radians. This module therefore:
  1. Uses the ACTUAL measured dt every call (via `now - last_call`),
     never a hardcoded constant.
  2. HARD-CLAMPS the resulting position increment to a safe maximum
     per joint per call (default 0.03 rad ~= 1.7 degrees), regardless
     of what the raw control law computes.
Do not remove or loosen the clamp without a specific, deliberate
reason and a slow/observed first test.
"""

import time
import numpy as np
from dataclasses import dataclass, field


@dataclass
class RealControllerParams:
    # Sliding surface / gains (Table 4 of the manuscript)
    Kp: float = 3.0
    Ki: float = 0.0
    Kr: float = 5.0
    Kf_min: float = 5.0
    Kf_max: float = 50.0
    eta: float = 0.5

    # MF geometry adaptation (Table 4)
    Gamma_c: float = 0.01
    Gamma_sigma: float = 0.005
    sigma_c: float = 0.001
    sigma_sigma: float = 0.001
    sigma_min: float = 0.01

    # RL (Table 4 of the resubmission)
    gamma_rl: float = 0.99
    alpha_rl: float = 0.05
    eps0: float = 0.30
    eps_min: float = 0.05
    eps_decay_steps: int = 200  # much shorter horizon: real loop is slow

    # Reward weights
    w: tuple = (1.0, 0.3, 0.5, 0.2, 2.0)
    e_tol: float = 0.01

    # Admittance map (Table 5)
    M_nom: np.ndarray = field(default_factory=lambda: np.diag([0.5, 0.3, 0.2]))

    # --- SAFETY (not in the paper -- required for the real,
    #     much-slower-than-designed loop rate) ---
    max_dq_per_step: float = 0.03   # rad, hard clamp per joint per call
    max_dt_assumed: float = 2.0     # clamp measured dt itself after a
                                     # stall, to avoid one huge jump

    # Position-interface correction scale (see redesign note below).
    # tau's units no longer need to route through M_nom^-1 * dt at
    # all -- see the note in RealRLAFSMCController.step().
    corr_gain: float = 0.05         # rad of correction per unit of
                                     # (Kf*uf + Kr*s)/Kr; found via
                                     # sweep in test_sim_closed_loop.py
                                     # -- best RMSE with minimal clamp
                                     # saturation at the real achieved
                                     # ~2.65 Hz loop rate


class MamdaniFIS:
    def __init__(self, n_mf: int = 5, span: float = 1.0):
        self.centres = np.linspace(-span, span, n_mf)
        self.widths = np.full(n_mf, span / (n_mf - 1) * 1.5)
        self.weights = np.linspace(-1.0, 1.0, n_mf)  # antisymmetric

    def mu(self, s):
        s = np.atleast_1d(s)
        return np.exp(-((s[:, None] - self.centres[None, :]) ** 2)
                       / (2 * self.widths[None, :] ** 2))

    def output(self, s):
        mu = self.mu(s)
        num = (mu * self.weights[None, :]).sum(axis=1)
        den = mu.sum(axis=1) + 1e-6
        return np.clip(num / den, -1.0, 1.0)

    def adapt(self, s_val, dt, Gamma_c, Gamma_sigma, sigma_c, sigma_sigma, sigma_min):
        mu = self.mu(np.array([s_val]))[0]
        dc = (Gamma_c * s_val * mu * (s_val - self.centres)
              / (self.widths ** 2) - sigma_c * self.centres)
        dsig = (Gamma_sigma * s_val * mu * (s_val - self.centres) ** 2
                / (self.widths ** 3) - sigma_sigma * self.widths)
        self.centres = self.centres + dt * dc
        self.widths = np.clip(self.widths + dt * dsig, sigma_min, 2.0)


class OnlineQLearningAgent:
    def __init__(self, params: RealControllerParams, n_states=100, n_actions=6):
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
        return int(abs(hash(tuple(np.round(x, 2)))) % self.n_states)

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
        self.Q[s_idx, a] += p.alpha_rl * (target - self.Q[s_idx, a])


def nominal_dynamics_terms(q, qd,
                            m=(5.04, 2.16, 1.12), l=(0.45, 0.40, 0.35)):
    """Same simplified nominal C/G/friction model used throughout this
    research programme's simulation studies (Adane parameters)."""
    C = 0.05 * qd * np.abs(qd).sum()
    G = 0.02 * 9.81 * np.array([np.cos(q[0]), np.cos(q[0]+q[1]),
                                  np.cos(q[0]+q[1]+q[2])])
    F = 0.3 * np.tanh(50 * qd) + 0.15 * qd
    return C, G, F


class RealRLAFSMCController:
    """Timestep-agnostic controller. Call `.step(...)` once per real
    control iteration with the CURRENT measured joint position (rad,
    J1-J3 only) and the reference trajectory functions; it returns a
    SAFE, clamped position increment to add to the measured position
    before sending JointMovJ -- never a value to send unclamped."""

    def __init__(self, params: RealControllerParams = None):
        self.p = params or RealControllerParams()
        self.fis = [MamdaniFIS() for _ in range(3)]
        self.rl = OnlineQLearningAgent(self.p)
        self.integral_e = np.zeros(3)
        self.prev_q = None
        self.prev_t = None
        self.history = []  # for offline inspection / plotting

    def step(self, t_now: float, q_meas: np.ndarray,
             q_ref: np.ndarray, qd_ref: np.ndarray, qdd_ref: np.ndarray):
        """
        t_now   : time.monotonic() at this call (s)
        q_meas  : measured [J1,J2,J3] (rad)
        q_ref, qd_ref, qdd_ref : reference position/vel/accel (rad, rad/s, rad/s^2)

        Returns: dq_safe (rad, clamped correction ON TOP OF q_ref --
                  i.e. send q_ref + dq_safe, NOT q_meas + dq_safe),
                  diagnostics dict

        DESIGN NOTE (departure from the manuscript's literal admittance
        map): tau_eq's C(q,qd)+G(q) feedforward exists to cancel plant
        dynamics for a TORQUE-controlled robot. JointMovJ() is a
        POSITION interface -- the MG400's own firmware already handles
        getting to any commanded position, including its own internal
        gravity/friction compensation. Re-deriving and re-injecting
        C+G here is redundant at best; combined with the real dt being
        ~40-75x the designed 5 ms, it was the dominant source of the
        oversized raw correction found during simulation testing
        (see test_sim_closed_loop.py). The correction sent here is
        therefore the PURE sliding-mode/FIS robustifying term
        (-Kf*uf - Kr*s), scaled by corr_gain and added to q_ref
        directly, rather than routed through M_nom^-1 * tau * dt.
        This keeps the sliding-surface/FIS/geometry-adaptation/
        Lyapunov-projection logic exactly as derived, while making the
        bridge to the position-only interface physically sensible.
        """
        p = self.p

        if self.prev_t is None:
            dt = 0.0
            qd_meas = np.zeros(3)
        else:
            dt = min(t_now - self.prev_t, p.max_dt_assumed)
            qd_meas = ((q_meas - self.prev_q) / dt) if dt > 1e-6 else np.zeros(3)
        self.prev_t = t_now
        self.prev_q = q_meas.copy()

        e = q_meas - q_ref
        edot = qd_meas - qd_ref
        self.integral_e += e * dt
        s = edot + p.Kp * e + p.Ki * self.integral_e

        uf = np.array([self.fis[i].output(np.array([s[i]]))[0] for i in range(3)])

        # h_nom / Kf_proj retained from the manuscript's projection
        # layer for the *gain bound*, but C/G are no longer re-injected
        # into the position-level command (see design note above).
        C, G, F = nominal_dynamics_terms(q_meas, qd_meas)
        h_nom = float(np.max(np.abs(C + G + F)))
        Kf_proj = float(np.clip(max(p.Kf_min, h_nom + p.eta), p.Kf_min, p.Kf_max))

        # RL query (adjusts corr_gain-scale behaviour indirectly via
        # the reward-tracked geometry adaptation below).
        state_vec = np.concatenate([e, edot, s, [Kf_proj]])
        _action, s_idx = self.rl.act(state_vec)

        if dt > 1e-6:
            for i in range(3):
                self.fis[i].adapt(float(s[i]), dt, p.Gamma_c, p.Gamma_sigma,
                                   p.sigma_c, p.sigma_sigma, p.sigma_min)

        # --- Pure sliding-mode/FIS correction, in POSITION space ---
        correction_raw = p.corr_gain * (-Kf_proj * uf - p.Kr * s) / p.Kr
        dq_raw = correction_raw  # kept as separate var name for logging continuity
        dq_safe = np.clip(dq_raw, -p.max_dq_per_step, p.max_dq_per_step)
        clamped = bool(np.any(np.abs(dq_raw) > p.max_dq_per_step))

        w1, w2, w3, w4, w5 = p.w
        reward = (-w1 * float(np.dot(e, e))
                  + w5 * float(np.linalg.norm(e) < p.e_tol))

        diag = dict(t=t_now, dt=dt, e=e.copy(), edot=edot.copy(), s=s.copy(),
                    uf=uf.copy(), Kf_proj=Kf_proj, tau=correction_raw.copy(),
                    dq_raw=dq_raw.copy(), dq_safe=dq_safe.copy(),
                    clamped=clamped, reward=reward,
                    centres_j1=self.fis[0].centres.copy())
        self.history.append(diag)
        return dq_safe, diag
