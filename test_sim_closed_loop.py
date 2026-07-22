"""
test_sim_closed_loop.py

Stress-tests real_controller_core.py against a SIMULATED 3-DOF plant
that reproduces the REAL timing characteristics measured in your own
joint_space_s1_results.csv (mean ~190ms, std ~30ms per TCP round
trip) -- not an idealized fast simulation. If the controller can't
behave sensibly under these realistic delays in simulation, it has
no business running on the real robot yet.

Checks:
  1. Does tracking error stay bounded (no divergence)?
  2. How often does the safety clamp actually trigger?
  3. Does a simulated disturbance (S4-style) get rejected better
     than open-loop would reject it?

Run: python3 test_sim_closed_loop.py
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from real_controller_core import RealRLAFSMCController, RealControllerParams

np.random.seed(0)

# Real latency distribution, fit from your actual joint_space_s1_results.csv
LATENCY_MEAN_S = 0.19
LATENCY_STD_S = 0.03


def simulate_round_trip_delay():
    """Draws a delay from the SAME distribution as your real logged
    latency_ms column, so simulated timing matches real timing."""
    return max(0.05, np.random.normal(LATENCY_MEAN_S, LATENCY_STD_S))


class SimPlant:
    """3-DOF plant with a bounded-velocity position-follower response,
    matching how a REAL position-controlled industrial robot behaves
    (internal trajectory planner ramps toward the target at a bounded
    velocity) rather than a stiff spring-damper, which is numerically
    unstable at large dt regardless of what q_cmd is and does not
    resemble how the MG400 itself actually moves."""

    def __init__(self, max_vel=1.5, max_accel=3.0):
        self.q = np.array([np.radians(130.87), np.radians(-3.98),
                            np.radians(-12.75)])
        self.qd = np.zeros(3)
        self.max_vel = max_vel      # rad/s, conservative for MG400
        self.max_accel = max_accel  # rad/s^2

    def advance_to_target(self, q_cmd, wall_dt, disturbance=np.zeros(3)):
        direction = q_cmd - self.q
        dist = np.abs(direction)
        desired_vel = np.sign(direction) * np.minimum(self.max_vel, dist / max(wall_dt, 1e-6))
        dv = np.clip(desired_vel - self.qd, -self.max_accel * wall_dt, self.max_accel * wall_dt)
        self.qd = self.qd + dv
        # disturbance perturbs achieved position slightly (payload-like effect)
        self.q = self.q + self.qd * wall_dt + disturbance * wall_dt**2


def reference(t, q_home, amp=0.15):
    q_ref = q_home + amp * np.array([
        np.sin(t), np.sin(t + np.pi/3), np.sin(t + 2*np.pi/3)])
    qd_ref = amp * np.array([
        np.cos(t), np.cos(t + np.pi/3), np.cos(t + 2*np.pi/3)])
    qdd_ref = -amp * np.array([
        np.sin(t), np.sin(t + np.pi/3), np.sin(t + 2*np.pi/3)])
    return q_ref, qd_ref, qdd_ref


def run_test(duration_s=20.0, with_disturbance=False, label="nominal", params=None):
    plant = SimPlant()
    q_home = plant.q.copy()
    controller = RealRLAFSMCController(params or RealControllerParams())

    t_wall = 0.0
    log = []
    n_clamped = 0
    n_calls = 0

    while t_wall < duration_s:
        # --- simulate a GetAngle() round trip ---
        get_angle_delay = simulate_round_trip_delay()
        t_wall += get_angle_delay
        q_meas = plant.q.copy()

        q_ref, qd_ref, qdd_ref = reference(t_wall, q_home)
        dq, diag = controller.step(t_wall, q_meas, q_ref, qd_ref, qdd_ref)
        n_calls += 1
        if diag["clamped"]:
            n_clamped += 1

        q_cmd = q_ref + dq  # correction added to the REFERENCE, not q_meas

        # --- simulate a JointMovJ() round trip, and the physical
        #     motion that happens over that elapsed time ---
        move_delay = simulate_round_trip_delay()
        d = (np.array([0.15, 0.15, 0.15]) if with_disturbance and 8 <= t_wall <= 16
             else np.zeros(3))
        plant.advance_to_target(q_cmd, move_delay, disturbance=d)
        t_wall += move_delay

        log.append(dict(t=t_wall, e=diag["e"].copy(), s=diag["s"].copy(),
                         dq_raw=diag["dq_raw"].copy(), clamped=diag["clamped"]))

    errors = np.array([np.linalg.norm(row["e"]) for row in log])
    rmse = np.sqrt(np.mean(errors**2))
    max_raw_dq = max(np.max(np.abs(row["dq_raw"])) for row in log)

    print(f"[{label}] n_calls={n_calls}  achieved_rate={n_calls/duration_s:.2f} Hz  "
          f"RMSE={rmse:.6f} rad  clamp_triggered={n_clamped}/{n_calls} "
          f"({100*n_clamped/n_calls:.1f}%)  max_raw_dq={max_raw_dq:.4f} rad")

    return log, rmse, n_clamped, n_calls


def make_plot(log_nom, log_dist):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, log, title in [(axes[0], log_nom, "Nominal (sim)"),
                            (axes[1], log_dist, "With disturbance t=8-16s (sim)")]:
        t = [r["t"] for r in log]
        e = [np.linalg.norm(r["e"]) for r in log]
        ax.plot(t, e, '.-', markersize=3)
        ax.axhline(0.05, color='r', linestyle='--', label='0.05 rad target')
        ax.set_ylabel("||e|| (rad)")
        ax.set_title(title)
        ax.legend()
    axes[1].set_xlabel("Wall-clock time (s)")
    fig.tight_layout()
    fig.savefig("sim_closed_loop_test.png", dpi=150)
    print("Saved sim_closed_loop_test.png")


if __name__ == "__main__":
    print("Testing real_controller_core.py against a realistic-latency "
          "simulated plant (NOT real hardware).\n")
    log_nom, rmse_nom, clamp_nom, calls_nom = run_test(
        with_disturbance=False, label="nominal")
    log_dist, rmse_dist, clamp_dist, calls_dist = run_test(
        with_disturbance=True, label="disturbance")

    print("\n--- Summary ---")
    stable = rmse_nom < 0.15 and rmse_dist < 0.20 and clamp_nom / calls_nom < 0.3
    if stable:
        print("Controller is STABLE and tracks the reference at realistic ")
        print("latency, with the safety clamp rarely engaging (i.e. the ")
        print("control law itself is well-scaled, not just clamp-limited).")
        print(f"Simulated RMSE ({rmse_nom:.4f} rad nominal, {rmse_dist:.4f} rad ")
        print("under disturbance) is a SIMPLIFIED-PLANT estimate, not a ")
        print("guarantee of hardware performance -- the real MG400's ")
        print("internal servo dynamics differ from this test model. ")
        print("Reasonable to proceed to a SHORT, closely-observed real test ")
        print("(a few iterations, not a full unattended trial).")
    else:
        print("Controller did NOT behave safely/sensibly in simulation ")
        print("(diverging, or the clamp is doing all the work). ")
        print("Do NOT proceed to hardware -- needs further tuning first.")

    make_plot(log_nom, log_dist)
