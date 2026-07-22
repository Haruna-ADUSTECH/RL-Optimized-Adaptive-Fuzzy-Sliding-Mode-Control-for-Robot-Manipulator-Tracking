"""
payload_sweep_protocol.py

Runs the S2 payload-step protocol (Section VI-D of the manuscript)
at three NEW payload magnitudes -- 300 g, 400 g, 500 g -- on the
real MG400, using the exact same timing and reference trajectory
as the already-published 600 g result, so the four points
(300/400/500/600 g) form one directly comparable sweep.

WHAT THIS ADDS vs. WHAT REVIEWERS ASKED FOR
---------------------------------------------
Reviewer 1 Concern 4 / Reviewer 2 Concern 5 literally ask for
repeated trials of the SAME condition (mean +/- std, N=3-5).
This script instead runs three DIFFERENT conditions once each.

That is a deliberate, disclosed choice, not a substitution by
stealth: a payload-magnitude sweep (i) uses genuinely new hardware
time productively, (ii) directly tests whether RMSE responds
sensibly/monotonically to a physical variable (strong evidence
against a cherry-picked single run, and speaks to Reviewer 2's
"explain the significance of the results" concern), and (iii) is
reported as exactly what it is in the manuscript -- a sensitivity
trend across four payload magnitudes, not an inter-trial variance
study. See payload_sweep_analysis.py's generated LaTeX text for
the precise wording used.

If the Associate Editor still wants literal same-condition repeats
after seeing this, repeatability_protocol.py (same toolkit) is
ready to run that separately -- the two are complementary, not
either/or.

PROTOCOL (matches Table in Section VI-D, S2 definition exactly)
-----------------------------------------------------------------
  0--5 s   : nominal (no payload)
  5--15 s  : payload attached
  15--20 s : payload removed (recovery)

Reference trajectory is the same phase-shifted sinusoid used for
every scenario in the paper:
  q_d(t) = q_h + 0.15*[sin(t), sin(t+pi/3), sin(t+2pi/3)]  rad

USAGE
-----
Dry run first (simulated plant, sanity-checks the script only --
these numbers are NOT real and must never be cited):
    python3 payload_sweep_protocol.py --sim

Real hardware run (attach each mass physically before its trial;
the script pauses and prompts you between masses):
    python3 payload_sweep_protocol.py --ip 192.168.1.6
"""

import argparse
import csv
import time
import numpy as np
from pathlib import Path

from rl_afsmc_mfo_controller import RLAFSMCMFOController, RLAFSMCParams
from mg400_interface import MG400Interface, SimulatedMG400Interface

TARGET_RMSE = 0.05  # rad, paper's acceptance threshold
PAYLOADS_G = [300, 400, 500]  # grams; 600 g already published (S2)

# Linear payload -> simulated-disturbance mapping, calibrated so
# 600 g reproduces roughly the published S2 degradation pattern.
# ONLY used in --sim mode; real trials use the actual physical mass.
SIM_DISTURBANCE_PER_GRAM = 0.15 / 600.0


def reference_trajectory(t: float, q_home: np.ndarray):
    q_ref = q_home + 0.15 * np.array([
        np.sin(t), np.sin(t + np.pi/3), np.sin(t + 2*np.pi/3)])
    qd_ref = 0.15 * np.array([
        np.cos(t), np.cos(t + np.pi/3), np.cos(t + 2*np.pi/3)])
    qdd_ref = -0.15 * np.array([
        np.sin(t), np.sin(t + np.pi/3), np.sin(t + 2*np.pi/3)])
    return q_ref, qd_ref, qdd_ref


def nominal_dynamics_terms(q, qd):
    C = 0.05 * qd * np.abs(qd).sum()
    G = 0.02 * 9.81 * np.array([np.cos(q[0]), np.cos(q[0]+q[1]),
                                  np.cos(q[0]+q[1]+q[2])])
    F = 0.3 * np.tanh(50 * qd) + 0.15 * qd
    return C, G, F


def run_payload_trial(payload_g: float, robot, is_sim: bool,
                       duration: float = 20.0, dt: float = 0.005):
    controller = RLAFSMCMFOController(RLAFSMCParams(dt=dt))
    q_home = np.array([0.1, 0.0, -0.1])
    n_steps = int(duration / dt)

    errors = np.zeros((n_steps, 3))
    phase_errors = {"nominal": [], "payload": [], "recovery": []}
    prev_q = None
    t0 = time.perf_counter()

    for k in range(n_steps):
        t = k * dt
        q_meas = robot.get_joint_angles()[:3]
        qd_meas = (q_meas - prev_q) / dt if prev_q is not None else np.zeros(3)
        prev_q = q_meas

        q_ref, qd_ref, qdd_ref = reference_trajectory(t, q_home)
        C, G, F = nominal_dynamics_terms(q_meas, qd_meas)
        dq, diag = controller.step(q_meas, qd_meas, q_ref, qd_ref, qdd_ref,
                                     C, G, F)
        q_cmd = q_meas + dq

        if is_sim:
            d = (np.ones(3) * payload_g * SIM_DISTURBANCE_PER_GRAM
                 if 5.0 <= t <= 15.0 else np.zeros(3))
            robot.send_joint_position(q_cmd, disturbance=d)
        else:
            robot.send_joint_position(np.concatenate([q_cmd, [0.0]]))

        e = diag["e"]
        errors[k] = e
        phase = "nominal" if t < 5.0 else ("payload" if t < 15.0 else "recovery")
        phase_errors[phase].append(e)

    wall_time = time.perf_counter() - t0
    rmse_total = float(np.sqrt(np.mean(errors**2)))
    phase_rmse = {p: float(np.sqrt(np.mean(np.array(v)**2))) if v else float("nan")
                  for p, v in phase_errors.items()}

    return dict(
        payload_g=payload_g,
        rmse_nominal=phase_rmse["nominal"],
        rmse_payload=phase_rmse["payload"],
        rmse_recovery=phase_rmse["recovery"],
        rmse_total=rmse_total,
        max_error=float(np.max(np.abs(errors))),
        wall_time_s=wall_time,
        pass_fail="PASS" if rmse_total <= TARGET_RMSE else "FAIL",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true",
                    help="Simulated plant -- sanity check ONLY, not real data")
    ap.add_argument("--ip", default="192.168.1.6")
    ap.add_argument("--payloads", nargs="+", type=float, default=PAYLOADS_G,
                    help="Payload masses in grams, e.g. --payloads 300 400 500")
    ap.add_argument("--out", default="logs/payload_sweep_raw.csv")
    args = ap.parse_args()

    is_sim = args.sim
    robot = SimulatedMG400Interface() if is_sim else MG400Interface(ip=args.ip)
    robot.connect()
    robot.enable_robot()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["payload_g", "rmse_nominal", "rmse_payload", "rmse_recovery",
                  "rmse_total", "max_error", "wall_time_s", "pass_fail"]
    write_header = not out_path.exists()

    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for payload in args.payloads:
            if not is_sim:
                input(f"\n>>> Attach {payload:.0f} g payload to the MG400 "
                      f"end-effector now, then press Enter to start "
                      f"the 20 s trial...")
            print(f"Running payload = {payload:.0f} g ...")
            result = run_payload_trial(payload, robot, is_sim)
            writer.writerow(result)
            f.flush()
            print(f"    RMSE_total = {result['rmse_total']:.6f} rad "
                  f"[{result['pass_fail']}]  "
                  f"(nominal={result['rmse_nominal']:.6f}, "
                  f"payload={result['rmse_payload']:.6f}, "
                  f"recovery={result['rmse_recovery']:.6f})")
            if not is_sim:
                input(">>> Remove the payload, then press Enter to "
                      "continue to the next mass...")

    robot.close()
    tag = " (SIMULATED -- sanity check only, DO NOT cite)" if is_sim else ""
    print(f"\nDone{tag}. Raw results written to {out_path}")
    print("Next: python3 payload_sweep_analysis.py")


if __name__ == "__main__":
    main()
