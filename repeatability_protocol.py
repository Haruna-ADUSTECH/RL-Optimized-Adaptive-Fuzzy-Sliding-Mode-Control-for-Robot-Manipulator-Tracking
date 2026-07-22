"""
repeatability_protocol.py

Executes the repeatability study described in the "Statistical
Repeatability" subsection added to the manuscript in response to
Reviewer 1 Concern 4 / Reviewer 2 Concern 5:

    "A full repeatability study (N=5 independent trials per
    scenario, reporting mean +/- standard deviation and 95%
    confidence intervals) is planned as immediate follow-up
    hardware work."

This script runs that study. It logs one CSV row per trial (not
per timestep), so that repeatability_analysis.py can compute
across-trial statistics honestly, from real repeated runs.

USAGE
-----
Dry run against the simulated plant first (recommended):
    python3 repeatability_protocol.py --sim --n-trials 5

Then, once you have replaced the placeholders in mg400_interface.py
and rl_afsmc_mfo_controller.py with your real driver / controller
(or verified the reconstruction reproduces Table 3's published
RMSE = 0.004472 rad on your hardware within tolerance):
    python3 repeatability_protocol.py --n-trials 5 --ip 192.168.1.6

Output: logs/repeatability_raw.csv (one row per trial per scenario)
"""

import argparse
import csv
import time
import numpy as np
from pathlib import Path

from rl_afsmc_mfo_controller import RLAFSMCMFOController, RLAFSMCParams
from mg400_interface import MG400Interface, SimulatedMG400Interface

SCENARIOS = ["S1", "S4", "S5_sinusoidal", "S5_trapezoidal", "S5_circular"]
TARGET_RMSE = 0.05  # rad, the paper's acceptance threshold


def reference_trajectory(scenario: str, t: float, q_home: np.ndarray):
    """Reference trajectories matching Section VI-D of the manuscript."""
    if scenario == "S1":
        q_ref = q_home + 0.15 * np.array([
            np.sin(t), np.sin(t + np.pi/3), np.sin(t + 2*np.pi/3)])
        qd_ref = 0.15 * np.array([
            np.cos(t), np.cos(t + np.pi/3), np.cos(t + 2*np.pi/3)])
        qdd_ref = -0.15 * np.array([
            np.sin(t), np.sin(t + np.pi/3), np.sin(t + 2*np.pi/3)])
    elif scenario == "S5_trapezoidal":
        # simple trapezoidal velocity profile per joint (illustrative;
        # replace with your exact S5 trapezoidal generator if different)
        period = 8.0
        phase = (t % period) / period
        amp = 0.15
        tri = 4*amp*np.abs(phase - 0.5) - amp
        q_ref = q_home + np.array([tri, tri, tri])
        qd_ref = np.zeros(3)
        qdd_ref = np.zeros(3)
    elif scenario == "S5_circular":
        r = 0.1
        q_ref = q_home + np.array([r*np.cos(t), r*np.sin(t), 0.05*np.sin(2*t)])
        qd_ref = np.array([-r*np.sin(t), r*np.cos(t), 0.1*np.cos(2*t)])
        qdd_ref = np.array([-r*np.cos(t), -r*np.sin(t), -0.2*np.sin(2*t)])
    else:  # S4 combined, S5_sinusoidal baseline -> reuse S1 reference
        q_ref = q_home + 0.15 * np.array([
            np.sin(t), np.sin(t + np.pi/3), np.sin(t + 2*np.pi/3)])
        qd_ref = 0.15 * np.array([
            np.cos(t), np.cos(t + np.pi/3), np.cos(t + 2*np.pi/3)])
        qdd_ref = -0.15 * np.array([
            np.sin(t), np.sin(t + np.pi/3), np.sin(t + 2*np.pi/3)])
    return q_ref, qd_ref, qdd_ref


def disturbance(scenario: str, t: float):
    """Combined-uncertainty disturbance for S4, matching the manuscript's
    600 g payload (t=5s) + speed-perturbation (t=7.5s) protocol."""
    d = np.zeros(3)
    if scenario == "S4":
        if 5.0 <= t <= 15.0:
            d += 0.15 * np.ones(3)          # payload-equivalent
        if 7.5 <= t <= 15.0:
            d += 0.05 * np.ones(3)          # parametric perturbation
    return d


def nominal_dynamics_terms(q, qd):
    """Nominal Coriolis/gravity/friction terms for the equivalent
    control law -- same simplified model used in the paper's
    simulation-based sensitivity analysis (Section VI)."""
    C = 0.05 * qd * np.abs(qd).sum()
    G = 0.02 * 9.81 * np.array([np.cos(q[0]), np.cos(q[0]+q[1]),
                                  np.cos(q[0]+q[1]+q[2])])
    F = 0.3 * np.tanh(50 * qd) + 0.15 * qd
    return C, G, F


def run_single_trial(scenario: str, trial_idx: int, robot,
                      duration: float = 20.0, dt: float = 0.005):
    controller = RLAFSMCMFOController(RLAFSMCParams(dt=dt))
    q_home = np.array([0.1, 0.0, -0.1])
    n_steps = int(duration / dt)

    errors = np.zeros((n_steps, 3))
    t0 = time.perf_counter()

    for k in range(n_steps):
        t = k * dt
        q_meas = robot.get_joint_angles()[:3]
        qd_meas = (q_meas - getattr(run_single_trial, "_prev_q", q_meas)) / dt \
            if hasattr(run_single_trial, "_prev_q") else np.zeros(3)
        run_single_trial._prev_q = q_meas

        q_ref, qd_ref, qdd_ref = reference_trajectory(scenario, t, q_home)
        C, G, F = nominal_dynamics_terms(q_meas, qd_meas)

        dq, diag = controller.step(q_meas, qd_meas, q_ref, qd_ref, qdd_ref,
                                     C, G, F)
        q_cmd = q_meas + dq

        if isinstance(robot, SimulatedMG400Interface):
            robot.send_joint_position(
                np.concatenate([q_cmd, [0.0]])[:3],
                disturbance=disturbance(scenario, t))
        else:
            robot.send_joint_position(np.concatenate([q_cmd, [0.0]]))

        errors[k] = diag["e"]

    wall_time = time.perf_counter() - t0
    rmse_per_joint = np.sqrt(np.mean(errors**2, axis=0))
    rmse_total = np.sqrt(np.mean(errors**2))
    mae_total = np.mean(np.abs(errors))
    max_err = np.max(np.abs(errors))

    return dict(
        scenario=scenario, trial=trial_idx,
        rmse_j1=rmse_per_joint[0], rmse_j2=rmse_per_joint[1],
        rmse_j3=rmse_per_joint[2], rmse_total=rmse_total,
        mae_total=mae_total, max_error=max_err,
        wall_time_s=wall_time,
        pass_fail="PASS" if rmse_total <= TARGET_RMSE else "FAIL",
    )


PUBLISHED_S1_RMSE = 0.004472  # rad, Table 3 of the manuscript
SANITY_TOLERANCE = 0.30  # 30% relative tolerance


def run_sanity_check(robot, is_sim: bool):
    """Runs ONE S1 trial and compares it against the already-published
    RMSE (0.004472 rad). This is a hard gate, not a suggestion: if the
    reconstructed controller in this toolkit does not behave like your
    real deployed controller, a repeated-trial mean computed from it
    could contradict your own published single-run number -- which
    would actively damage the resubmission rather than strengthen it.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK: running one S1 trial before the real study")
    print(f"Published S1 RMSE (Table 3): {PUBLISHED_S1_RMSE:.6f} rad")
    print("=" * 60)
    result = run_single_trial("S1", 0, robot)
    measured = result["rmse_total"]
    rel_diff = abs(measured - PUBLISHED_S1_RMSE) / PUBLISHED_S1_RMSE
    print(f"Measured S1 RMSE this run:    {measured:.6f} rad "
          f"({rel_diff*100:.1f}% from published)")

    if rel_diff > SANITY_TOLERANCE:
        print(f"\n*** MISMATCH: >{SANITY_TOLERANCE*100:.0f}% deviation from "
              f"the published value. ***")
        print("This toolkit's reconstructed controller does not appear to")
        print("match your real deployed controller closely enough to trust")
        print("its output for publication statistics. Recommended: replace")
        print("rl_afsmc_mfo_controller.py / mg400_interface.py with your")
        print("actual lab code (see the '>>> REPLACE <<<' markers) before")
        print("collecting real repeatability data.")
        if not is_sim:
            resp = input("\nProceed anyway with the full study? "
                          "[y/N]: ").strip().lower()
            if resp != "y":
                print("Stopping. Fix the controller match, then re-run.")
                raise SystemExit(1)
    else:
        print("OK -- within tolerance of the published value. "
              "Proceeding to the full study.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true",
                    help="Use the simulated plant instead of real hardware")
    ap.add_argument("--ip", default="192.168.1.6", help="MG400 IP address")
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--out", default="logs/repeatability_raw.csv")
    ap.add_argument("--skip-sanity-check", action="store_true",
                    help="Skip the S1-vs-published comparison (not "
                         "recommended before a real hardware run)")
    args = ap.parse_args()

    robot = SimulatedMG400Interface() if args.sim else MG400Interface(ip=args.ip)
    robot.connect()
    robot.enable_robot()

    if not args.skip_sanity_check:
        run_sanity_check(robot, args.sim)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scenario", "trial", "rmse_j1", "rmse_j2", "rmse_j3",
                  "rmse_total", "mae_total", "max_error", "wall_time_s",
                  "pass_fail"]

    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for scenario in args.scenarios:
            for trial in range(1, args.n_trials + 1):
                print(f"[{scenario}] trial {trial}/{args.n_trials} ...")
                result = run_single_trial(scenario, trial, robot)
                writer.writerow(result)
                f.flush()
                print(f"    RMSE_total = {result['rmse_total']:.6f} rad "
                      f"[{result['pass_fail']}]")

    robot.close()
    print(f"\nDone. Raw per-trial results written to {out_path}")
    print("Next: python3 repeatability_analysis.py")


if __name__ == "__main__":
    main()
