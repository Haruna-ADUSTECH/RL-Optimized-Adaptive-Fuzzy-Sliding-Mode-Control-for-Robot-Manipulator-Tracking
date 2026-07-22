#!/usr/bin/env python3
"""
sN_real_closed_loop.py

Extends s1_real_closed_loop.py's validated real closed-loop controller
to Scenarios S2-S5, reusing real_controller_core.py UNCHANGED (its
existing dt-clamp, max_dt_assumed=2.0s, already makes a payload-
attachment pause safe -- no controller changes needed) and the exact
same proven TCP pattern (same sockets, same ports, same GetAngle
regex) as the validated S1 script.

SCENARIOS
---------
S2 -- Payload step: 600 g attached at nominal t=5s, removed at t=15s.
      Script pauses with an explicit prompt at each boundary; the
      controller's existing dt-clamp (2.0s max) makes the pause safe
      regardless of how long you take to physically attach/remove
      the mass -- this does NOT distort the reference trajectory,
      which is scheduled by step index, not wall-clock time.
S3 -- Parametric: SpeedFactor(10->30) at t=5s, restored at t=15s.
      Fully automatic (dashboard command), no physical intervention.
S4 -- Combined: 600 g payload at t=5s (prompted) + SpeedFactor(30)
      at t=7.5s (nearest step boundary: applied between steps 7 and
      8; disclosed as an approximation of the original t=7.5s design
      given the ~1s/step schedule granularity), both removed at t=15s.
S5 -- Zero-shot transfer: sinusoidal (baseline, = S1's trajectory),
      trapezoidal, or circular reference, NO retraining between
      trajectory families (each is a fresh process invocation using
      the same fixed default parameters -- nothing is specially
      re-tuned per trajectory).

SAFETY STAGING -- identical requirement to the validated S1 script.
Run in this order for EACH new scenario, do not skip steps:
  1. --scenario S4 --dry-run          (reads real sensors, sends
                                        nothing; for S2/S4 you will
                                        still be prompted at the
                                        payload boundary so you can
                                        rehearse the timing, but no
                                        JointMovJ() is ever sent)
  2. --scenario S4 --n-steps 3 --confirm   (tiny real test)
  3. --scenario S4 --n-steps 20 --confirm  (full trial)

Recommended order given engineering priority: S4 first (your
flagship result currently comes from the unvalidated open-loop
script), then S3 (S4 builds on it), then S2, then S5.

USAGE
-----
    python3 sN_real_closed_loop.py --scenario S4 --dry-run
    python3 sN_real_closed_loop.py --scenario S4 --n-steps 3 --confirm
    python3 sN_real_closed_loop.py --scenario S4 --n-steps 20 --confirm \\
        --out closed_loop_s4_trial1.csv
"""

import argparse
import csv
import json
import math
import re
import socket
import threading
import time

import numpy as np
import psutil

from real_controller_core import RealRLAFSMCController, RealControllerParams

PC_SPECS = {
    "Processor": "Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz (2.71GHz boost)",
    "Installed RAM": "8.00 GB (7.88 GB usable)",
    "System type": "64-bit operating system, x64-based processor",
    "OS": "Ubuntu 22.04.5 LTS (WSL2, kernel 6.18.33.2-microsoft-standard-WSL2)",
}


class ResourceMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.proc = psutil.Process()
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        self.proc.cpu_percent(interval=None)
        t0 = time.monotonic()
        while not self._stop.is_set():
            cpu = self.proc.cpu_percent(interval=None)
            rss_mb = self.proc.memory_info().rss / (1024 ** 2)
            self.samples.append((time.monotonic() - t0, cpu, rss_mb))
            time.sleep(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def summary(self):
        if not self.samples:
            return None
        cpu_vals = np.array([s[1] for s in self.samples[1:]])
        mem_vals = np.array([s[2] for s in self.samples])
        return dict(
            n_samples=len(self.samples),
            cpu_mean=float(cpu_vals.mean()) if len(cpu_vals) else 0.0,
            cpu_max=float(cpu_vals.max()) if len(cpu_vals) else 0.0,
            mem_mean_mb=float(mem_vals.mean()),
            mem_max_mb=float(mem_vals.max()),
        )


class ClosedLoopScenario:
    def __init__(self, scenario, robot_ip="192.168.1.6", port=29999,
                 dry_run=False, payload_mass_g=600.0):
        self.scenario = scenario
        self.robot_ip = robot_ip
        self.port = port
        self.dry_run = dry_run
        self.payload_mass_g = payload_mass_g

        self.J1_home = math.radians(130.87)
        self.J2_home = math.radians(-3.98)
        self.J3_home = math.radians(-12.75)
        self.J4_home = math.radians(141.39)
        self.amp = 0.15

        self.controller = RealRLAFSMCController(RealControllerParams())
        self.log_data = []
        self.start_time = None
        self.resource_summary = None
        self._payload_attached = False
        self._speed_raised = False

    # ── identical TCP pattern to the validated S1 script ──
    def send_command(self, cmd):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((self.robot_ip, self.port))
            s.send((cmd + '\r\n').encode())
            response = s.recv(1024).decode()
            s.close()
            return response
        except Exception as e:
            print(f"[TCP error] {e}")
            return None

    def get_joint_angles(self):
        try:
            response = self.send_command('GetAngle()')
            match = re.search(r'\{([\d.,\-]+)\}', response)
            if match:
                vals = [float(v) for v in match.group(1).split(',')]
                return np.array([math.radians(v) for v in vals[:3]])
        except Exception as e:
            print(f"[GetAngle failed] {e}")
        return None

    def move_joints(self, J1, J2, J3):
        J1_deg, J2_deg, J3_deg = math.degrees(J1), math.degrees(J2), math.degrees(J3)
        J4_deg = math.degrees(self.J4_home)
        cmd = f'JointMovJ({J1_deg:.4f},{J2_deg:.4f},{J3_deg:.4f},{J4_deg:.4f})'
        t_send = time.monotonic()
        response = self.send_command(cmd) if not self.dry_run else "(dry-run, not sent)"
        latency = (time.monotonic() - t_send) * 1000
        return latency, response

    def wait_convergence(self, J_d, threshold=0.02, timeout=8.0):
        start = time.time()
        while time.time() - start < timeout:
            actual = self.get_joint_angles()
            if actual is not None:
                error = float(np.linalg.norm(actual - J_d))
                if error < threshold:
                    return actual, error
            time.sleep(0.1)
        actual = self.get_joint_angles()
        error = float(np.linalg.norm(actual - J_d)) if actual is not None else 999.0
        return actual, error

    def ensure_enabled(self, timeout=10.0):
        if self.dry_run:
            return True
        print("Clearing errors and enabling robot...")
        self.send_command('ClearError()')
        time.sleep(0.3)
        self.send_command('EnableRobot()')
        start = time.time()
        while time.time() - start < timeout:
            resp = self.send_command('RobotMode()')
            m = re.search(r'\{(-?\d+)\}', resp) if resp else None
            mode = int(m.group(1)) if m else None
            print(f"  RobotMode = {mode}")
            if mode in (5, 7):
                print("Robot enabled.")
                return True
            time.sleep(0.5)
        print(f"ERROR: robot did not reach ENABLE/RUNNING within {timeout}s.")
        return False

    # ── reference trajectories for S1-S5 ──
    def reference(self, t):
        q_home = np.array([self.J1_home, self.J2_home, self.J3_home])
        if self.scenario == "S5_trapezoidal":
            # Genuine trapezoidal profile: ramp up, hold flat (plateau),
            # ramp down, hold flat at the other extreme, repeat --
            # distinct from a pure triangle wave (no flat segment).
            period = 8.0
            ramp_frac = 0.25  # each ramp takes 25% of the period
            phase = (t % period) / period
            if phase < ramp_frac:
                tri = -self.amp + 2*self.amp*(phase/ramp_frac)
            elif phase < 0.5:
                tri = self.amp  # plateau (high)
            elif phase < 0.5 + ramp_frac:
                tri = self.amp - 2*self.amp*((phase-0.5)/ramp_frac)
            else:
                tri = -self.amp  # plateau (low)
            q_ref = q_home + np.array([tri, tri, tri])
            return q_ref, np.zeros(3), np.zeros(3)
        elif self.scenario == "S5_circular":
            r = 0.1
            q_ref = q_home + np.array([r*np.cos(t), r*np.sin(t), 0.05*np.sin(2*t)])
            qd_ref = np.array([-r*np.sin(t), r*np.cos(t), 0.1*np.cos(2*t)])
            qdd_ref = np.array([-r*np.cos(t), -r*np.sin(t), -0.2*np.sin(2*t)])
            return q_ref, qd_ref, qdd_ref
        else:  # S1, S2, S3, S4, S5_sinusoidal all share the base sinusoid
            q_ref = q_home + self.amp * np.array([
                math.sin(t), math.sin(t + math.pi/3), math.sin(t + 2*math.pi/3)])
            qd_ref = self.amp * np.array([
                math.cos(t), math.cos(t + math.pi/3), math.cos(t + 2*math.pi/3)])
            qdd_ref = -self.amp * np.array([
                math.sin(t), math.sin(t + math.pi/3), math.sin(t + 2*math.pi/3)])
            return q_ref, qd_ref, qdd_ref

    # ── scenario-specific perturbation hooks, called once per step ──
    def apply_perturbation_hooks(self, t, step_idx):
        """Prompts for payload attach/detach (S2, S4) and issues
        SpeedFactor changes (S3, S4) at the right schedule points.
        Uses the NOMINAL step-index-based t, matching how the
        reference trajectory itself is scheduled -- a pause here does
        NOT distort q_ref for subsequent steps."""
        if self.scenario in ("S2", "S4") and t >= 5.0 and not self._payload_attached:
            self._payload_attached = True
            if not self.dry_run:
                input(f"\n>>> [{self.scenario}] t={t:.1f}s: ATTACH the "
                      f"{self.payload_mass_g:.0f} g payload now, then "
                      f"press Enter to continue "
                      f"(controller's dt-clamp makes this pause safe)...")
            else:
                print(f"[DRY-RUN] {self.scenario}: would prompt to attach "
                      f"{self.payload_mass_g:.0f} g payload here (t={t:.1f}s)")

        if self.scenario == "S3" and t >= 5.0 and not self._speed_raised:
            self._speed_raised = True
            self.send_command('SpeedFactor(30)')
            print(f"[{self.scenario}] t={t:.1f}s: SpeedFactor(30) sent")

        if self.scenario == "S4" and t >= 7.0 and not self._speed_raised:
            self._speed_raised = True
            self.send_command('SpeedFactor(30)')
            print(f"[{self.scenario}] t={t:.1f}s: SpeedFactor(30) sent "
                  f"(nearest step to the designed t=7.5s boundary)")

        if self.scenario in ("S2", "S4") and t >= 15.0 and self._payload_attached \
                and step_idx == self._payload_removal_step:
            if not self.dry_run:
                input(f"\n>>> [{self.scenario}] t={t:.1f}s: REMOVE the "
                      f"{self.payload_mass_g:.0f} g payload now, then "
                      f"press Enter to continue...")
            else:
                print(f"[DRY-RUN] {self.scenario}: would prompt to remove "
                      f"{self.payload_mass_g:.0f} g payload here (t={t:.1f}s)")

        if self.scenario in ("S3", "S4") and t >= 15.0 and self._speed_raised \
                and step_idx == self._speed_restore_step:
            self.send_command('SpeedFactor(10)')
            print(f"[{self.scenario}] t={t:.1f}s: SpeedFactor(10) restored")

    def run(self, duration=20.0, n_steps=20):
        # precompute the step index nearest each removal/restore boundary
        self._payload_removal_step = round(15.0 / duration * n_steps)
        self._speed_restore_step = round(15.0 / duration * n_steps)

        mode = "DRY-RUN (no commands sent)" if self.dry_run else "REAL HARDWARE"
        print("=" * 60)
        print(f"CLOSED-LOOP {self.scenario} -- {mode}")
        print(f"Steps: {n_steps}, duration: {duration}s")
        print("=" * 60)

        if not self.ensure_enabled():
            return []

        if not self.dry_run:
            self.send_command('SpeedFactor(10)')
            time.sleep(0.5)

        monitor = ResourceMonitor(interval=0.2)
        monitor.start()

        self.start_time = time.monotonic()
        t_monotonic_start = self.start_time
        seen_boundaries = set()

        for i in range(n_steps):
            t = (i / n_steps) * duration

            # fire perturbation hooks exactly once per boundary crossing
            key_5 = t >= 5.0 and 5.0 not in seen_boundaries
            key_7 = t >= 7.0 and 7.0 not in seen_boundaries
            key_15 = t >= 15.0 and 15.0 not in seen_boundaries
            if key_5: seen_boundaries.add(5.0)
            if key_7: seen_boundaries.add(7.0)
            if key_15: seen_boundaries.add(15.0)
            self.apply_perturbation_hooks(t, i)

            q_ref, qd_ref, qdd_ref = self.reference(t)
            q_meas = self.get_joint_angles()
            if q_meas is None:
                print(f"[{i+1:02d}/{n_steps}] GetAngle() failed, skipping step")
                continue

            t_now = time.monotonic() - t_monotonic_start
            dq, diag = self.controller.step(t_now, q_meas, q_ref, qd_ref, qdd_ref)
            q_cmd = q_ref + dq

            print(f"[{i+1:02d}/{n_steps}] t={t:.1f}s  "
                  f"e={np.degrees(diag['e'])} deg  "
                  f"dq_safe={np.degrees(dq)} deg  clamped={diag['clamped']}")

            if self.dry_run:
                actual, error = q_meas, 0.0
                latency = 0.0
            else:
                latency, _ = self.move_joints(q_cmd[0], q_cmd[1], q_cmd[2])
                actual, error = self.wait_convergence(q_cmd)

            elapsed = time.monotonic() - self.start_time
            self.log_data.append(dict(
                scenario=self.scenario,
                payload_mass_g=(self.payload_mass_g
                                if self.scenario in ("S2", "S4") else None),
                time=elapsed, t_ref=t,
                q_ref_deg=list(np.degrees(q_ref)),
                q_meas_deg=list(np.degrees(q_meas)),
                q_cmd_deg=list(np.degrees(q_cmd)),
                dq_safe_deg=list(np.degrees(dq)),
                clamped=diag['clamped'], latency_ms=latency,
                error_to_cmd_rad=error,
            ))

        monitor.stop()
        res = monitor.summary()
        print("\nDone.")
        if res:
            n_ok = len(self.log_data)
            achieved_hz = (n_ok / elapsed) if elapsed > 0 else float("nan")
            self.resource_summary = dict(res, achieved_hz=achieved_hz,
                                          n_steps=n_ok, elapsed_s=elapsed)
            print(f"Achieved rate: {achieved_hz:.3f} Hz  "
                  f"CPU mean: {res['cpu_mean']:.1f}%  "
                  f"Mem mean: {res['mem_mean_mb']:.1f} MB")
        return self.log_data

    def save_csv(self, path):
        if not self.log_data:
            print("No data to save.")
            return
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.log_data[0].keys())
            writer.writeheader()
            writer.writerows(self.log_data)
        print(f"Saved {len(self.log_data)} rows to {path}")
        if self.resource_summary:
            res_path = path.rsplit('.', 1)[0] + '_resources.json'
            with open(res_path, 'w') as f:
                json.dump(dict(pc_specs=PC_SPECS, **self.resource_summary), f, indent=2)
            print(f"Saved resource summary to {res_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True,
                     choices=["S2", "S3", "S4", "S5_sinusoidal",
                              "S5_trapezoidal", "S5_circular"])
    ap.add_argument("--ip", default="192.168.1.6")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--payload-mass", type=float, default=600.0,
                     help="Payload mass in grams for S2/S4 prompts and "
                          "CSV logging (default 600g, matching the "
                          "originally published S2/S4 protocol). Use "
                          "e.g. --payload-mass 300 for a payload-"
                          "magnitude sweep instead of same-condition "
                          "repeats.")
    args = ap.parse_args()

    duration = args.duration if args.duration is not None else float(args.n_steps)
    out = args.out or f"closed_loop_{args.scenario.lower()}_result.csv"

    if not args.dry_run and not args.confirm:
        print("!" * 60)
        print(f"REAL HARDWARE MODE ({args.scenario}) requires --confirm.")
        if args.scenario in ("S2", "S4"):
            print(f"This scenario will PAUSE mid-trial and prompt you to")
            print(f"physically attach/remove a {args.payload_mass:.0f} g payload.")
        print(f"Re-run with --confirm added.")
        print("!" * 60)
        return

    runner = ClosedLoopScenario(args.scenario, robot_ip=args.ip,
                                 dry_run=args.dry_run,
                                 payload_mass_g=args.payload_mass)
    runner.run(duration=duration, n_steps=args.n_steps)
    runner.save_csv(out)


if __name__ == "__main__":
    main()
