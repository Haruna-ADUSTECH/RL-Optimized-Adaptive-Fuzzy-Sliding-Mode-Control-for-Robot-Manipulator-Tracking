#!/usr/bin/env python3
"""
s1_real_closed_loop.py

The real, closed-loop version of joint_space_s1.py: instead of
sending the sinusoidal reference directly to JointMovJ(), this
computes RealRLAFSMCController.step() from the LIVE measured error
each iteration and sends q_ref + (clamped sliding-mode correction).

Follows the EXACT proven TCP pattern from your working
joint_space_s1.py (new socket per command, port 29999, '\r\n' line
ending, same GetAngle() regex) -- nothing about the communication
layer itself has been changed, only what gets computed before
sending JointMovJ().

SAFETY STAGING -- run in this order, do not skip steps:
  1. --dry-run        Reads real GetAngle() data, computes and PRINTS
                       what it would send, but NEVER calls JointMovJ().
                       Confirms the controller behaves sensibly against
                       your robot's actual current state before it is
                       ever allowed to move anything.
  2. --n-steps 3       A tiny real trial: 3 commanded steps only,
                       ~10-15s including convergence waits. Watch the
                       robot the entire time; keep the E-stop reachable.
  3. --n-steps 20      The full-length trial, once (2) looks correct.

USAGE
-----
    python3 s1_real_closed_loop.py --dry-run
    python3 s1_real_closed_loop.py --n-steps 3
    python3 s1_real_closed_loop.py --n-steps 20 --out closed_loop_s1_trial1.csv
"""

import argparse
import csv
import math
import re
import socket
import threading
import time

import numpy as np
import psutil

from real_controller_core import RealRLAFSMCController, RealControllerParams


# Declared once, from the machine actually running these trials --
# reported alongside LIVE-measured CPU/memory below, per Reviewer 2's
# request for CPU utilization, sampling frequency, and memory
# consumption (not just algorithmic timing in isolation).
PC_SPECS = {
    "Processor": "Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz (2.71GHz boost)",
    "Installed RAM": "8.00 GB (7.88 GB usable)",
    "System type": "64-bit operating system, x64-based processor",
    "OS": "Ubuntu 22.04.5 LTS (WSL2, kernel 6.18.33.2-microsoft-standard-WSL2)",
}


class ResourceMonitor:
    """Samples this process's CPU% and RSS memory in a background
    thread while the real control loop runs, so the reported figures
    reflect ACTUAL measured usage during a real trial, not an
    isolated microbenchmark. Uses psutil against the current PID."""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.proc = psutil.Process()
        self.samples = []  # (t, cpu_percent, rss_mb)
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        self.proc.cpu_percent(interval=None)  # prime the internal counter
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
        cpu_vals = np.array([s[1] for s in self.samples[1:]])  # skip first (priming) sample
        mem_vals = np.array([s[2] for s in self.samples])
        return dict(
            n_samples=len(self.samples),
            cpu_mean=float(cpu_vals.mean()) if len(cpu_vals) else 0.0,
            cpu_max=float(cpu_vals.max()) if len(cpu_vals) else 0.0,
            mem_mean_mb=float(mem_vals.mean()),
            mem_max_mb=float(mem_vals.max()),
        )


class ClosedLoopS1:
    def __init__(self, robot_ip="192.168.1.6", port=29999, dry_run=False):
        self.robot_ip = robot_ip
        self.port = port
        self.dry_run = dry_run

        # Same home position as joint_space_s1.py
        self.J1_home = math.radians(130.87)
        self.J2_home = math.radians(-3.98)
        self.J3_home = math.radians(-12.75)
        self.J4_home = math.radians(141.39)
        self.amp = 0.15

        self.controller = RealRLAFSMCController(RealControllerParams())
        self.log_data = []
        self.start_time = None
        self.resource_summary = None

    # --- identical communication pattern to joint_space_s1.py ---
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
        t_send = time.monotonic()  # monotonic: immune to wall-clock/NTP jumps
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

    def joint_reference(self, t):
        q_ref = np.array([
            self.J1_home + self.amp * math.sin(t),
            self.J2_home + self.amp * math.sin(t + math.pi/3),
            self.J3_home + self.amp * math.sin(t + 2*math.pi/3),
        ])
        qd_ref = self.amp * np.array([
            math.cos(t), math.cos(t + math.pi/3), math.cos(t + 2*math.pi/3),
        ])
        qdd_ref = -self.amp * np.array([
            math.sin(t), math.sin(t + math.pi/3), math.sin(t + 2*math.pi/3),
        ])
        return q_ref, qd_ref, qdd_ref

    def ensure_enabled(self, timeout=10.0):
        """Clears any latched error and enables the robot, then
        verifies RobotMode() actually reports ENABLE/RUNNING before
        proceeding. Neither joint_space_s1.py nor earlier versions of
        this script did this automatically -- it was done manually in
        an earlier session and does not persist across power cycles.
        A disabled robot silently accepts JointMovJ() without moving,
        which looks identical to a working run except q_meas never
        changes -- this check prevents that failure mode."""
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
            if mode in (5, 7):  # ENABLE or RUNNING
                print("Robot enabled.")
                return True
            time.sleep(0.5)

        print(f"ERROR: robot did not reach ENABLE/RUNNING within "
              f"{timeout}s (last mode={mode}). Aborting before sending "
              f"any motion commands.")
        return False

    def run(self, duration=20.0, n_steps=20):
        mode = "DRY-RUN (no commands sent)" if self.dry_run else "REAL HARDWARE"
        print("=" * 60)
        print(f"CLOSED-LOOP S1 -- {mode}")
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

        for i in range(n_steps):
            t = (i / n_steps) * duration
            q_ref, qd_ref, qdd_ref = self.joint_reference(t)

            q_meas = self.get_joint_angles()
            if q_meas is None:
                print(f"[{i+1:02d}/{n_steps}] GetAngle() failed, skipping step")
                continue

            t_now = time.monotonic() - t_monotonic_start
            dq, diag = self.controller.step(t_now, q_meas, q_ref, qd_ref, qdd_ref)
            q_cmd = q_ref + dq

            print(f"[{i+1:02d}/{n_steps}] t={t:.1f}s  "
                  f"e={np.degrees(diag['e'])} deg  "
                  f"dq_raw={np.degrees(diag['dq_raw'])} deg  "
                  f"dq_safe={np.degrees(dq)} deg  "
                  f"clamped={diag['clamped']}")

            if self.dry_run:
                print(f"           would send: J1={math.degrees(q_cmd[0]):.4f}, "
                      f"J2={math.degrees(q_cmd[1]):.4f}, "
                      f"J3={math.degrees(q_cmd[2]):.4f}")
                actual, error = q_meas, 0.0
                latency = 0.0
            else:
                latency, _resp = self.move_joints(q_cmd[0], q_cmd[1], q_cmd[2])
                actual, error = self.wait_convergence(q_cmd)

            elapsed = time.monotonic() - self.start_time
            row = dict(
                time=elapsed, t_ref=t,
                q_ref_deg=list(np.degrees(q_ref)),
                q_meas_deg=list(np.degrees(q_meas)),
                q_cmd_deg=list(np.degrees(q_cmd)),
                dq_raw_deg=list(np.degrees(diag['dq_raw'])),
                dq_safe_deg=list(np.degrees(dq)),
                clamped=diag['clamped'],
                s=list(diag['s']), uf=list(diag['uf']),
                Kf_proj=diag['Kf_proj'],
                latency_ms=latency,
                error_to_cmd_rad=error,
            )
            self.log_data.append(row)

        monitor.stop()
        res = monitor.summary()

        print("\nDone.")
        if res:
            n_ok = sum(1 for r in self.log_data if r is not None)
            achieved_hz = (n_ok / elapsed) if elapsed > 0 else float("nan")
            print("=" * 60)
            print("LIVE RESOURCE USAGE (measured during this trial)")
            print(f"  PC: {PC_SPECS['Processor']}")
            print(f"  RAM: {PC_SPECS['Installed RAM']}")
            print(f"  OS:  {PC_SPECS['OS']}")
            print(f"  CPU utilization -- mean: {res['cpu_mean']:.1f}%  "
                  f"max: {res['cpu_max']:.1f}%  (this Python process, "
                  f"{res['n_samples']} samples @ 0.2s)")
            print(f"  Memory (RSS)    -- mean: {res['mem_mean_mb']:.1f} MB  "
                  f"max: {res['mem_max_mb']:.1f} MB")
            print(f"  Achieved sampling rate: {achieved_hz:.3f} Hz "
                  f"({n_ok} steps / {elapsed:.2f} s)")
            print("=" * 60)
            self.resource_summary = dict(res, achieved_hz=achieved_hz,
                                          n_steps=n_ok, elapsed_s=elapsed)
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
            import json
            res_path = path.rsplit('.', 1)[0] + '_resources.json'
            with open(res_path, 'w') as f:
                json.dump(dict(pc_specs=PC_SPECS, **self.resource_summary), f, indent=2)
            print(f"Saved resource summary to {res_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.6")
    ap.add_argument("--dry-run", action="store_true",
                     help="Read real sensor data and print what would be "
                          "sent, but NEVER call JointMovJ(). Run this first.")
    ap.add_argument("--n-steps", type=int, default=3,
                     help="Number of commanded steps. Default 3 (short "
                          "test). Use 20 only after a 3-step test looks "
                          "correct.")
    ap.add_argument("--duration", type=float, default=None,
                     help="Trial duration in seconds. Defaults to "
                          "n_steps (i.e. ~1s/step, matching the "
                          "original joint_space_s1.py cadence).")
    ap.add_argument("--out", default="closed_loop_s1_result.csv")
    ap.add_argument("--confirm", action="store_true",
                     help="Required for any real-hardware run (not "
                          "--dry-run). Confirms you have already run "
                          "--dry-run, the E-stop is within reach, and "
                          "the workspace is clear. There is no other "
                          "prompt -- passing this flag IS the "
                          "confirmation.")
    args = ap.parse_args()

    duration = args.duration if args.duration is not None else float(args.n_steps)

    if not args.dry_run and not args.confirm:
        print("!" * 60)
        print("REAL HARDWARE MODE requires the --confirm flag.")
        print("Before adding it, confirm: E-stop is within reach,")
        print("workspace is clear, and --dry-run already ran cleanly.")
        print("Re-run with --confirm added, e.g.:")
        print(f"    python3 {__file__} --n-steps {args.n_steps} --confirm")
        print("!" * 60)
        return

    runner = ClosedLoopS1(robot_ip=args.ip, dry_run=args.dry_run)
    runner.run(duration=duration, n_steps=args.n_steps)
    runner.save_csv(args.out)


if __name__ == "__main__":
    main()
