"""
mg400_interface.py

Thin TCP/IP wrapper around the Dobot MG400 dashboard/feedback ports,
matching the interface documented in the manuscript:

  - Port 29999 : JointMovJ() dashboard command port
  - Port 30004 : real-time feedback stream, GetAngle() equivalent
  - Firmware   : 1.6.0.0, must be in TCP/IP secondary development mode

This module provides only the communication plumbing. It intentionally
mirrors your existing lab setup (ROS2 Humble, Python 3.10, Ubuntu 22.04
under WSL2) so the repeatability_protocol.py script can be dropped in
next to your real driver code with minimal changes.

*** IMPORTANT ***
This is a reference/scaffold implementation of the TCP dashboard
protocol sufficient to run the repeatability trials. If you already
have a working driver (the one used to produce Tables 3-7), prefer
importing and calling that directly -- just implement the same
`get_joint_angles()` / `send_joint_position(q_cmd)` interface used
below so the rest of the toolkit does not need to change.
"""

import socket
import time
import numpy as np


class MG400Interface:
    def __init__(self, ip: str = "192.168.1.6",
                 dashboard_port: int = 29999,
                 feedback_port: int = 30004,
                 timeout: float = 2.0):
        self.ip = ip
        self.dashboard_port = dashboard_port
        self.feedback_port = feedback_port
        self.timeout = timeout
        self.dash_sock = None
        self.fb_sock = None

    def connect(self):
        self.dash_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.dash_sock.settimeout(self.timeout)
        self.dash_sock.connect((self.ip, self.dashboard_port))

        self.fb_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.fb_sock.settimeout(self.timeout)
        self.fb_sock.connect((self.ip, self.feedback_port))
        print(f"[MG400Interface] Connected to {self.ip} "
              f"(dashboard:{self.dashboard_port}, feedback:{self.feedback_port})")

    def close(self):
        if self.dash_sock:
            self.dash_sock.close()
        if self.fb_sock:
            self.fb_sock.close()

    def _send_cmd(self, cmd: str) -> str:
        self.dash_sock.sendall((cmd + "\n").encode())
        return self.dash_sock.recv(1024).decode(errors="ignore")

    def enable_robot(self):
        return self._send_cmd("EnableRobot()")

    def get_joint_angles(self) -> np.ndarray:
        """Equivalent of GetAngle(): parses the 1 kHz feedback packet
        and returns the current joint angles [J1, J2, J3, J4] in
        radians. Replace the parsing below with your actual feedback
        packet layout (the MG400 real-time feedback protocol uses a
        fixed-size binary packet; see Dobot's TCP/IP secondary
        development manual for the exact byte offsets)."""
        raw = self.fb_sock.recv(1440)  # MG400 feedback packet size
        # >>> REPLACE <<< with your actual struct.unpack offsets
        # for the joint-angle field in the feedback packet.
        # Placeholder: assumes angles were parsed into `angles_deg`.
        angles_deg = np.zeros(4)  # TODO: parse `raw` here
        return np.deg2rad(angles_deg)

    def send_joint_position(self, q_cmd: np.ndarray):
        """Equivalent of JointMovJ(j1, j2, j3, j4) -- q_cmd in radians
        for J1-J3 (J4 held fixed per the manuscript's hardware scope)."""
        q_deg = np.rad2deg(q_cmd)
        j4_fixed = 0.0
        cmd = f"JointMovJ({q_deg[0]:.4f},{q_deg[1]:.4f},{q_deg[2]:.4f},{j4_fixed:.4f})"
        return self._send_cmd(cmd)


class SimulatedMG400Interface:
    """Drop-in replacement for MG400Interface that simulates the
    3-DOF Euler-Lagrange plant instead of talking to real hardware.
    Use this to dry-run and debug the repeatability protocol before
    connecting to the physical robot, and to sanity-check that this
    reconstruction reproduces the published S1 RMSE (0.004472 rad)
    before trusting it for real trials."""

    def __init__(self, m=(5.04, 2.16, 1.12), l=(0.45, 0.40, 0.35), dt=0.005):
        self.m = np.array(m)
        self.l = np.array(l)
        self.dt = dt
        self.q = np.array([0.1, 0.0, -0.1])
        self.qd = np.zeros(3)
        self._rng = np.random.default_rng()

    def connect(self):
        print("[SimulatedMG400Interface] Simulated plant ready (no hardware).")

    def close(self):
        pass

    def enable_robot(self):
        return "OK (simulated)"

    def _mass_matrix(self):
        c2, c3 = np.cos(self.q[1]), np.cos(self.q[2])
        m, l = self.m, self.l
        M = np.diag([
            m[0]*l[0]**2/3 + m[1]*(l[0]**2 + l[1]**2*c2**2/3)
            + m[2]*(l[0]**2+l[1]**2*c2**2+l[2]**2*c3**2/3),
            m[1]*l[1]**2/3 + m[2]*(l[1]**2 + l[2]**2*c3**2/3),
            m[2]*l[2]**2/3
        ])
        return M + 0.05*np.eye(3)

    def get_joint_angles(self) -> np.ndarray:
        noise = self._rng.normal(0, 1e-4, size=3)  # encoder-level noise
        return np.concatenate([self.q + noise, [0.0]])

    def send_joint_position(self, q_cmd: np.ndarray, disturbance=np.zeros(3)):
        M = self._mass_matrix()
        # Simple servo dynamics toward the commanded position increment,
        # perturbed by any injected disturbance (for S2/S3/S4 scenarios).
        accel = np.linalg.solve(M, (q_cmd - self.q) * 50.0 - disturbance)
        self.qd = self.qd + accel * self.dt
        self.q = self.q + self.qd * self.dt
        return "OK (simulated)"
