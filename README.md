# Real Closed-Loop RL-AFSMC-MFO Controller

This replaces `joint_space_s1.py`'s open-loop "compute reference,
send it directly" pattern with an actual closed loop: measure error,
compute the sliding surface, evaluate the Mamdani FIS, adapt MF
geometry, apply the Lyapunov projection, and send a corrected
position — not the raw reference.

## What changed from the paper's literal design, and why

The manuscript's admittance map (`Δq = M_nom⁻¹·τ·Δt`) was designed
for a 5 ms (200 Hz) control period. Your real TCP dashboard round
trip is ~150–220 ms per command (see your own
`joint_space_s1_results.csv`, `latency_ms` column) — the achievable
loop rate is closer to **2.5–3 Hz**, not 200 Hz. Plugging the real
elapsed time into the original formula unchanged produces position
jumps of several radians (confirmed in simulation testing — see
below). Two changes were necessary:

1. **`τ_eq`'s gravity/Coriolis feedforward is no longer re-injected
   into the position command.** That term exists to cancel dynamics
   for a torque-controlled robot. `JointMovJ()` is a position
   interface — the MG400's own firmware already handles getting to
   any commanded position. Re-deriving that feedforward on top of an
   already-position-controlled interface was the dominant source of
   the oversized correction found during testing.
2. **The controller sends `q_ref + correction`**, where `correction`
   is the pure sliding-mode/FIS robustifying term
   `corr_gain · (−K_f·u_f − K_r·s)/K_r`, hard-clamped to
   ±0.03 rad/step regardless of what the raw computation produces.

The sliding surface, Mamdani FIS, MF geometry adaptation, Lyapunov
projection, and online Q-learning are otherwise exactly as derived
in the manuscript — only the bridge from `τ` to a physical position
command was redesigned for the real (much slower than designed)
achievable rate.

## Files

| File | Purpose |
|---|---|
| `real_controller_core.py` | The actual control law. Timestep-agnostic — uses real measured `dt` every call, never an assumed constant. |
| `test_sim_closed_loop.py` | Stress test against a simulated plant using *your own measured latency distribution* (mean 190ms, std 30ms, fit from your real CSV) — not an idealized fast simulation. |
| `s1_real_closed_loop.py` | The real hardware script. Same TCP pattern as your working `joint_space_s1.py` (same socket handling, same port 29999, same `\r\n`, same `GetAngle()` regex) — only what gets computed before sending changed. |

## Validation already done (before this ever reaches your robot)

Ran `test_sim_closed_loop.py` against a bounded-velocity plant model
driven by your real latency distribution:

- **First attempt diverged** (RMSE ~10⁴², clamp triggering 98% of
  calls) — this is what caught the `τ_eq` feedforward problem above.
- **After the redesign:** RMSE 0.060 rad (nominal) / 0.063 rad (with
  a simulated mid-trial disturbance), clamp engaging only 3–9% of
  calls — meaning the control law is doing the work, not the safety
  clamp. No divergence. See `sim_closed_loop_test.png`.

This is a simplified test plant, not a certified prediction of real
hardware RMSE — the MG400's actual internal servo dynamics will
differ. What it does establish: **the control law is stable and
well-scaled at your real achievable rate**, which is the thing that
needed confirming before it touches the robot at all.

## Required safety staging on real hardware — do not skip steps

```bash
# 1. Dry run: reads REAL sensor data (robot must be powered on and
#    reachable), computes and PRINTS what it would send, but NEVER
#    calls JointMovJ(). Confirms sane behavior against your robot's
#    actual current state before anything is allowed to move.
python3 s1_real_closed_loop.py --dry-run

# 2. Tiny real test: 3 commanded steps only (~10-15s including
#    convergence waits). Watch the robot the entire time. Keep the
#    E-stop within reach.
python3 s1_real_closed_loop.py --n-steps 3

# 3. Full trial, only once step 2 looks correct:
python3 s1_real_closed_loop.py --n-steps 20 --out closed_loop_s1_trial1.csv
```

The script itself requires typing `yes` before any real-hardware run
(not shown for `--dry-run`), and prints every computed correction —
raw and clamped — before it's sent, every single step.

## What to send back

After step 3 succeeds once, send me the resulting CSV (or just the
computed RMSE from it). I'll:
1. Compare it honestly against the open-loop S1 result already in
   the paper (Table 3, 0.004472 rad) — it may not beat that number,
   since the open-loop script's convergence-wait pattern is a
   different (also legitimate) strategy, and the closed-loop version
   is now correctly rate-limited by real TCP latency.
2. If it's stable and reasonable, move to the repeated-trials study
   (`repeatability_protocol.py`) using *this* controller instead of
   my equation-only reconstruction, so the resulting mean±std
   actually characterizes the algorithm the paper describes.
3. Help you decide how to honestly frame this in the manuscript —
   likely acknowledging the achieved rate explicitly rather than the
   originally stated 200 Hz, which is a real, disclosable finding
   about deploying this design on this hardware, not a flaw to hide.
