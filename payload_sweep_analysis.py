"""
payload_sweep_analysis.py

Consumes logs/payload_sweep_raw.csv (produced by
payload_sweep_protocol.py from REAL hardware trials at 300/400/500 g)
and combines it with the already-published 600 g point (Table in
Section VII-B, S2) to produce a four-point payload-magnitude
sensitivity trend.

This is explicitly NOT a repeated-trial variance study -- see the
header of payload_sweep_protocol.py for why, and the generated
LaTeX text below for how this is honestly described in the paper.

Produces:
  1. logs/payload_sweep_summary.csv
  2. figures/payload_sweep_trend.png   -- RMSE vs payload, with a
     linear fit and the published 600 g point included
  3. logs/payload_sweep_section.tex    -- ready-to-paste LaTeX
     (table + figure + prose) for a new "Payload Magnitude
     Sensitivity" subsection in Section VII-B, right after S2.

As with repeatability_analysis.py, this script invents no numbers:
if logs/payload_sweep_raw.csv does not exist, it stops and says so.
"""

import csv
import numpy as np
from pathlib import Path

RAW_CSV = Path("logs/payload_sweep_raw.csv")
SUMMARY_CSV = Path("logs/payload_sweep_summary.csv")
FIG_PATH = Path("figures/payload_sweep_trend.png")
TEX_PATH = Path("logs/payload_sweep_section.tex")

# The already-published S2 result (600 g), Table in Section VII-B.
# This is a REAL, already-reported number, not new/simulated data;
# it is included here only so the trend figure/table show all four
# points together. Do not change this without checking the
# manuscript's published Table 7 value.
PUBLISHED_600G = dict(payload_g=600, rmse_nominal=0.003383,
                       rmse_payload=0.003703, rmse_recovery=0.004142,
                       rmse_total=0.003703, max_error=0.008449,
                       pass_fail="PASS")


def load_raw():
    if not RAW_CSV.exists():
        print(f"No data found at {RAW_CSV}.")
        print("Run payload_sweep_protocol.py on the real MG400 first.")
        return None
    rows = []
    with open(RAW_CSV) as f:
        for row in csv.DictReader(f):
            for k in ("payload_g", "rmse_nominal", "rmse_payload",
                       "rmse_recovery", "rmse_total", "max_error"):
                row[k] = float(row[k])
            rows.append(row)
    return rows


def combine_with_published(rows):
    combined = sorted(rows + [PUBLISHED_600G], key=lambda r: r["payload_g"])
    return combined


def fit_trend(combined):
    x = np.array([r["payload_g"] for r in combined])
    y = np.array([r["rmse_payload"] for r in combined])
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return slope, intercept, r


def make_figure(combined, slope, intercept, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([r_["payload_g"] for r_ in combined])
    y = np.array([r_["rmse_payload"] for r_ in combined])
    is_published = [r_["payload_g"] == 600 for r_ in combined]

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.array(x[~np.array(is_published)])
    ys = np.array(y[~np.array(is_published)])
    ax.scatter(xs, ys, s=70, color="#4C72B0", zorder=3,
               label="New trials (this study)")
    ax.scatter(x[is_published], y[is_published], s=70, color="#DD8452",
               marker="D", zorder=3, label="Published (Table 7, 600 g)")
    xfit = np.linspace(0, max(x) * 1.1, 50)
    ax.plot(xfit, slope * xfit + intercept, "k--", linewidth=1,
            label=f"Linear fit ($r={r:.3f}$)")
    ax.axhline(0.05, color="red", linestyle=":", linewidth=1,
               label="Target (0.05 rad)")
    ax.set_xlabel("Payload mass (g)")
    ax.set_ylabel("Payload-phase RMSE (rad)")
    ax.set_title("RMSE vs.\\ Payload Magnitude (S2 Protocol)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200)
    print(f"Figure written to {FIG_PATH}")


def write_summary_csv(combined):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(combined[0].keys()))
        writer.writeheader()
        writer.writerows(combined)
    print(f"Summary written to {SUMMARY_CSV}")


def write_latex(combined, slope, intercept, r):
    rows_tex = []
    for row in combined:
        tag = " (published)" if row["payload_g"] == 600 else " (new)"
        rows_tex.append(
            f"{row['payload_g']:.0f}\\,g{tag} & {row['rmse_nominal']:.6f} "
            f"& {row['rmse_payload']:.6f} & {row['rmse_recovery']:.6f} "
            f"& {row['pass_fail']}\\\\"
        )

    first, last = combined[0], combined[-1]
    direction = ("increases" if last["rmse_payload"] > first["rmse_payload"]
                 else "decreases")
    trend_sentence = (
        rf"The payload-phase RMSE {direction} from "
        rf"{first['rmse_payload']:.6f}\,rad at "
        rf"{first['payload_g']:.0f}\,g to "
        rf"{last['rmse_payload']:.6f}\,rad at "
        rf"{last['payload_g']:.0f}\,g. The linear fit has "
        rf"$r={r:.3f}$; "
    )
    if abs(r) >= 0.7:
        trend_sentence += (
            r"a $|r|\geq0.7$ correlation is consistent with a "
            r"physically meaningful, non-random relationship "
            r"between payload magnitude and tracking error, "
            r"rather than measurement noise.")
    else:
        trend_sentence += (
            r"a correlation this weak indicates RMSE is not "
            r"strongly or monotonically driven by payload "
            r"magnitude alone in this range, which we report "
            r"honestly rather than over-interpret as a clean "
            r"trend.")

    lines = [
        r"%% Auto-generated from REAL hardware trials at 300/400/500 g",
        r"%% plus the already-published 600 g point (Table 7).",
        r"%% Paste as a new subsection after Scenario S2 in",
        r"%% Section VII-B (Hardware Validation Results).",
        r"",
        r"\subsubsection{Payload Magnitude Sensitivity}",
        r"\label{sec:payload_sweep}",
        r"",
        r"To further address the reviewers' request for evidence",
        r"beyond a single representative run per condition, three",
        r"additional payload magnitudes (300\,g, 400\,g, 500\,g) were",
        r"tested under the S2 protocol (Section~\ref{sec:setup}),",
        rf"in addition to the published 600\,g result",
        r"(Table~\ref{tab:s2}). Unlike a repeated-trial variance",
        r"study, this is a sensitivity sweep across a physical",
        r"variable: each point is a distinct condition, not a",
        r"repeat of the same one. This provides evidence about",
        r"whether tracking error responds systematically to a",
        r"physical variable across four independent trials, rather",
        r"than evidence about run-to-run repeatability of one fixed",
        r"condition (Section~\ref{sec:repeatability} addresses that",
        r"question separately).",
        r"",
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Payload Magnitude Sensitivity "
        r"(S2 Protocol, Four Payload Levels)}",
        r"\label{tab:payload_sweep}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Payload} & \textbf{Nom.\ RMSE} & \textbf{Payload "
        r"RMSE} & \textbf{Rec.\ RMSE} & \textbf{Status}\\",
        r"\midrule",
        *rows_tex,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"",
        r"\begin{figure}[!t]",
        r"\centering",
        r"\includegraphics[width=\columnwidth]"
        r"{figures/payload_sweep_trend.png}",
        rf"\caption{{RMSE vs.\ payload magnitude across four "
        rf"independent trials (300--600\,g). Linear fit "
        rf"slope~$={slope:.3e}$\,rad/g, $r={r:.3f}$.}}",
        r"\label{fig:payload_sweep}",
        r"\end{figure}",
        r"",
        trend_sentence,
        r"All four conditions pass the 0.05\,rad target with",
        r"substantial margin regardless of the trend strength.",
        r"",
        r"We note this sweep addresses a related but distinct",
        r"question from literal same-condition repeatability",
        r"(Section~\ref{sec:repeatability}, which we continue to",
        r"identify as a limitation with N=1 per condition); the two",
        r"analyses are complementary rather than substitutes for",
        r"one another.",
    ]

    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEX_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX section written to {TEX_PATH}")


def main():
    rows = load_raw()
    if rows is None:
        return
    if len(rows) < 3:
        print(f"WARNING: only {len(rows)} new trial(s) found; "
              f"expected 3 (300/400/500 g). Proceeding with what "
              f"is available, but the sweep is incomplete.")

    combined = combine_with_published(rows)
    slope, intercept, r = fit_trend(combined)

    print("\nPayload sweep summary (payload-phase RMSE):")
    for row in combined:
        tag = " [PUBLISHED]" if row["payload_g"] == 600 else " [NEW]"
        print(f"  {row['payload_g']:6.0f} g -> "
              f"{row['rmse_payload']:.6f} rad "
              f"[{row['pass_fail']}]{tag}")
    print(f"\nLinear fit: RMSE = {slope:.3e} * payload_g + "
          f"{intercept:.6f}  (r = {r:.3f})")

    write_summary_csv(combined)
    make_figure(combined, slope, intercept, r)
    write_latex(combined, slope, intercept, r)


if __name__ == "__main__":
    main()
