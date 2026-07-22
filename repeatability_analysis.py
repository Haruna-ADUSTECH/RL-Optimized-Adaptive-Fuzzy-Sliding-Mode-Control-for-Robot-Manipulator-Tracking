"""
repeatability_analysis.py

Consumes logs/repeatability_raw.csv (produced by
repeatability_protocol.py from REAL repeated hardware trials) and
computes the statistics the reviewers asked for: mean +/- standard
deviation and 95% confidence intervals, per scenario.

Produces:
  1. logs/repeatability_summary.csv   -- one row per scenario
  2. figures/repeatability_errorbars.png -- RMSE with error bars
  3. logs/repeatability_table.tex     -- ready-to-paste LaTeX table
     for the "Statistical Repeatability" subsection of the manuscript

This script performs NO simulation and invents NO numbers: it only
summarises whatever is actually present in the CSV. If you run it
before collecting real trials, it will simply report "no data" --
by design, so that the manuscript is never populated with anything
that was not genuinely measured.
"""

import csv
import numpy as np
from pathlib import Path
from scipy import stats

RAW_CSV = Path("logs/repeatability_raw.csv")
SUMMARY_CSV = Path("logs/repeatability_summary.csv")
FIG_PATH = Path("figures/repeatability_errorbars.png")
TEX_PATH = Path("logs/repeatability_table.tex")


def load_raw():
    if not RAW_CSV.exists():
        print(f"No data found at {RAW_CSV}.")
        print("Run repeatability_protocol.py first to collect real trials.")
        return None
    rows = []
    with open(RAW_CSV) as f:
        for row in csv.DictReader(f):
            row["rmse_total"] = float(row["rmse_total"])
            row["mae_total"] = float(row["mae_total"])
            row["max_error"] = float(row["max_error"])
            rows.append(row)
    return rows


def summarise(rows):
    scenarios = sorted(set(r["scenario"] for r in rows))
    summary = []
    for sc in scenarios:
        vals = np.array([r["rmse_total"] for r in rows if r["scenario"] == sc])
        n = len(vals)
        mean = vals.mean()
        std = vals.std(ddof=1) if n > 1 else 0.0
        if n > 1:
            ci = stats.t.interval(0.95, n - 1, loc=mean,
                                   scale=std / np.sqrt(n))
        else:
            ci = (mean, mean)
        n_pass = sum(1 for r in rows if r["scenario"] == sc
                     and r["pass_fail"] == "PASS")
        summary.append(dict(
            scenario=sc, n=n, mean_rmse=mean, std_rmse=std,
            ci95_lo=ci[0], ci95_hi=ci[1],
            n_pass=n_pass, n_total=n,
        ))
        flag = "" if n >= 5 else "  ** WARNING: n<5, see caveat below **"
        print(f"{sc:16s}  n={n:2d}  mean={mean:.6f}  std={std:.6f}  "
              f"95% CI=[{ci[0]:.6f}, {ci[1]:.6f}]  "
              f"pass {n_pass}/{n}{flag}")
    return summary


def write_summary_csv(summary):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSummary written to {SUMMARY_CSV}")


def make_errorbar_figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = [s["scenario"] for s in summary]
    means = [s["mean_rmse"] for s in summary]
    stds = [s["std_rmse"] for s in summary]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(scenarios))
    ax.bar(x, means, yerr=stds, capsize=5, color="#4C72B0",
           edgecolor="black", alpha=0.85)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1,
               label="Target (0.05 rad)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20)
    ax.set_ylabel("RMSE (rad)")
    ax.set_title(f"Repeatability Across Trials (mean \u00B1 std, "
                 f"n={summary[0]['n_total']} per scenario)")
    ax.legend()
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200)
    print(f"Figure written to {FIG_PATH}")


def write_latex_table(summary):
    lines = [
        r"%% Auto-generated from REAL repeated-trial data.",
        r"%% Paste into the Statistical Repeatability subsection.",
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Repeatability Study: Mean $\pm$ Std.\ RMSE "
        rf"over $N={summary[0]['n_total']}$ Independent Trials}}",
        r"\label{tab:repeatability}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{Mean $\pm$ Std.\ (rad)} "
        r"& \textbf{95\% CI} & \textbf{Pass Rate}\\",
        r"\midrule",
    ]
    for s in summary:
        lines.append(
            f"{s['scenario']} & {s['mean_rmse']:.6f} $\\pm$ "
            f"{s['std_rmse']:.6f} & "
            f"[{s['ci95_lo']:.6f}, {s['ci95_hi']:.6f}] & "
            f"{s['n_pass']}/{s['n_total']}\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEX_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table written to {TEX_PATH}")


def main():
    rows = load_raw()
    if rows is None:
        return
    summary = summarise(rows)
    if any(s["n"] < 5 for s in summary):
        print("\nNOTE: at least one scenario has fewer than 5 trials.")
        print("The manuscript commits to N=5; collect more trials before")
        print("citing these numbers as the completed repeatability study.")
    write_summary_csv(summary)
    make_errorbar_figure(summary)
    write_latex_table(summary)


if __name__ == "__main__":
    main()
