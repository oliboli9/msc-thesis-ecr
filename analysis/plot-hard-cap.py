"""
Figure for §6.1.4 (maximum delay constraints).

Reads results/delay-experiments/exp_D_max_delay.csv and draws a dual-axis chart:
total lateness as bars (left axis) and decision-window fill rate as a line
(right axis), across the soft penalty (d_max = inf) and hard caps of 7/3/1 days.
Each bar is annotated with the number of containers the cap rejects.

    python analysis/plot-hard-cap.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import apply_style, save_fig, MODEL

apply_style()

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV  = REPO_ROOT / "results" / "delay-experiments" / "exp_D_max_delay.csv"

BAR_COLOR  = MODEL["integrated"]   # lateness bars
LINE_COLOR = MODEL["sequential"]   # fill-rate bars

df = pd.read_csv(CSV)
# Order categories loosest to tightest: inf (encoded as 0) first, then
# decreasing cap length.
df["ord"] = df["max_delay_days"].apply(lambda d: float("-inf") if d == 0 else -d)
df = df.sort_values("ord").reset_index(drop=True)

labels = [r"$\infty$" if d == 0 else str(int(d)) for d in df["max_delay_days"]]
rejected = (df["demand"] - df["fulfilled"]).round().astype(int)
x = range(len(df))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))

# ── (a) total lateness ────────────────────────────────────────────────────────
ax1.bar(x, df["total_lateness_cd"], width=0.6, color=BAR_COLOR, zorder=2)
ax1.set_xlabel(r"Maximum delay cap $d_{\max}$ (days)")
ax1.set_ylabel("Total lateness (container-days)")
ax1.set_xticks(list(x))
ax1.set_xticklabels(labels)
ymax = df["total_lateness_cd"].max()
ax1.set_ylim(0, ymax * 1.08)
ax1.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

# ── (b) fill rate ─────────────────────────────────────────────────────────────
ax2.bar(x, df["fill_rate_pct"], width=0.6, color=LINE_COLOR, zorder=2)
ax2.set_xlabel(r"Maximum delay cap $d_{\max}$ (days)")
ax2.set_ylabel("Fill rate (%)")
ax2.set_xticks(list(x))
ax2.set_xticklabels(labels)
ax2.set_ylim(98.5, 100.1)
ax2.set_yticks([98.6, 98.8, 99.0, 99.2, 99.4, 99.6, 99.8, 100.0])
ax2.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

fig.tight_layout()
save_fig(fig, "exp_D_hard_cap")
