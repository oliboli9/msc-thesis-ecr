"""
analysis/stress-fillrate.py
Plot per-iteration decision-window fill rate under a vessel-unavailability
stress test (§6.3.2).

Reads results/stress-experiments/exp_stress_by_iter.csv (produced by
framework/experiment_stress.py) and draws fill rate against rolling-horizon
iteration, one line per scenario: a no-disruption baseline plus one line per
vessel removed for the whole horizon.

Output: msc-thesis-writing/Pictures/exp_stress_fillrate.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt

from plot_style import apply_style, save_fig, MODEL, line_color

apply_style()

CSV_PATH = "results/stress-experiments/exp_stress_by_iter.csv"

# Nicer legend labels + a stable draw order / styling per scenario. Each removed
# vessel is coloured by its own service line; the baseline is neutral grey.
LABELS = {
    "baseline": ("Baseline", MODEL["baseline"], "o", "-"),
    "DET":      ("DET removed (Red, 2148 TEU)",  line_color("Red"),   "s", "--"),
    "JOR":      ("JOR removed (Blue, 1853 TEU)", line_color("Blue"),  "^", "-."),
    "BAK":      ("BAK removed (Green, 880 TEU)", line_color("Green"), "D", ":"),
}

df = pd.read_csv(CSV_PATH)

fig, ax = plt.subplots(figsize=(6.0, 4.0))

for scen in [s for s in LABELS if s in df["scenario"].unique()]:
    g = df[df["scenario"] == scen].sort_values("iteration")
    label, color, marker, ls = LABELS[scen]
    ax.plot(g["iteration"], g["fill_rate_pct"], color=color, marker=marker,
            ls=ls, lw=1.8, ms=6, label=label)

ax.set_xlabel("Rolling-horizon iteration")
ax.set_ylabel("Decision-window fill rate (%)")
ax.set_xticks(sorted(df["iteration"].unique()))
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left", frameon=False, fontsize=9)
fig.tight_layout()

save_fig(fig, "exp_stress_fillrate")
print("\nPer-iteration fill rate (%):")
print(df.pivot(index="iteration", columns="scenario",
               values="fill_rate_pct").to_string())
