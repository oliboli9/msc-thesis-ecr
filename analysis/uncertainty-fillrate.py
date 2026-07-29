"""
analysis/uncertainty-fillrate.py
Plot decision-window fill rate vs demand-forecast uncertainty (§6.3.1).

Reads results/uncertainty-experiments/exp_uncertainty.csv (produced by
framework/experiment_uncertainty.py) and draws fill rate against sigma (the
per-week growth of forecast-noise std), with a mean line and a +/- std band
aggregated across instances and seeds. sigma = 0 is the perfect-foresight
baseline.

Output: msc-thesis-writing/Pictures/exp_uncertainty_fillrate.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "results/uncertainty-experiments/exp_uncertainty.csv"
OUT_PATH = "msc-thesis-writing/Pictures/exp_uncertainty_fillrate.pdf"

df = pd.read_csv(CSV_PATH)

# Aggregate across instances and seeds: mean +/- std per sigma.
overall = (df.groupby("sigma")["fill_rate_pct"]
             .agg(["mean", "std"]).reset_index().sort_values("sigma"))
overall["std"] = overall["std"].fillna(0.0)

fig, ax = plt.subplots(figsize=(6.0, 4.0))

# Per-instance mean trajectories (thin, for context).
for inst, g in df.groupby("instance"):
    gm = g.groupby("sigma")["fill_rate_pct"].mean().reset_index().sort_values("sigma")
    ax.plot(gm["sigma"], gm["fill_rate_pct"], color="0.7", lw=1.0, marker="o",
            ms=3, zorder=1, label=None)

# Aggregate mean +/- std band.
ax.fill_between(overall["sigma"], overall["mean"] - overall["std"],
                overall["mean"] + overall["std"], alpha=0.20,
                color="C0", zorder=2, label="$\\pm$1 std")
ax.plot(overall["sigma"], overall["mean"], color="C0", lw=2.0, marker="s",
        ms=6, zorder=3, label="mean across instances")

ax.set_xlabel("Forecast uncertainty $\\sigma$ (per-week std growth)")
ax.set_ylabel("Decision-window fill rate (%)")
ax.set_xticks(sorted(df["sigma"].unique()))
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left", frameon=False, fontsize=9)
fig.tight_layout()

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
print("\nPer-sigma fill rate (mean +/- std):")
print(overall.to_string(index=False))
