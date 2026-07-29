"""
analysis/visualise-comparison.py

Visualise the comparison between integrated and sequential models
using pre-computed metrics from results/comparison/metrics.csv.

Usage:
    python analysis/visualise-comparison.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Load metrics ──────────────────────────────────────────────────────────────

metrics_df = pd.read_csv("results/comparison/metrics.csv")
int_df = metrics_df[metrics_df["model"] == "integrated"].reset_index(drop=True)
seq_df = metrics_df[metrics_df["model"] == "sequential"].reset_index(drop=True)

labels = int_df["iteration"].tolist()
x = np.arange(len(labels))
w = 0.32

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# ── 1. Fill rate per iteration ────────────────────────────────────────────────
ax = axes[0, 0]
ax.bar(x - w/2, int_df["fill_rate"] * 100, w, label="Integrated", color="#2c5f99")
ax.bar(x + w/2, seq_df["fill_rate"] * 100, w, label="Sequential", color="#b85c1a")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Fill rate (%)")
ax.set_title("Demand Fill Rate per Iteration")
ax.set_ylim(bottom=max(0, min(int_df["fill_rate"].min(), seq_df["fill_rate"].min()) * 100 - 5))
ax.legend(fontsize=8)
for i_x in x:
    ax.text(i_x - w/2, int_df["fill_rate"].iloc[i_x] * 100 + 0.3,
            f"{int_df['fill_rate'].iloc[i_x]*100:.1f}", ha="center", fontsize=7)
    ax.text(i_x + w/2, seq_df["fill_rate"].iloc[i_x] * 100 + 0.3,
            f"{seq_df['fill_rate'].iloc[i_x]*100:.1f}", ha="center", fontsize=7)

# ── 2. Unfulfilled containers per iteration ───────────────────────────────────
ax = axes[0, 1]
ax.bar(x - w/2, int_df["unfulfilled"], w, label="Integrated", color="#2c5f99")
ax.bar(x + w/2, seq_df["unfulfilled"], w, label="Sequential", color="#b85c1a")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Unfulfilled containers")
ax.set_title("Unfulfilled Demand per Iteration")
ax.legend(fontsize=8)

# ── 3. Empty repositioning moves per iteration ───────────────────────────────
ax = axes[1, 0]
ax.bar(x - w/2, int_df["empty_moves"], w, label="Integrated", color="#2c5f99")
ax.bar(x + w/2, seq_df["empty_moves"], w, label="Sequential", color="#b85c1a")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Empty container moves")
ax.set_title("Empty Repositioning per Iteration")
ax.legend(fontsize=8)

# ── 4. Cumulative fulfilled ──────────────────────────────────────────────────
ax = axes[1, 1]
cum_int = int_df["fulfilled"].cumsum()
cum_seq = seq_df["fulfilled"].cumsum()
cum_dem = int_df["demand"].cumsum()
ax.plot(x, cum_dem, "k--", label="Demand", linewidth=1.5)
ax.plot(x, cum_int, "o-", label="Integrated", color="#2c5f99", linewidth=2)
ax.plot(x, cum_seq, "s-", label="Sequential", color="#b85c1a", linewidth=2)
ax.fill_between(x, cum_seq, cum_int, alpha=0.15, color="#2c5f99")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Cumulative containers")
ax.set_title("Cumulative Fulfillment")
ax.legend(fontsize=8)

# Add total annotation
total_int = int(int_df["fulfilled"].sum())
total_seq = int(seq_df["fulfilled"].sum())
total_dem = int(int_df["demand"].sum())
ax.annotate(f"Int: {total_int}/{total_dem} ({total_int/total_dem:.1%})\n"
            f"Seq: {total_seq}/{total_dem} ({total_seq/total_dem:.1%})\n"
            f"Gap: {total_int - total_seq:+d}",
            xy=(x[-1], cum_int.iloc[-1]),
            xytext=(-80, -50), textcoords="offset points",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))

fig.suptitle("Integrated vs Sequential Model Comparison", fontsize=13, y=0.98)
plt.tight_layout()
out = "visualisation/integrated-vs-sequential.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved to {out}")

# Print summary
print(f"\nIntegrated: {total_int}/{total_dem} ({total_int/total_dem:.1%})")
print(f"Sequential: {total_seq}/{total_dem} ({total_seq/total_dem:.1%})")
print(f"Improvement: {total_int - total_seq:+d} containers "
      f"({(total_int - total_seq) / total_dem * 100:+.2f} pp)")

plt.show()
