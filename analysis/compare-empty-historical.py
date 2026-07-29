"""
analysis/compare-empty-historical.py
Compare model empty carrier repositioning to historical empty carrier movements.

Usage:
    python analysis/compare-empty-historical.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from framework.runner import (START_YEAR, START_WEEK, NUM_ITERATIONS,
                               DECISION_HORIZON_WEEKS, advance_by_weeks)

# ── Build iteration labels ───────────────────────────────────────────────────
iter_labels = []
y, w = START_YEAR, START_WEEK
for i in range(NUM_ITERATIONS):
    iter_labels.append((y, w, f"{y}w{w}"))
    if i < NUM_ITERATIONS - 1:
        y, w = advance_by_weeks(y, w, DECISION_HORIZON_WEEKS)

# ── Load model flows for all iterations ──────────────────────────────────────
model_frames = []
hist_weeks = []
for yr, wk, label in iter_labels:
    path = f"results/model-flows/{label}.csv"
    df = pd.read_csv(path)
    df["_year"] = yr
    df["_week"] = wk
    model_frames.append(df)
    hist_weeks.append((yr, wk))

flows = pd.concat(model_frames, ignore_index=True)

# Extract empty carrier sailing moves (exclude WAIT arcs and transshipment arcs)
empty = flows[
    flows["Commodity"].str.startswith("EMPTY_") &
    (flows["Vessel"] != "WAIT") &
    (flows["DepPort"] != flows["ArrPort"])
].copy()

if not empty.empty:
    parts = empty["Commodity"].str.split("_", expand=True)
    empty["ContainerType"] = parts[1]
    empty["ContainerSize"] = parts[2].astype(float).astype(int)
    empty["Owner"] = parts[3]
    empty_carrier = empty[empty["Owner"] == "Carrier"].copy()
    empty_carrier["TEU"] = empty_carrier["ContainerSize"].apply(lambda s: 1 if s == 20 else 2)
    empty_carrier["TEU_flow"] = empty_carrier["TEU"] * empty_carrier["Flow"]
    model_od = (
        empty_carrier
        .groupby(["DepPort", "ArrPort"])["TEU_flow"]
        .sum()
        .reset_index()
        .rename(columns={"TEU_flow": "Model_TEU"})
    )
    model_od["OD"] = model_od["DepPort"] + " → " + model_od["ArrPort"]
else:
    empty_carrier = pd.DataFrame(columns=["TEU_flow"])
    model_od = pd.DataFrame(columns=["DepPort", "ArrPort", "Model_TEU", "OD"])

# ── Load historical empty carrier movements ──────────────────────────────────
hist_df = pd.read_csv("data/clean/Eimskip_data_final.csv")

# Filter to same weeks, empty, carrier only
_mask = False
for yr, wk in hist_weeks:
    _mask = _mask | ((hist_df["Year"] == yr) & (hist_df["Week"] == wk))
hist_filtered = hist_df[
    _mask &
    (hist_df["Full/Empty"] == "E") &
    (hist_df["Owner"] == "CARRIER") &
    (hist_df["ContainerSize"].isin([20, 40, 45]))
].copy()

hist_filtered["TEU"] = hist_filtered["ContainerSize"].apply(lambda s: 1 if s == 20 else 2)
hist_filtered["TEU_flow"] = hist_filtered["TEU"] * hist_filtered["Total"]

hist_od = (
    hist_filtered
    .groupby(["Load 1", "Discharge 1"])["TEU_flow"]
    .sum()
    .reset_index()
    .rename(columns={"Load 1": "DepPort", "Discharge 1": "ArrPort", "TEU_flow": "Hist_TEU"})
)
hist_od["OD"] = hist_od["DepPort"] + " → " + hist_od["ArrPort"]

# ── Merge ────────────────────────────────────────────────────────────────────
merged = pd.merge(hist_od[["OD", "Hist_TEU"]], model_od[["OD", "Model_TEU"]],
                  on="OD", how="outer").fillna(0)
merged = merged.sort_values("Hist_TEU", ascending=True).reset_index(drop=True)

print(f"\nHistorical empty carrier moves: {len(hist_filtered)} rows, "
      f"{hist_filtered['TEU_flow'].sum():.0f} TEUs")
print(f"Model empty carrier moves: {len(empty_carrier)} rows, "
      f"{empty_carrier['TEU_flow'].sum():.0f} TEUs")

# ── Summary table ────────────────────────────────────────────────────────────
print("\n=== Empty Carrier Repositioning: Historical vs Model (TEUs) ===")
table = merged.sort_values("Hist_TEU", ascending=False)
print(table.to_string(index=False))

# ── Figure 1: Grouped bar chart ─────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, max(4, len(merged) * 0.45 + 1)))
y_pos = np.arange(len(merged))
bar_h = 0.35

ax1.barh(y_pos + bar_h / 2, merged["Hist_TEU"], bar_h,
         label="Historical", color="#4c72b0", alpha=0.85)
ax1.barh(y_pos - bar_h / 2, merged["Model_TEU"], bar_h,
         label="Model", color="#dd8452", alpha=0.85)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(merged["OD"], fontsize=8)
ax1.set_xlabel("TEUs repositioned (empty carrier containers)")
ax1.set_title("Empty Carrier Repositioning: Historical vs Model")
ax1.legend()
plt.tight_layout()
fig1.savefig("visualisation/empty-comparison-bar.png", dpi=160,
             bbox_inches="tight", facecolor="white")
plt.show()

# ── Figure 2: Scatter plot ──────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 7))
ax2.scatter(merged["Hist_TEU"], merged["Model_TEU"],
            s=60, alpha=0.7, edgecolors="black", linewidth=0.5)

# Reference line (y = x)
max_val = max(merged["Hist_TEU"].max(), merged["Model_TEU"].max()) * 1.1
ax2.plot([0, max_val], [0, max_val], "k--", alpha=0.4, label="y = x")

# Label points
for _, row in merged.iterrows():
    if row["Hist_TEU"] > 0 or row["Model_TEU"] > 0:
        ax2.annotate(row["OD"], (row["Hist_TEU"], row["Model_TEU"]),
                     fontsize=6, alpha=0.7,
                     xytext=(4, 4), textcoords="offset points")

ax2.set_xlabel("Historical TEUs")
ax2.set_ylabel("Model TEUs")
ax2.set_title("Empty Carrier Repositioning by OD Pair")
ax2.legend()
ax2.set_xlim(-5, max_val)
ax2.set_ylim(-5, max_val)
plt.tight_layout()
fig2.savefig("visualisation/empty-comparison-scatter.png", dpi=160,
             bbox_inches="tight", facecolor="white")
plt.show()
