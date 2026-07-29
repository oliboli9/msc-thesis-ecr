"""
analysis/vessel-utilisation.py
Summarise and visualise vessel utilisation from model_flows.csv.

For each sailing arc (excluding wait and transshipment arcs), compute:
  - TEU used (cargo + empty repositioning)
  - Arc capacity
  - Utilisation %
Then produce:
  1. Per-arc utilisation bar chart, grouped by vessel
  2. Summary table (mean/max utilisation per vessel)
  3. CSV: results/vessel_utilisation.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, "pipeline")
from config import EPOCH_HOUR_OFFSET, HORIZON_MONDAY, CALENDAR_WEEKS, ARCS_PATH

_v = pd.read_csv("data/raw/Eimskip_voyages.csv", parse_dates=["etdDateTime"])
_mask = False
for _yr, _wk in CALENDAR_WEEKS:
    _mask = _mask | ((_v["year"] == _yr) & (_v["week"] == _wk))
_etds = _v[_mask]["etdDateTime"]
EPOCH_DT = (_etds.min().normalize().to_pydatetime() if not _etds.empty
            else datetime(HORIZON_MONDAY.year, HORIZON_MONDAY.month, HORIZON_MONDAY.day))
def to_dt(h): return (EPOCH_DT + timedelta(hours=int(h))).strftime("%d %b %H:%M")

# ----------------------------
# Load data
# ----------------------------

flows_df = pd.read_csv("results/model_flows.csv")
arcs_df  = pd.read_csv(ARCS_PATH)

# TEU multiplier from commodity name
def teu_of(commodity):
    if "_20_" in commodity:
        return 1
    return 2  # 40ft and 45ft

flows_df["TEU"] = flows_df["Flow"] * flows_df["Commodity"].apply(teu_of)

# ----------------------------
# Sailing arcs only
# ----------------------------

sailing_arcs = arcs_df[
    (arcs_df["Vessel"] != "WAIT") &
    (~arcs_df["Line"].isin(["TRANS"]))
].copy()

sailing_arcs["arc_label"] = (
    sailing_arcs["DepPort"] + "→" + sailing_arcs["ArrPort"]
    + "\n" + sailing_arcs["Vessel"]
    + " " + sailing_arcs["DepHour"].apply(to_dt)
)

# ----------------------------
# Aggregate flow TEU per sailing arc
# ----------------------------

arc_flow = (
    flows_df[flows_df["DepPort"] != flows_df["ArrPort"]]  # exclude wait arcs
    .groupby(["DepPort", "DepHour", "ArrPort", "ArrHour", "Vessel"])["TEU"]
    .sum()
    .reset_index()
)

util_df = sailing_arcs.merge(
    arc_flow,
    on=["DepPort", "DepHour", "ArrPort", "ArrHour", "Vessel"],
    how="left"
).fillna({"TEU": 0})

util_df["Utilisation"] = (util_df["TEU"] / util_df["Capacity"]).clip(upper=1.0)
util_df["arc_label"] = (
    util_df["DepPort"] + "→" + util_df["ArrPort"]
    + " " + util_df["DepHour"].apply(to_dt)
)

# ----------------------------
# Save CSV
# ----------------------------

out = util_df[["Line", "Vessel", "DepPort", "DepHour", "ArrPort", "ArrHour",
               "Capacity", "TEU", "Utilisation"]].copy()
out["DepDate"] = out["DepHour"].apply(to_dt)
out["ArrDate"] = out["ArrHour"].apply(to_dt)
out["Utilisation%"] = (out["Utilisation"] * 100).round(1)
out = out.sort_values(["Vessel", "DepHour"]).reset_index(drop=True)
out.to_csv("results/vessel_utilisation.csv", index=False)
print(f"Vessel utilisation saved to vessel_utilisation.csv ({len(out)} sailing arcs)")

# ----------------------------
# Summary per vessel
# ----------------------------

summary = (
    util_df.groupby("Vessel")
    .agg(
        Line=("Line", "first"),
        Arcs=("Utilisation", "count"),
        MeanUtil=("Utilisation", "mean"),
        MaxUtil=("Utilisation", "max"),
        TotalTEU=("TEU", "sum"),
        TotalCapacity=("Capacity", "sum"),
    )
    .reset_index()
)
summary["OverallUtil%"] = (summary["TotalTEU"] / summary["TotalCapacity"] * 100).round(1)
summary["MeanUtil%"]    = (summary["MeanUtil"]  * 100).round(1)
summary["MaxUtil%"]     = (summary["MaxUtil"]   * 100).round(1)
summary = summary.sort_values("OverallUtil%", ascending=False)

print("\n=== Vessel Utilisation Summary ===")
print(summary[["Vessel", "Line", "Arcs", "TotalTEU", "TotalCapacity",
               "OverallUtil%", "MeanUtil%", "MaxUtil%"]].to_string(index=False))

# ----------------------------
# Plot 1: utilisation per arc, grouped by vessel
# ----------------------------

vessels = sorted(util_df["Vessel"].unique())
n = len(vessels)
fig, axes = plt.subplots(1, n, figsize=(max(6, 4 * n), 6), sharey=True)
if n == 1:
    axes = [axes]

for ax, vessel in zip(axes, vessels):
    sub = util_df[util_df["Vessel"] == vessel].sort_values("DepHour")
    colors = ["#d9534f" if u > 0.9 else "#f0ad4e" if u > 0.6 else "#5cb85c"
              for u in sub["Utilisation"]]
    y = np.arange(len(sub))
    ax.barh(y, sub["Utilisation"] * 100, color=colors, edgecolor="white")
    ax.axvline(100, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["arc_label"], fontsize=7)
    ax.set_xlabel("Utilisation (%)")
    ax.set_title(f"{vessel}\n{sub['Line'].iloc[0]}", fontsize=9)
    ax.set_xlim(0, 115)

fig.suptitle("Vessel Arc Utilisation", fontsize=13)
plt.tight_layout()
plt.show()

# ----------------------------
# Plot 2: overall utilisation per vessel (bar chart)
# ----------------------------

fig2, ax2 = plt.subplots(figsize=(8, 4))
colors2 = ["#d9534f" if u > 90 else "#f0ad4e" if u > 60 else "#5cb85c"
           for u in summary["OverallUtil%"]]
ax2.bar(summary["Vessel"], summary["OverallUtil%"], color=colors2, edgecolor="white")
ax2.axhline(100, color="black", linewidth=0.8, linestyle="--")
ax2.set_ylabel("Overall Utilisation (%)")
ax2.set_title("Overall Vessel Utilisation (Total TEU / Total Capacity)")
ax2.set_ylim(0, 115)
for _, row in summary.iterrows():
    ax2.text(row["Vessel"], row["OverallUtil%"] + 1, f"{row['OverallUtil%']}%",
             ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.show()
