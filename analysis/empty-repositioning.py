"""
analysis/empty-repositioning.py
Summarise and visualise empty container repositioning decisions from the model.

Produces:
  1. Sankey-style chord / flow table: which ports send/receive empties via sailing arcs
  2. Bar chart: total empty TEUs repositioned per OD pair, broken down by container spec
  3. Net empty flow per port (surplus = more incoming empties than outgoing)
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, "pipeline")
from config import EPOCH_HOUR_OFFSET, HORIZON_MONDAY, CALENDAR_WEEKS

# ── EPOCH_DT (same derivation as other analysis scripts) ─────────────────────
_v = pd.read_csv("data/raw/Eimskip_voyages.csv", parse_dates=["etdDateTime"])
_mask = False
for _yr, _wk in CALENDAR_WEEKS:
    _mask = _mask | ((_v["year"] == _yr) & (_v["week"] == _wk))
_etds = _v[_mask]["etdDateTime"]
EPOCH_DT = (_etds.min().normalize().to_pydatetime() if not _etds.empty
            else datetime(HORIZON_MONDAY.year, HORIZON_MONDAY.month, HORIZON_MONDAY.day))

def to_dt(h):
    return (EPOCH_DT + timedelta(hours=int(h))).strftime("%d %b %H:%M")

# ── Load flows ────────────────────────────────────────────────────────────────
flows = pd.read_csv("results/model_flows.csv")

empty = flows[flows["Commodity"].str.startswith("EMPTY_")].copy()

# Parse commodity string: EMPTY_<Type>_<Size>_<Owner>
empty[["_", "ContainerType", "ContainerSize", "Owner"]] = (
    empty["Commodity"].str.split("_", expand=True)
)
empty["ContainerSize"] = empty["ContainerSize"].astype(int)
empty["TEU"] = empty["ContainerSize"].apply(lambda s: 1 if s == 20 else 2)
empty["TEU_flow"] = empty["TEU"] * empty["Flow"]

# Separate sailing/transshipment moves from wait arcs
sailing_empty = empty[empty["Vessel"] != "WAIT"].copy()
wait_empty    = empty[empty["Vessel"] == "WAIT"].copy()

print(f"Total empty repositioning arcs (sailing+trans): {len(sailing_empty)}")
print(f"Total TEUs repositioned (sailing+trans): {sailing_empty['TEU_flow'].sum():.0f}")

# ── Figure 1: OD flow table (bar chart) ──────────────────────────────────────
od = (
    sailing_empty
    .groupby(["DepPort", "ArrPort", "ContainerType", "ContainerSize"])["TEU_flow"]
    .sum()
    .reset_index()
)
od["OD"] = od["DepPort"] + " → " + od["ArrPort"]
od["Spec"] = od["ContainerType"] + " " + od["ContainerSize"].astype(str)

od_pivot = od.pivot_table(index="OD", columns="Spec", values="TEU_flow", fill_value=0)

fig1, ax1 = plt.subplots(figsize=(12, max(4, len(od_pivot) * 0.6 + 1)))
od_pivot.plot(kind="barh", ax=ax1, colormap="tab10")
ax1.set_xlabel("TEUs repositioned")
ax1.set_title("Empty container repositioning by OD pair and container spec")
ax1.legend(title="Spec", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.show()

# ── Figure 2: Net empty flow per port ────────────────────────────────────────
# Net = TEUs arriving - TEUs departing on sailing/trans arcs
# Positive = net receiver of empties (shortage at that port)
# Negative = net sender of empties (surplus at that port)

dep_by_port = sailing_empty.groupby("DepPort")["TEU_flow"].sum().rename("TEU_out")
arr_by_port = sailing_empty.groupby("ArrPort")["TEU_flow"].sum().rename("TEU_in")

net = pd.concat([dep_by_port, arr_by_port], axis=1).fillna(0)
net["Net"] = net["TEU_in"] - net["TEU_out"]
net = net.sort_values("Net")

colors = ["#d62728" if v < 0 else "#2ca02c" for v in net["Net"]]

fig2, ax2 = plt.subplots(figsize=(10, max(4, len(net) * 0.5 + 1)))
ax2.barh(net.index, net["Net"], color=colors)
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_xlabel("Net TEUs (arriving − departing empties)")
ax2.set_title("Net empty container flow per port")
surplus_patch = mpatches.Patch(color="#d62728", label="Net sender (surplus empties)")
deficit_patch = mpatches.Patch(color="#2ca02c", label="Net receiver (deficit empties)")
ax2.legend(handles=[surplus_patch, deficit_patch])
plt.tight_layout()
plt.show()

# ── Figure 3: Timeline of empty moves ────────────────────────────────────────
# Each sailing empty move shown as a horizontal bar
if not sailing_empty.empty:
    sailing_empty = sailing_empty.sort_values("DepHour").reset_index(drop=True)
    sailing_empty["Label"] = (
        sailing_empty["DepPort"] + "→" + sailing_empty["ArrPort"]
        + " " + sailing_empty["ContainerType"]
        + " " + sailing_empty["ContainerSize"].astype(str)
        + " (" + sailing_empty["Flow"].astype(int).astype(str) + ")"
    )

    specs = sailing_empty["ContainerType"].unique()
    cmap  = plt.cm.get_cmap("tab10", len(specs))
    spec_color = {s: cmap(i) for i, s in enumerate(specs)}

    unique_moves = sailing_empty.drop_duplicates(subset=["DepPort","ArrPort","DepHour","ArrHour","ContainerType","ContainerSize"])

    fig3, ax3 = plt.subplots(figsize=(12, max(4, len(unique_moves) * 0.55 + 1)))
    yticks, ylabels = [], []
    for idx, (_, row) in enumerate(unique_moves.iterrows()):
        ax3.barh(
            idx,
            row["ArrHour"] - row["DepHour"],
            left=row["DepHour"],
            color=spec_color[row["ContainerType"]],
            edgecolor="white",
            height=0.6,
        )
        label = f"{row['DepPort']}→{row['ArrPort']}  {row['ContainerType']} {int(row['ContainerSize'])} ({int(row['Flow'])} boxes)"
        yticks.append(idx)
        ylabels.append(label)

    ax3.set_yticks(yticks)
    ax3.set_yticklabels(ylabels, fontsize=8)

    # x-axis: epoch hours → date labels
    x_ticks = ax3.get_xticks()
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels([to_dt(h) for h in x_ticks], rotation=30, ha="right", fontsize=8)

    legend_patches = [mpatches.Patch(color=spec_color[s], label=s) for s in specs]
    ax3.legend(handles=legend_patches, title="Container type", loc="lower right")
    ax3.set_xlabel("Time")
    ax3.set_title("Timeline of empty container repositioning moves")
    plt.tight_layout()
    plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n=== Empty repositioning summary (sailing arcs only) ===")
summary = (
    sailing_empty
    .groupby(["DepPort", "ArrPort", "Vessel", "ContainerType", "ContainerSize"])
    .agg(Count=("Flow", "sum"), TEUs=("TEU_flow", "sum"))
    .reset_index()
)
summary["Depart"] = sailing_empty.groupby(
    ["DepPort","ArrPort","Vessel","ContainerType","ContainerSize"]
)["DepHour"].first().values
summary["Depart"] = summary["Depart"].apply(to_dt)
print(summary.sort_values("Depart").to_string(index=False))
