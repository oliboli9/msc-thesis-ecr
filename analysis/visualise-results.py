import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from datetime import timedelta

from plot_style import apply_style, TYPE_COLORS as TYPE_COLORS_AGG

apply_style()

# ── Toggle ───────────────────────────────────────────────────────────────────
# True → group into Dry/Reefer; False → show exact codes.
# Override per-run with the AGGREGATE_TYPES=1 environment variable.
AGGREGATE_TYPES = os.environ.get("AGGREGATE_TYPES", "0") == "1"

# Results base dir; override with RESULTS_DIR to read an isolated run.
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results").rstrip("/")

# ── Runner parameters (mirror runner.py) ─────────────────────────────────────
sys.path.insert(0, ".")
from framework.runner import (START_YEAR, START_WEEK, PLANNING_HORIZON_WEEKS,
                    DECISION_HORIZON_WEEKS, NUM_ITERATIONS,
                    build_calendar_weeks, compute_epoch_dt, _iso_monday,
                    advance_by_weeks)

voyages_df = pd.read_csv("data/raw/Eimskip_voyages.csv",
                         parse_dates=["etaDateTime", "etdDateTime"])

# Build labels for all iterations
iter_labels = []
y, w = START_YEAR, START_WEEK
for i in range(NUM_ITERATIONS):
    iter_labels.append(f"{y}w{w}")
    if i < NUM_ITERATIONS - 1:
        y, w = advance_by_weeks(y, w, DECISION_HORIZON_WEEKS)

LAST_LABEL = iter_labels[-1]

# Epoch for the last iteration (for datetime formatting)
calendar_weeks = build_calendar_weeks(y, w, PLANNING_HORIZON_WEEKS)
horizon_monday = _iso_monday(y, w)
EPOCH_DT = compute_epoch_dt(voyages_df, calendar_weeks, horizon_monday)

def to_dt(h):
    return (EPOCH_DT + timedelta(hours=int(h))).strftime("%d %b %Y %H:%M")


def classify_type(code: str) -> str:
    """3rd character == 'R' → Reefer, else → Dry. Untyped stock → Unknown."""
    code = str(code).strip()
    if code in ("", "nan", "Unknown"):
        return "Unknown"
    return "Reefer" if len(code) >= 3 and code[2].upper() == "R" else "Dry"


def maybe_aggregate(df, col="ContainerType"):
    """If AGGREGATE_TYPES is on, map container type codes to Dry/Reefer."""
    if AGGREGATE_TYPES:
        df[col] = df[col].apply(classify_type)
    return df


print(f"Visualising {NUM_ITERATIONS} iterations: {', '.join(iter_labels)}")
print(f"  Container type display: {'Dry/Reefer' if AGGREGATE_TYPES else 'exact codes'}")

# ----------------------------
# Load data
# ----------------------------

# All decision-horizon fulfillment files (original demands only)
all_fulfillment = []
for label in iter_labels:
    df = pd.read_csv(f"{RESULTS_DIR}/demand-fulfilment/{label}.csv")
    df = df[~df["Commodity"].str.startswith(("CARRY_", "TLEG_"))]
    df["Iteration"] = label
    all_fulfillment.append(df)
fulfillment_df = pd.concat(all_fulfillment, ignore_index=True)

# Initial stock (iteration 0 input) and final stock (last iteration output)
initial_stock_df = pd.read_csv(f"data/raw/Stock_0101{START_YEAR}.csv")
final_stock_df = pd.read_csv(f"{RESULTS_DIR}/stock/{LAST_LABEL}.csv")

# Last iteration flows for vessel summary
flows_df = pd.read_csv(f"{RESULTS_DIR}/model-flows/{LAST_LABEL}.csv")

# ----------------------------
# Vessel movement summary (last iteration)
# ----------------------------

def summarize_vessel_moves(flows_df, include_waiting=False, include_empty=True):
    df = flows_df.copy()
    if not include_waiting:
        df = df[df["DepPort"] != df["ArrPort"]]
    if not include_empty:
        df = df[~df["Commodity"].str.startswith("EMPTY")]

    summary = (
        df
        .groupby(["Vessel", "DepPort", "ArrPort"])["Flow"]
        .sum()
        .reset_index()
        .sort_values(["Vessel", "DepPort", "ArrPort"])
    )
    print(f"\n=== Vessel Movement Summary ({LAST_LABEL}) ===")
    if not include_empty:
        print("(Full containers only)\n")
    print(summary.to_string(index=False))
    print("\nTotal Containers Flowed (over all arcs):", df["Flow"].sum())
    return summary

summarize_vessel_moves(flows_df)

# ----------------------------
# Helpers
# ----------------------------

from matplotlib.patches import Patch


def extract_container_type(commodity_id):
    parts = str(commodity_id).split("_")
    return parts[1] if len(parts) >= 3 else None


# ----------------------------
# Container distribution by port  (initial vs final)
# ----------------------------

# Initial stock. Drop rows with no ContainerType (ownerless, untyped artifact rows
# in the stock file) so they are excluded from both panels. Both the disaggregated
# groupby and the Dry/Reefer aggregation then ignore them, keeping the two plots
# consistent.
init_ports = initial_stock_df[initial_stock_df["Location"] != "VESSEL"].copy()
init_ports = init_ports[init_ports["ContainerType"].notna()]
maybe_aggregate(init_ports)
init_ports = init_ports[init_ports["ContainerType"] != "Unknown"]
init_agg = (
    init_ports.groupby(["Location", "ContainerType"])["count"]
    .sum().reset_index()
    .rename(columns={"Location": "Port", "count": "Initial"})
)

# Final stock (model output is fully typed; the filter is a safety net).
final_ports = final_stock_df[final_stock_df["Location"] != "VESSEL"].copy()
final_ports = final_ports[final_ports["ContainerType"].notna()]
maybe_aggregate(final_ports)
final_ports = final_ports[final_ports["ContainerType"] != "Unknown"]
final_agg = (
    final_ports.groupby(["Location", "ContainerType"])["count"]
    .sum().reset_index()
    .rename(columns={"Location": "Port", "count": "Final"})
)

# Only network ports
network_ports = set(final_agg["Port"].unique())
init_agg = init_agg[init_agg["Port"].isin(network_ports)]
port_type = init_agg.merge(final_agg, on=["Port", "ContainerType"], how="outer").fillna(0)
ports = sorted(port_type["Port"].unique())
ctypes = sorted(port_type["ContainerType"].unique())

# ── Stacked bars: one Initial bar (left) + one Final bar (right) per port ────
if AGGREGATE_TYPES:
    type_color_map = {ct: TYPE_COLORS_AGG.get(ct, ("#666", "#bbb"))[0] for ct in ctypes}
else:
    cmap_types = cm.get_cmap("tab20", max(len(ctypes), 1))
    type_color_map = {ct: cmap_types(i) for i, ct in enumerate(ctypes)}

pivot_init  = port_type.pivot_table(index="Port", columns="ContainerType",
                                    values="Initial", aggfunc="sum").reindex(
                                        index=ports, columns=ctypes).fillna(0)
pivot_final = port_type.pivot_table(index="Port", columns="ContainerType",
                                    values="Final", aggfunc="sum").reindex(
                                        index=ports, columns=ctypes).fillna(0)

x = np.arange(len(ports))
bar_width = 0.38

fig, ax = plt.subplots(figsize=(max(12, len(ports) * 0.8), 7))

# Stacked initial bars (left)
bottom_init = np.zeros(len(ports))
for ct in ctypes:
    vals = pivot_init[ct].values
    ax.bar(x - bar_width / 2, vals, bar_width, bottom=bottom_init,
           color=type_color_map[ct], alpha=0.5)
    bottom_init += vals

# Stacked final bars (right)
bottom_final = np.zeros(len(ports))
for ct in ctypes:
    vals = pivot_final[ct].values
    ax.bar(x + bar_width / 2, vals, bar_width, bottom=bottom_final,
           color=type_color_map[ct], alpha=1.0)
    bottom_final += vals

# Legend: one patch per type + initial/final tone
type_patches = [Patch(facecolor=type_color_map[c], label=c) for c in ctypes]
tone_patches = [Patch(facecolor="#888", alpha=0.5, label="Initial (left)"),
                Patch(facecolor="#888", alpha=1.0, label="Final (right)")]
ax.legend(handles=type_patches + tone_patches,
          fontsize=12 if AGGREGATE_TYPES else 10,
          loc="upper right", ncol=1 if AGGREGATE_TYPES else 3)

ax.set_xticks(x)
ax.set_xticklabels(ports, rotation=45, ha="right")
ax.set_ylabel("Containers")
plt.tight_layout()
out = ("visualisation/container-distribution-aggregated.png"
       if AGGREGATE_TYPES else "visualisation/container-distribution.png")
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
plt.show()

# ----------------------------
# Demand fulfillment across all iterations (original demands only)
# ----------------------------

fulfillment_df["ContainerType"] = fulfillment_df["Commodity"].apply(extract_container_type)
maybe_aggregate(fulfillment_df)
fulfillment_df["Delivered"] = fulfillment_df.apply(
    lambda r: r["Fulfilled"] if r["Status"] == "Delivered" else 0, axis=1)
fulfillment_df["InTransit"] = fulfillment_df.apply(
    lambda r: r["Fulfilled"] if r["Status"] in ("InTransit", "WaitingTransship") else 0, axis=1)

if AGGREGATE_TYPES:
    # ── Heatmap split by container type columns ──────────────────────────────
    sub = (
        fulfillment_df.groupby(["Origin", "Destination", "ContainerType"])
        [["Demand", "Fulfilled", "Delivered", "InTransit"]]
        .sum().reset_index()
    )
    sub["OD"] = sub["Origin"] + " → " + sub["Destination"]
    od_labels_list = sorted(sub["OD"].unique())
    ftypes = sorted(sub["ContainerType"].dropna().unique())

    demand_pivot    = sub.pivot_table(index="OD", columns="ContainerType", values="Demand",    aggfunc="sum").reindex(index=od_labels_list, columns=ftypes)
    fulfilled_pivot = sub.pivot_table(index="OD", columns="ContainerType", values="Fulfilled", aggfunc="sum").reindex(index=od_labels_list, columns=ftypes)
    transit_pivot   = sub.pivot_table(index="OD", columns="ContainerType", values="InTransit", aggfunc="sum").reindex(index=od_labels_list, columns=ftypes)

    matrix_rate = (fulfilled_pivot / demand_pivot).astype(float)

    def fmt_cell(row, col):
        d = demand_pivot.iloc[row, col]
        f = fulfilled_pivot.iloc[row, col]
        t = transit_pivot.iloc[row, col]
        if pd.isna(d): return ""
        d, f, t = int(d), int(f or 0), int(t or 0)
        return f"{f-t}+{t}/{d}" if t > 0 else f"{f}/{d}"

    cmap_hm = plt.cm.RdYlGn
    cmap_hm.set_bad(color="#eeeeee")
    data = matrix_rate.values.astype(float)

    fig, ax = plt.subplots(figsize=(max(5, len(ftypes) * 2),
                                    max(4, len(od_labels_list) * 0.4)))
    im = ax.imshow(data, cmap=cmap_hm, vmin=0, vmax=1, aspect="auto")
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            text = fmt_cell(r, c)
            if text:
                val = data[r, c]
                ax.text(c, r, text, ha="center", va="center", fontsize=8,
                        color="white" if val <= 0.3 else "black")

    ax.set_yticks(range(len(od_labels_list)))
    ax.set_yticklabels(od_labels_list, fontsize=8)
    ax.set_xticks(range(len(ftypes)))
    ax.set_xticklabels(ftypes, fontsize=9)
    plt.colorbar(im, ax=ax, label="Fill rate", format="{x:.0%}", shrink=0.6)

else:
    # ── Heatmap aggregated across types (OD only) ────────────────────────────
    sub_od = (
        fulfillment_df.groupby(["Origin", "Destination"])
        [["Demand", "Fulfilled", "Delivered", "InTransit"]]
        .sum().reset_index()
    )
    sub_od["OD"] = sub_od["Origin"] + " → " + sub_od["Destination"]
    sub_od["FillRate"] = sub_od["Fulfilled"] / sub_od["Demand"]
    od_labels_list = sorted(sub_od["OD"].unique())
    sub_od = sub_od.set_index("OD").reindex(od_labels_list)

    data = sub_od["FillRate"].values.reshape(-1, 1).astype(float)

    def fmt_od(idx):
        r = sub_od.iloc[idx]
        d, f, t = int(r["Demand"]), int(r["Fulfilled"]), int(r["InTransit"])
        return f"{f-t}+{t}/{d}" if t > 0 else f"{f}/{d}"

    cmap_hm = plt.cm.RdYlGn
    cmap_hm.set_bad(color="#eeeeee")

    fig, ax = plt.subplots(figsize=(4, max(4, len(od_labels_list) * 0.35)))
    im = ax.imshow(data, cmap=cmap_hm, vmin=0, vmax=1, aspect="auto")
    for r in range(len(od_labels_list)):
        text = fmt_od(r)
        val = data[r, 0]
        ax.text(0, r, text, ha="center", va="center", fontsize=8,
                color="white" if val <= 0.3 else "black")

    ax.set_yticks(range(len(od_labels_list)))
    ax.set_yticklabels(od_labels_list, fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels(["All types"], fontsize=9)
    plt.colorbar(im, ax=ax, label="Fill rate", format="{x:.0%}", shrink=0.6)

    # ── Per-type breakdown table ─────────────────────────────────────────────
    sub = (
        fulfillment_df.groupby(["Origin", "Destination", "ContainerType"])
        [["Demand", "Fulfilled", "Delivered", "InTransit"]]
        .sum().reset_index()
    )

# Print summary
type_summary = (
    sub.groupby(["Origin", "Destination", "ContainerType"])
    [["Demand", "Fulfilled", "Delivered", "InTransit"]]
    .sum().reset_index()
)
type_summary["LoadedRate"] = type_summary["Fulfilled"] / type_summary["Demand"]
print("\n=== Demand Fulfillment by Type (all iterations, original demands) ===")
print(type_summary.to_string(index=False))

totals = type_summary[["Demand", "Fulfilled", "Delivered", "InTransit"]].sum()
print(f"\nOverall: {int(totals['Fulfilled'])}/{int(totals['Demand'])} "
      f"({totals['Fulfilled']/totals['Demand']:.1%})")

plt.tight_layout()
out = "visualisation/demand-fulfilment.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
plt.show()
