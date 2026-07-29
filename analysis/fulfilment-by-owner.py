"""
analysis/fulfilment-by-owner.py
Fill-rate heatmap (OD pair × container size) split into three panels:
Carrier, Leased, and Shipper.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

sys.path.insert(0, "pipeline")
from config import EPOCH_HOUR_OFFSET

EPOCH_DT = datetime(2023, 1, 2)
def to_dt(h): return (EPOCH_DT + timedelta(hours=int(h))).strftime("%d %b %Y %H:%M")

# ----------------------------
# Load
# ----------------------------

fulfillment_df = pd.read_csv("results/demand_fulfillment.csv")

fulfillment_df[["ContainerType", "SizeLabel"]] = (
    fulfillment_df["Commodity"]
    .str.extract(r'_(\w+?)_(\d+)_')
    .rename(columns={0: "ContainerType", 1: "SizeLabel"})
)
fulfillment_df["SizeLabel"] = fulfillment_df["SizeLabel"] + "ft"

fulfillment_df["Delivered"] = fulfillment_df.apply(
    lambda r: r["Fulfilled"] if r["Status"] == "Delivered" else 0, axis=1)
fulfillment_df["InTransit"] = fulfillment_df.apply(
    lambda r: r["Fulfilled"] if r["Status"] in ("InTransit", "WaitingTransship") else 0, axis=1)

owners = ["Carrier", "Leased", "Shipper"]

for owner in owners:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sub = (
        fulfillment_df[fulfillment_df["Owner"] == owner]
        .groupby(["Origin", "Destination", "SizeLabel"])[["Demand", "Fulfilled", "InTransit"]]
        .sum()
        .reset_index()
    )

    if sub.empty:
        ax.set_title(f"{owner}\n(no data)")
        ax.axis("off")
        continue

    sub["OD"] = sub["Origin"] + " → " + sub["Destination"]
    sizes = sorted(sub["SizeLabel"].unique())
    od_labels = sub.groupby("OD").sum(numeric_only=True).index.tolist()

    demand_pivot    = sub.pivot_table(index="OD", columns="SizeLabel", values="Demand",    aggfunc="sum").reindex(index=od_labels, columns=sizes)
    fulfilled_pivot = sub.pivot_table(index="OD", columns="SizeLabel", values="Fulfilled", aggfunc="sum").reindex(index=od_labels, columns=sizes)
    transit_pivot   = sub.pivot_table(index="OD", columns="SizeLabel", values="InTransit", aggfunc="sum").reindex(index=od_labels, columns=sizes)

    matrix_rate = (fulfilled_pivot / demand_pivot).astype(float)

    def fmt_cell(r, c):
        d = demand_pivot.iloc[r, c]
        f = fulfilled_pivot.iloc[r, c] if not pd.isna(fulfilled_pivot.iloc[r, c]) else 0
        t = transit_pivot.iloc[r, c]   if not pd.isna(transit_pivot.iloc[r, c])   else 0
        if pd.isna(d): return ""
        d, f, t = int(d), int(f), int(t)
        if t > 0:
            return f"{f-t}+{t}/{d}"
        return f"{f}/{d}"

    data = matrix_rate.values.astype(float)
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#eeeeee")

    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            text = fmt_cell(r, c)
            if text:
                val = data[r, c]
                color = "white" if val <= 0.3 else "black"
                ax.text(c, r, text, ha="center", va="center", fontsize=7, color=color)

    ax.set_yticks(range(len(od_labels)))
    ax.set_yticklabels(od_labels, fontsize=8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes, fontsize=9)
    ax.set_title(f"{owner} — Fill Rate by OD Pair & Size\n(delivered+transit/demand)", fontsize=11)
    plt.colorbar(im, ax=ax, label="Loaded Rate", format="{x:.0%}", shrink=0.5)
    plt.tight_layout()
    plt.show()
