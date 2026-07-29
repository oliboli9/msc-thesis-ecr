"""
Demand-fulfilment bar chart by origin-destination pair for one iteration.

Reconstructs the baseline fulfilment figure (fig:baseline-fulfilment): one
horizontal bar per OD pair, stacked into delivered / in-transit / unfulfilled
containers, sorted by total demand. No chart title --- the caption carries it.

Reads:
  results/demand-fulfilment/{LABEL}.csv

Produces:
  msc-thesis-writing/Pictures/baseline-demand-fulfilment.png

Usage:
  python analysis/baseline-fulfilment-od.py --label 2026w1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import apply_style, save_fig, STATUS

REPO = Path(__file__).resolve().parent.parent

apply_style()

DELIVERED_COLOUR   = STATUS["delivered"]
INTRANSIT_COLOUR   = STATUS["in_transit"]
UNFULFILLED_COLOUR = STATUS["unfulfilled"]


def plot_fulfilment_by_od(label: str, results_dir: str = "results") -> Path:
    df = pd.read_csv(REPO / results_dir / "demand-fulfilment" / f"{label}.csv")

    # Original demands only: drop synthetic carryover / truncated-leg commodities.
    df = df[~df["Commodity"].str.startswith(("CARRY_", "TLEG_", "EMPTY"))]

    # Delivered = arrived at destination within the horizon; everything else
    # that is fulfilled (in transit, waiting to transship, ...) is "in transit".
    df["Delivered"] = np.where(df["Status"] == "Delivered", df["Fulfilled"], 0.0)

    od = (
        df.groupby(["Origin", "Destination"])[["Demand", "Fulfilled", "Delivered"]]
        .sum().reset_index()
    )
    od["InTransit"] = (od["Fulfilled"] - od["Delivered"]).clip(lower=0)
    od["Unfulfilled"] = (od["Demand"] - od["Fulfilled"]).clip(lower=0)
    od["OD"] = od["Origin"] + " → " + od["Destination"]
    od = od.sort_values("Demand", ascending=False)   # largest on the left

    x = np.arange(len(od))
    fig, ax = plt.subplots(figsize=(max(8, len(od) * 0.32), 6))

    ax.bar(x, od["Delivered"], color=DELIVERED_COLOUR, label="Delivered")
    ax.bar(x, od["InTransit"], bottom=od["Delivered"],
           color=INTRANSIT_COLOUR, label="In Transit")
    ax.bar(x, od["Unfulfilled"], bottom=od["Delivered"] + od["InTransit"],
           color=UNFULFILLED_COLOUR, label="Unfulfilled")

    ax.set_xticks(x)
    ax.set_xticklabels(od["OD"], rotation=90, ha="center", fontsize=8)
    ax.set_ylabel("Containers")
    ax.margins(y=0.05)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    out = save_fig(fig, "baseline-demand-fulfilment")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="2026w1", help="Iteration label, e.g. 2026w1")
    ap.add_argument("--results-dir", default="results",
                    help="Results base dir (relative to repo root), e.g. results/_baseline")
    args = ap.parse_args()
    out = plot_fulfilment_by_od(args.label, args.results_dir)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
