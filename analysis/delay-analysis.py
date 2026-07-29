"""
analysis/delay-analysis.py  —  Lateness characterisation for Section 6.1.

Reads demand-fulfilment CSVs produced by min-lateness-fulfilment.py and
produces summary tables and figures:

  1. Overall lateness statistics
  2. Histogram of mean lateness per commodity
  3. Breakdown by Dry vs Reefer
  4. Breakdown by Owner
  5. Top-N most-delayed OD pairs

Usage:
    python analysis/delay-analysis.py
    python analysis/delay-analysis.py results/model-comparison-rolling/2025w1_delay
    python analysis/delay-analysis.py results/my-experiment/mu0.5
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import apply_style, LINE_COLORS

apply_style()

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "model-comparison-rolling" / "2025w1_delay"
TOP_N_OD = 15
OUTPUT_DIR = REPO_ROOT / "results" / "delay-analysis"

results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
fulfilment_dir = results_dir / "demand-fulfilment"

if not fulfilment_dir.exists():
    # maybe the path itself is the demand-fulfilment directory
    if (results_dir / "2025w1.csv").exists() or any(results_dir.glob("*.csv")):
        fulfilment_dir = results_dir
    else:
        print(f"No demand-fulfilment directory found under: {results_dir}")
        sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Reading from: {fulfilment_dir}")

# ── Load and combine all iteration CSVs ───────────────────────────────────────

dfs = []
for csv in sorted(fulfilment_dir.glob("*.csv")):
    df = pd.read_csv(csv)
    df["_week"] = csv.stem
    dfs.append(df)

if not dfs:
    print("No CSV files found.")
    sys.exit(1)

df = pd.concat(dfs, ignore_index=True)

if "LatenessDays" not in df.columns:
    print("ERROR: LatenessDays column not found. These results may be from "
          "max-demand-fulfilment.py, not min-lateness-fulfilment.py.")
    sys.exit(1)

# ── Derive ContainerType (Dry/Reefer) from Commodity ID ───────────────────────
# Commodity IDs look like: k5_45R1_40_CARRIER, CARRY_2_IS REY_NL RTM, etc.

def _classify(commodity_id: str) -> str:
    parts = commodity_id.split("_")
    for p in parts:
        if len(p) == 4 and p[0].isdigit():
            return "Reefer" if p[2].upper() == "R" else "Dry"
    return "Unknown"

df["ContainerType"] = df["Commodity"].apply(_classify)

# Exclude CARRY_ (carryover tracking) but keep TLEG_ (transshipment second
# legs): these represent real two-leg shipments whose late arrival is meaningful.
in_horizon = df[~df["Commodity"].str.startswith("CARRY_")].copy()
in_horizon_late = in_horizon[in_horizon["LatenessDays"] > 0].copy()

# ── 1. Overall statistics ─────────────────────────────────────────────────────

total_demand     = in_horizon["Demand"].sum()
total_fulfilled  = in_horizon["Fulfilled"].sum()
total_late_cd    = in_horizon["LatenessDays"].sum()
n_late_comms     = (in_horizon["LatenessDays"] > 0).sum()
n_total_comms    = len(in_horizon)
fill_rate        = total_fulfilled / total_demand if total_demand else 0

print("\n=== Overall Statistics ===")
print(f"  Commodities:         {n_total_comms}")
print(f"  Fill rate:           {fill_rate:.2%}")
print(f"  Commodities w/ lateness: {n_late_comms} ({n_late_comms/n_total_comms:.1%})")
print(f"  Total lateness:      {total_late_cd:.0f} container-days")
if n_late_comms:
    print(f"  Mean lateness (late only): {in_horizon_late['MeanLatenessDays'].mean():.2f} days")
    print(f"  Max lateness (commodity):  {in_horizon['MeanLatenessDays'].max():.1f} days")

# ── 2. Histogram of MeanLatenessDays ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(in_horizon["MeanLatenessDays"], bins=30, color=LINE_COLORS["Blue"], edgecolor="white", linewidth=0.4)
ax.set_xlabel("Mean lateness per commodity (days)")
ax.set_ylabel("Number of commodities")
ax.set_title("Distribution of commodity lateness")
ax.set_xlim(left=0)
fig.tight_layout()
out = OUTPUT_DIR / "lateness_histogram.pdf"
fig.savefig(out)
plt.close(fig)
print(f"\nSaved: {out.name}")

# ── 3. Breakdown by Dry vs Reefer ─────────────────────────────────────────────

type_stats = (
    in_horizon.groupby("ContainerType")
    .agg(
        Demand=("Demand", "sum"),
        Fulfilled=("Fulfilled", "sum"),
        LatenessDays=("LatenessDays", "sum"),
        Commodities=("Commodity", "count"),
        LateCount=("LatenessDays", lambda x: (x > 0).sum()),
    )
    .assign(FillRate=lambda d: d["Fulfilled"] / d["Demand"],
            MeanLateDays=lambda d: d["LatenessDays"] / d["Fulfilled"].clip(lower=1))
    .reset_index()
)
print("\n=== Breakdown by Container Type ===")
print(type_stats[["ContainerType", "Commodities", "LateCount", "FillRate",
                   "LatenessDays", "MeanLateDays"]].to_string(index=False))
type_stats.to_csv(OUTPUT_DIR / "lateness_by_type.csv", index=False)

# ── 4. Breakdown by Owner ──────────────────────────────────────────────────────

owner_stats = (
    in_horizon.groupby("Owner")
    .agg(
        Demand=("Demand", "sum"),
        Fulfilled=("Fulfilled", "sum"),
        LatenessDays=("LatenessDays", "sum"),
        Commodities=("Commodity", "count"),
        LateCount=("LatenessDays", lambda x: (x > 0).sum()),
    )
    .assign(FillRate=lambda d: d["Fulfilled"] / d["Demand"],
            MeanLateDays=lambda d: d["LatenessDays"] / d["Fulfilled"].clip(lower=1))
    .reset_index()
)
print("\n=== Breakdown by Owner ===")
print(owner_stats[["Owner", "Commodities", "LateCount", "FillRate",
                    "LatenessDays", "MeanLateDays"]].to_string(index=False))
owner_stats.to_csv(OUTPUT_DIR / "lateness_by_owner.csv", index=False)

# ── 5. Top-N most delayed OD pairs ────────────────────────────────────────────

od_stats = (
    in_horizon.groupby(["Origin", "Destination"])
    .agg(
        Demand=("Demand", "sum"),
        Fulfilled=("Fulfilled", "sum"),
        LatenessDays=("LatenessDays", "sum"),
        Commodities=("Commodity", "count"),
    )
    .assign(FillRate=lambda d: d["Fulfilled"] / d["Demand"],
            MeanLateDays=lambda d: d["LatenessDays"] / d["Fulfilled"].clip(lower=1))
    .sort_values("LatenessDays", ascending=False)
    .reset_index()
)
top_od = od_stats.head(TOP_N_OD)
print(f"\n=== Top {TOP_N_OD} OD Pairs by Total Lateness ===")
print(top_od[["Origin", "Destination", "Demand", "Fulfilled",
              "FillRate", "LatenessDays", "MeanLateDays"]].to_string(index=False))
od_stats.to_csv(OUTPUT_DIR / "lateness_by_od.csv", index=False)

# ── 6. Bar chart: lateness by Dry/Reefer and Owner ────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, (group_col, stats_df, title) in zip(axes, [
    ("ContainerType", type_stats, "Total lateness by container type"),
    ("Owner",         owner_stats, "Total lateness by owner"),
]):
    ax.bar(stats_df[group_col], stats_df["LatenessDays"], color=LINE_COLORS["Blue"])
    ax.set_xlabel(group_col)
    ax.set_ylabel("Container-days late")
    ax.set_title(title)

fig.tight_layout()
out = OUTPUT_DIR / "lateness_breakdown.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved: {out.name}")

print(f"\nAll outputs written to: {OUTPUT_DIR}")
