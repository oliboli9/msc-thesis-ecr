"""
Build the small anonymised sample that ships with this repository.

The real Eimskip dataset is confidential and is not redistributed. This script takes
the real raw exports and writes much smaller, numerically anonymised stand-ins over
`data/raw/`, so the pipeline and the optimisation models can still be run end to end
as a proof of concept.

What is preserved
    Port codes, vessel codes, trade-line names, sailing schedules, container types and
    the categorical structure (owner, contract type, full/empty, size). These are
    public network information and the models depend on them.

What is anonymised
    Every volume. Demand `Total`, vessel `TEUs` and stock `count` are redrawn from a
    Poisson distribution around `value * scale`, so no real figure survives while the
    OD, seasonal and utilisation shape stays realistic.

What is reduced
    Only a short window of weeks is kept (default 2025 w1-w8, matching the default
    START_YEAR/START_WEEK in framework/runner.py), and only a fraction of the demand
    rows within it — sampled stratified so every owner / contract type / container
    class still appears.

Usage (from the repository root, with the real files still in place):

    python pipeline/make_sample_data.py --year 2025 --start-week 1 --num-weeks 8 \
           --row-frac 0.25 --scale 0.25 --seed 42
"""

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")

DEMAND_FILE   = RAW / "Eimskip_data_final_LeaseTypes.csv"
VOYAGES_FILE  = RAW / "Eimskip_voyages.csv"
CAPACITY_FILE = RAW / "vessel-capacities.csv"
STOCK_YEARS   = [2023, 2024, 2025, 2026]

MAIN_LINES = {"Red Line", "Green Line", "Yellow Line", "Blue Line"}
NO_VOYAGE  = {"", "000", " 000", "nan", "NAN"}


def iso_monday(year: int, week: int) -> date:
    """Monday of the given ISO year/week (same helper as pipeline/config.py)."""
    jan4 = date(year, 1, 4)
    return jan4 - timedelta(days=jan4.weekday()) + timedelta(weeks=week - 1)


def window_weeks(year: int, start_week: int, num_weeks: int, pad: int = 1):
    """(iso_year, iso_week) pairs of the sample window, with `pad` weeks either side."""
    first = iso_monday(year, start_week) - timedelta(weeks=pad)
    weeks = []
    for i in range(num_weeks + 2 * pad):
        iso = (first + timedelta(weeks=i)).isocalendar()
        weeks.append((iso[0], iso[1]))
    return weeks


def poisson_counts(values, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Redraw counts around value*scale; never returns 0 so no row becomes a no-op."""
    lam = np.asarray(values, dtype=float) * scale
    return np.maximum(1, rng.poisson(np.maximum(lam, 0.05))).astype(int)


# Voyages

def sample_voyages(weeks, out_path: Path) -> pd.DataFrame:
    """Keep whole voyages that touch the window. Schedules are left untouched."""
    voy = pd.read_csv(VOYAGES_FILE, parse_dates=["etaDateTime", "etdDateTime"])
    keys = set(weeks)
    in_window = voy.apply(lambda r: (int(r["year"]), int(r["week"])) in keys, axis=1)
    # Keep every stop of any voyage with at least one stop in the window, so no
    # rotation is truncated half way through.
    keep_voyages = set(voy.loc[in_window, "voyage"])
    out = voy[voy["voyage"].isin(keep_voyages)].copy()
    out = out.sort_values(["year", "week", "voyage", "etaDateTime"])
    out.to_csv(out_path, index=False)
    print(f"voyages     : {len(out):,} stops, {out['voyage'].nunique()} voyages, "
          f"{out['portID'].nunique()} ports -> {out_path}")
    return out


# Demand

def _reefer(code) -> bool:
    code = str(code).strip()
    return len(code) >= 3 and code[2].upper() == "R"


def sample_demand(weeks, voyages: pd.DataFrame, row_frac: float, scale: float,
                  rng: np.random.Generator, out_path: Path) -> pd.DataFrame:
    dem = pd.read_csv(DEMAND_FILE)
    keys = set(weeks)
    dem = dem[[(int(y), int(w)) in keys
               for y, w in zip(dem["Year"], dem["Week"])]].copy()

    # Drop main-line rows whose voyage is not in the sampled schedule, otherwise the
    # commodity builder silently discards them later.
    known = set(voyages["voyage"].astype(str))

    def voyage_ok(row) -> bool:
        for leg, line_col in (("Voyage 1", "Line 1"), ("Voyage 2", "Line 2")):
            v = str(row.get(leg) or "").strip()
            if v in NO_VOYAGE or v == "nan":
                continue
            if str(row.get(line_col) or "").strip() in MAIN_LINES and v not in known:
                return False
        return True

    dem = dem[dem.apply(voyage_ok, axis=1)].copy()

    # Stratified sample so every categorical value survives the downsampling.
    dem["_strata"] = list(zip(
        dem["Owner"].fillna("NA"),
        dem["Contract type"].fillna("NA"),
        dem["Full/Empty"].fillna("NA"),
        dem["Container type"].map(_reefer),
    ))
    parts = []
    for _, group in dem.groupby("_strata", sort=False):
        n = max(1, int(round(len(group) * row_frac)))
        parts.append(group.sample(n=min(n, len(group)), random_state=rng.integers(2**31)))
    out = pd.concat(parts).drop(columns="_strata")
    out = out.sort_values(["Year", "Week", "Load 1", "Discharge 1"]).reset_index(drop=True)

    out["Total"] = poisson_counts(out["Total"], scale, rng)
    out.to_csv(out_path, index=False)
    print(f"demand      : {len(out):,} rows ({len(dem):,} in window), "
          f"{out['Total'].sum():,} containers -> {out_path}")
    return out


# Vessel capacities

def sample_capacities(supply_scale: float, rng: np.random.Generator,
                      out_path: Path) -> None:
    # Every vessel is kept: pipeline/2_construct_arcs_schedule.py has a hard-coded
    # VESSEL_ROTATION and raises KeyError on any vessel missing from this file.
    cap = pd.read_csv(CAPACITY_FILE, sep=r"\s+")
    jitter = rng.uniform(0.9, 1.1, size=len(cap))
    # Scaled by the same factor as the total demand volume, so fill rates and vessel
    # utilisation stay in the same regime as the full dataset instead of becoming
    # trivially feasible.
    cap["TEUs"] = np.maximum(
        10, np.round(cap["TEUs"] * supply_scale * jitter, -1)).astype(int)
    with open(out_path, "w") as fh:
        fh.write("Vessel TEUs\n")
        for _, r in cap.iterrows():
            fh.write(f"{r['Vessel']}\t{r['TEUs']}\n")
    print(f"capacities  : {len(cap)} vessels -> {out_path}")


# Stock snapshots

def _in_transit_legs(voyages: pd.DataFrame, epoch: datetime, limit: int = 40):
    """(last_loc, next_loc) pairs of vessels actually at sea at the snapshot instant.

    runner._find_vessel_arrival only resolves a pair when some voyage visits last_loc
    before next_loc and departed last_loc before the epoch, so the pairs are taken
    straight out of the sampled schedule rather than copied from the real snapshot.
    """
    legs = []
    for _, group in voyages.groupby("voyage"):
        route = group.sort_values("etaDateTime").reset_index(drop=True)
        for i in range(len(route) - 1):
            dep, arr = route.iloc[i], route.iloc[i + 1]
            if dep["etdDateTime"] < epoch <= arr["etaDateTime"]:
                legs.append((dep["portID"], arr["portID"]))
    seen, out = set(), []
    for leg in legs:
        if leg[0] != leg[1] and leg not in seen:
            seen.add(leg)
            out.append(leg)
    return out[:limit]


def sample_stock(year: int, voyages: pd.DataFrame, sample_years: set,
                 supply_scale: float, rng: np.random.Generator) -> None:
    path = RAW / f"Stock_0101{year}.csv"
    stock = pd.read_csv(path)
    cols = list(stock.columns)

    port_rows = stock[stock["Location"] != "VESSEL"].copy()
    port_rows["count"] = poisson_counts(port_rows["count"], supply_scale, rng)

    vessel_rows = pd.DataFrame(columns=cols)
    if year in sample_years:
        # The snapshot is read at iteration 0, whose epoch is the Monday of the
        # first horizon week; week 1 is the only case the file name implies.
        epoch = datetime.combine(iso_monday(year, 1), datetime.min.time())
        legs = _in_transit_legs(voyages, epoch)
        if legs:
            template = stock[stock["Location"] == "VESSEL"]
            if template.empty:
                template = stock[stock["Location"] != "VESSEL"]
            template = template.sample(n=min(len(legs) * 3, len(template)),
                                       random_state=rng.integers(2**31))
            built = []
            for i, (_, row) in enumerate(template.iterrows()):
                last, nxt = legs[i % len(legs)]
                r = row.copy()
                r["Location"] = "VESSEL"
                r["Last location"] = last
                r["Next location"] = nxt
                built.append(r)
            vessel_rows = pd.DataFrame(built)
            vessel_rows["count"] = poisson_counts(vessel_rows["count"],
                                                  supply_scale, rng)

    out = pd.concat([port_rows, vessel_rows], ignore_index=True)[cols]
    out.to_csv(path, index=False)
    print(f"stock {year} : {len(port_rows):,} port rows + {len(vessel_rows):,} "
          f"in-transit rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2025, help="ISO year of the first week")
    ap.add_argument("--start-week", type=int, default=1, help="ISO week to start from")
    ap.add_argument("--num-weeks", type=int, default=8, help="weeks kept in the sample")
    ap.add_argument("--row-frac", type=float, default=0.25,
                    help="fraction of demand rows kept per stratum")
    ap.add_argument("--scale", type=float, default=0.25,
                    help="volume scale applied to each demand row's Total")
    ap.add_argument("--supply-scale", type=float, default=None,
                    help="volume scale for vessel capacity and stock; defaults to "
                         "row-frac * scale, i.e. the factor by which total demand "
                         "shrinks, so the sample is neither trivially feasible nor "
                         "hopelessly infeasible")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    supply_scale = (args.supply_scale if args.supply_scale is not None
                    else args.row_frac * args.scale)
    rng = np.random.default_rng(args.seed)
    weeks = window_weeks(args.year, args.start_week, args.num_weeks)
    sample_years = {y for y, _ in weeks}
    print(f"window: {weeks[0]} .. {weeks[-1]} (incl. 1 padding week either side)")
    print(f"demand scale {args.scale} x row fraction {args.row_frac}; "
          f"supply scale {supply_scale:.4g}\n")

    voyages = sample_voyages(weeks, VOYAGES_FILE)
    sample_demand(weeks, voyages, args.row_frac, args.scale, rng, DEMAND_FILE)
    sample_capacities(supply_scale, rng, CAPACITY_FILE)
    for year in STOCK_YEARS:
        sample_stock(year, voyages, sample_years, supply_scale, rng)

    print("\nSample written. Now regenerate the derived data:")
    print("  python pipeline/0_clean_data.py")
    print("  python pipeline/augment_demand.py --all --method scale --seed 42")
    print("  python pipeline/config.py")


if __name__ == "__main__":
    main()
