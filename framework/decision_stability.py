

import argparse
import shutil
import sys
import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "data")

from framework import runner

# ── Experiment configuration ─────────────────────────────────────────────────

YEARS            = [2023, 2024, 2025]
MODELS = {
    "rejection": "models/max-demand-fulfilment.py",
    "delay":     "models/min-lateness-fulfilment.py",
}
START_WEEK       = 1
PLANNING_WEEKS   = 3
DECISION_WEEKS   = 1
NUM_ITERATIONS   = 6   # 6 iters -> 5 comparison weeks (iter-pairs)
SCENARIO         = "baseline"

OUT_DIR = Path("results/decision-stability")

# Booking identity used to match commodities across iterations. The pipeline
# assigns commodity IDs from a row index that is not stable across iterations,
# so we match on underlying attributes instead.
MATCH_KEYS = [
    "Year", "Week", "Origin", "Destination",
    "ContainerType", "ContainerSize", "Owner",
    "FinalDestination", "ContractType",
]


# ── Phase 1: run rolling horizon + snapshot ──────────────────────────────────

def _reload_model_module(model_file: str) -> None:
    """Swap runner.solve / runner.ModelResults to the requested model file."""
    spec = importlib.util.spec_from_file_location("model", model_file)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    runner.solve        = mod.solve
    runner.ModelResults = mod.ModelResults


def _iter_labels(year: int) -> list[tuple[int, str, str]]:
    """Return [(iter_index, label, horizon_stem)] for the configured run."""
    out = []
    y, w = year, START_WEEK
    for i in range(NUM_ITERATIONS):
        label = f"{y}w{w}"
        end_y, end_w = runner.advance_by_weeks(y, w, PLANNING_WEEKS - 1)
        if PLANNING_WEEKS == 1:
            stem = f"{y}w{w}"
        elif end_y == y:
            stem = f"{y}w{w}-w{end_w}"
        else:
            stem = f"{y}w{w}-{end_y}w{end_w}"
        out.append((i, label, stem))
        y, w = runner.advance_by_weeks(y, w, DECISION_WEEKS)
    return out


def snapshot_paths(model_tag: str, year: int, iter_idx: int) -> dict[str, Path]:
    base = OUT_DIR / f"{model_tag}_{year}" / f"iter_{iter_idx:02d}"
    return {
        "dir":        base,
        "flows":      base / "flows.csv",
        "fulfilment": base / "fulfilment.csv",
        "commodities": base / "commodities.csv",
    }


def already_snapshotted(model_tag: str, year: int) -> bool:
    """Return True iff the iter-00 snapshot exists. Accepts partial runs
    (some years' pipelines crash mid-rolling-horizon; we keep what we have)."""
    p = snapshot_paths(model_tag, year, 0)
    return p["flows"].exists() and p["fulfilment"].exists() and p["commodities"].exists()


def _do_snapshot(model_tag: str, year: int) -> int:
    """Copy whatever per-iteration files exist on disk into the snapshot dir.
    Returns the number of iters with all three files captured."""
    n_ok = 0
    for i, label, stem in _iter_labels(year):
        p = snapshot_paths(model_tag, year, i)
        flows_src       = Path(f"results/model-flows/{label}.csv")
        fulfil_src      = Path(f"results/demand-fulfilment/{label}.csv")
        commodities_src = Path(f"data/processed/commodities/{stem}.csv")
        if not (flows_src.exists() and fulfil_src.exists() and commodities_src.exists()):
            continue
        p["dir"].mkdir(parents=True, exist_ok=True)
        for src, dst in [(flows_src,       p["flows"]),
                         (fulfil_src,      p["fulfilment"]),
                         (commodities_src, p["commodities"])]:
            shutil.copy(src, dst)
        n_ok += 1
    return n_ok


def run_and_snapshot(model_tag: str, model_file: str, year: int) -> None:
    """Run the rolling horizon for one (model, year) and snapshot per-iter files."""
    print(f"\n{'#'*70}")
    print(f"  RUN  model={model_tag}  year={year}  P={PLANNING_WEEKS}w  D={DECISION_WEEKS}w  N={NUM_ITERATIONS}")
    print(f"{'#'*70}")

    runner.START_YEAR             = year
    runner.START_WEEK             = START_WEEK
    runner.PLANNING_HORIZON_WEEKS = PLANNING_WEEKS
    runner.DECISION_HORIZON_WEEKS = DECISION_WEEKS
    runner.NUM_ITERATIONS         = NUM_ITERATIONS
    runner.SCENARIO               = SCENARIO
    runner.MODEL_FILE             = model_file
    _reload_model_module(model_file)

    try:
        runner.main()
    except Exception as e:
        print(f"  runner.main() raised {type(e).__name__}: {e}")
        print(f"  -> snapshotting partial run state and continuing")

    n_ok = _do_snapshot(model_tag, year)
    print(f"  Snapshotted {n_ok}/{NUM_ITERATIONS} iterations -> {OUT_DIR / f'{model_tag}_{year}'}/")


# ── Phase 2: stability metrics ───────────────────────────────────────────────

def _load_iter(model_tag: str, year: int, iter_idx: int) -> dict[str, pd.DataFrame] | None:
    p = snapshot_paths(model_tag, year, iter_idx)
    if not (p["flows"].exists() and p["fulfilment"].exists() and p["commodities"].exists()):
        return None
    return {
        "flows":       pd.read_csv(p["flows"]),
        "fulfilment":  pd.read_csv(p["fulfilment"]),
        "commodities": pd.read_csv(p["commodities"]),
    }


def _real_booking_mask(commodities_df: pd.DataFrame) -> pd.Series:
    """Exclude synthetic carryovers (CARRY_, TLEG_, DELAY_, EMPTY_)."""
    s = commodities_df["Commodity"].astype(str)
    return ~s.str.startswith(("CARRY_", "TLEG_", "DELAY_", "EMPTY_"))


def _teu_per_container(size: float) -> float:
    return 1.0 if int(size) == 20 else 2.0


def _bookings_in_window(snap: dict, dep_lo: int, dep_hi: int) -> pd.DataFrame:
    """Return per-key aggregated bookings whose DepartureTime falls in [dep_lo, dep_hi).

    Each row: MATCH_KEYS + Demand (containers) + Fulfilled + LatenessDays + TEU +
              Commodities (set of iter-local commodity IDs sharing this key in window).
    """
    comm = snap["commodities"]
    ful  = snap["fulfilment"]

    in_win = comm[(comm["DepartureTime"] >= dep_lo) &
                  (comm["DepartureTime"] <  dep_hi) &
                  _real_booking_mask(comm)].copy()
    if in_win.empty:
        return pd.DataFrame(columns=MATCH_KEYS + ["Demand", "Fulfilled",
                                                  "LatenessDays", "TEU",
                                                  "Commodities"])

    in_win["ContractType"] = in_win["ContractType"].fillna("")

    ful_cols = ["Commodity", "Demand", "Fulfilled"]
    if "LatenessDays" in ful.columns:
        ful_cols.append("LatenessDays")
    ful_small = ful[ful_cols].copy()
    if "LatenessDays" not in ful_small.columns:
        ful_small["LatenessDays"] = 0.0

    merged = in_win.merge(ful_small, on="Commodity", how="left")
    # Some commodities may not appear in fulfilment if the solver didn't touch
    # them; treat as zero-fulfilled with declared demand from the commodities file.
    merged["Demand"]       = merged["Demand"].fillna(merged["Count"])
    merged["Fulfilled"]    = merged["Fulfilled"].fillna(0.0)
    merged["LatenessDays"] = merged["LatenessDays"].fillna(0.0)
    merged["TEU"]          = merged["ContainerSize"].apply(_teu_per_container)

    agg = (merged
           .groupby(MATCH_KEYS, dropna=False)
           .agg(Demand=("Demand", "sum"),
                Fulfilled=("Fulfilled", "sum"),
                LatenessDays=("LatenessDays", "sum"),
                TEU=("TEU", "first"),
                Commodities=("Commodity", lambda s: set(s)))
           .reset_index())
    return agg


def _sailing_flows_by_key(snap: dict,
                          dep_lo: int, dep_hi: int,
                          calendar_offset_hours: int,
                          allowed_commodities: set[str]) -> dict[tuple, dict[tuple, float]]:
    """For sailing arcs only, return {match_key: {(cal_dep_hour, DepPort, ArrPort, Vessel): flow}}.

    Flow is in containers (raw model var). Calendar hour = local DepHour +
    calendar_offset_hours, so arcs are comparable across iterations with
    different local epochs.
    """
    flows = snap["flows"]
    comm  = snap["commodities"]

    sail = flows[(flows["Vessel"] != "WAIT") & (flows["Line"] != "TRANS")].copy()
    if sail.empty or not allowed_commodities:
        return {}
    sail = sail[sail["Commodity"].isin(allowed_commodities)]
    sail = sail[(sail["DepHour"] >= dep_lo) & (sail["DepHour"] < dep_hi)]
    if sail.empty:
        return {}

    # Map commodity → match key (need ContractType filled)
    comm_keyed = comm[comm["Commodity"].isin(allowed_commodities)].copy()
    comm_keyed["ContractType"] = comm_keyed["ContractType"].fillna("")
    key_lookup = {row.Commodity: tuple(getattr(row, k) for k in MATCH_KEYS)
                  for row in comm_keyed.itertuples(index=False)}

    out: dict[tuple, dict[tuple, float]] = defaultdict(lambda: defaultdict(float))
    for r in sail.itertuples(index=False):
        key = key_lookup.get(r.Commodity)
        if key is None:
            continue
        arc = (int(r.DepHour) + calendar_offset_hours, r.DepPort, r.ArrPort, r.Vessel)
        out[key][arc] += float(r.Flow)
    return out


def _empty_flows_calendar(snap: dict,
                          dep_lo: int, dep_hi: int,
                          calendar_offset_hours: int) -> dict[tuple, float]:
    """Empty (EMPTY_*) sailing-arc flows aggregated by calendar-anchored arc.

    Returns {(cal_dep_hour, DepPort, ArrPort): total empty container flow}.
    Empties are not tied to bookings, so they are keyed purely by arc. Calendar
    hour = local DepHour + calendar_offset_hours, so arcs are comparable across
    iterations with different local epochs (same scheme as _sailing_flows_by_key).
    """
    flows = snap["flows"]
    sail = flows[(flows["Vessel"] != "WAIT") & (flows["Line"] != "TRANS")].copy()
    if sail.empty:
        return {}
    sail = sail[sail["Commodity"].astype(str).str.startswith("EMPTY_")]
    sail = sail[(sail["DepHour"] >= dep_lo) & (sail["DepHour"] < dep_hi)]
    if sail.empty:
        return {}

    out: dict[tuple, float] = defaultdict(float)
    for r in sail.itertuples(index=False):
        arc = (int(r.DepHour) + calendar_offset_hours, r.DepPort, r.ArrPort)
        out[arc] += float(r.Flow)
    return out


def stability_for_pair(model_tag: str, year: int, i: int) -> dict | None:
    """Compute stability metrics for iter pair (i, i+1)."""
    snap_i  = _load_iter(model_tag, year, i)
    snap_i1 = _load_iter(model_tag, year, i + 1)
    if snap_i is None or snap_i1 is None:
        return None

    HOURS_WEEK = 168
    HOURS_D = DECISION_WEEKS * HOURS_WEEK

    # Compare a single calendar week: iter i's *first uncommitted* week (the
    # first planning-horizon week beyond the decision window) against iter i+1's
    # decision (committed) week. Both correspond to the same calendar week.
    #   Iter i   first-uncommitted window, local epoch: [D, D+1week)
    #   Iter i+1 decision window,          local epoch: [0, 1week)
    book_i  = _bookings_in_window(snap_i,  HOURS_D, HOURS_D + HOURS_WEEK)
    book_i1 = _bookings_in_window(snap_i1, 0,       HOURS_WEEK)

    if book_i.empty and book_i1.empty:
        return None

    # Outer merge on MATCH_KEYS for fulfilment delta
    merged = book_i.merge(book_i1, on=MATCH_KEYS, how="outer",
                          suffixes=("_i", "_i1"))
    for col in ["Demand_i", "Fulfilled_i", "LatenessDays_i",
                "Demand_i1", "Fulfilled_i1", "LatenessDays_i1"]:
        merged[col] = merged[col].fillna(0.0)
    merged["TEU"] = merged["TEU_i"].fillna(merged["TEU_i1"]).fillna(1.0)

    total_demand_containers = float(max(merged["Demand_i"].sum(),
                                        merged["Demand_i1"].sum()))
    abs_fulfilled_delta = float((merged["Fulfilled_i"] -
                                 merged["Fulfilled_i1"]).abs().sum())
    fulfilment_change_pct = (abs_fulfilled_delta / total_demand_containers * 100
                             if total_demand_containers > 0 else 0.0)

    # Lateness delta — only meaningful for delay model; per-container mean of
    # |LatenessDays_i - LatenessDays_{i+1}|, restricted to matched bookings
    # fulfilled in both iters.
    matched_full = merged[(merged["Fulfilled_i"] > 0) &
                          (merged["Fulfilled_i1"] > 0)]
    lateness_abs_delta = float((matched_full["LatenessDays_i"] -
                                matched_full["LatenessDays_i1"]).abs().sum())
    fulfilled_both_containers = float(np.minimum(matched_full["Fulfilled_i"],
                                                 matched_full["Fulfilled_i1"]).sum())
    mean_lateness_change_days = (lateness_abs_delta / fulfilled_both_containers
                                 if fulfilled_both_containers > 0 else 0.0)

    # Booking-level change: a key is "changed" if |Fulfilled_i - Fulfilled_i1| > 0
    n_matched = int(((merged["Demand_i"] > 0) & (merged["Demand_i1"] > 0)).sum())
    n_changed = int(((merged["Fulfilled_i"] - merged["Fulfilled_i1"]).abs() > 1e-6).sum())
    pct_bookings_changed = n_changed / n_matched * 100 if n_matched > 0 else 0.0

    # ── Route change on matched-and-fulfilled bookings ───────────────────────
    matched_keys = set()
    for row in matched_full.itertuples(index=False):
        matched_keys.add(tuple(getattr(row, k) for k in MATCH_KEYS))

    # Commodities (iter-local IDs) we care about
    comms_i  = {c for row in book_i.itertuples(index=False)
                if tuple(getattr(row, k) for k in MATCH_KEYS) in matched_keys
                for c in row.Commodities}
    comms_i1 = {c for row in book_i1.itertuples(index=False)
                if tuple(getattr(row, k) for k in MATCH_KEYS) in matched_keys
                for c in row.Commodities}

    # Calendar-anchor offsets: iter i's tentative window starts at calendar
    # hour (i*D + D)*168 from a global origin, iter i+1's locked window at
    # ((i+1)*D + 0)*168 = same value. We can use any common origin; use
    # "global hour 0 = start of iter 0".
    off_i  = i       * DECISION_WEEKS * 168
    off_i1 = (i + 1) * DECISION_WEEKS * 168

    flows_i  = _sailing_flows_by_key(snap_i,  HOURS_D, HOURS_D + HOURS_WEEK,
                                     off_i, comms_i)
    flows_i1 = _sailing_flows_by_key(snap_i1, 0,       HOURS_WEEK,
                                     off_i1, comms_i1)

    total_arc_flow = 0.0
    diff_arc_flow  = 0.0
    for key in matched_keys:
        arcs_i  = flows_i.get(key,  {})
        arcs_i1 = flows_i1.get(key, {})
        all_arcs = set(arcs_i) | set(arcs_i1)
        for arc in all_arcs:
            fi  = arcs_i.get(arc,  0.0)
            fi1 = arcs_i1.get(arc, 0.0)
            diff_arc_flow  += abs(fi - fi1)
            total_arc_flow += (fi + fi1)

    # Convention: route_change_pct = symmetric difference / average flow.
    # Equivalent to TEU-weighted "fraction of container-moves on sailing arcs
    # that differ". Range [0%, 100%].
    route_change_pct = (diff_arc_flow / total_arc_flow * 100
                        if total_arc_flow > 0 else 0.0)

    # ── Repositioning change on empty (EMPTY_*) sailing arcs ──────────────────
    # Independent of bookings: compare all empty arc flows in the overlap window.
    empt_i  = _empty_flows_calendar(snap_i,  HOURS_D, HOURS_D + HOURS_WEEK, off_i)
    empt_i1 = _empty_flows_calendar(snap_i1, 0,       HOURS_WEEK,           off_i1)
    total_empty_flow = 0.0
    diff_empty_flow  = 0.0
    for arc in set(empt_i) | set(empt_i1):
        fi  = empt_i.get(arc,  0.0)
        fi1 = empt_i1.get(arc, 0.0)
        diff_empty_flow  += abs(fi - fi1)
        total_empty_flow += (fi + fi1)
    repo_change_pct = (diff_empty_flow / total_empty_flow * 100
                       if total_empty_flow > 0 else 0.0)

    return {
        "model":                 model_tag,
        "year":                  year,
        "pair":                  f"iter{i:02d}->iter{i+1:02d}",
        "n_bookings_matched":    n_matched,
        "n_bookings_changed":    n_changed,
        "pct_bookings_changed":  round(pct_bookings_changed, 2),
        "demand_containers":     int(total_demand_containers),
        "fulfilment_change_pct": round(fulfilment_change_pct, 2),
        "mean_lateness_change_days": round(mean_lateness_change_days, 3),
        "route_change_pct":      round(route_change_pct, 2),
        "repo_change_pct":       round(repo_change_pct, 2),
    }


# ── Reporting ────────────────────────────────────────────────────────────────

def collect_all() -> pd.DataFrame:
    rows = []
    for model_tag in MODELS:
        for year in YEARS:
            for i in range(NUM_ITERATIONS - 1):
                r = stability_for_pair(model_tag, year, i)
                if r is not None:
                    rows.append(r)
    return pd.DataFrame(rows)


def print_per_pair(df: pd.DataFrame) -> None:
    if df.empty:
        print("No stability rows.")
        return
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    cols_rej = ["model", "year", "pair", "n_bookings_matched",
                "pct_bookings_changed", "fulfilment_change_pct",
                "route_change_pct", "repo_change_pct"]
    cols_del = cols_rej + ["mean_lateness_change_days"]
    print("\n=== Decision stability — per iteration pair ===")
    for model_tag in MODELS:
        sub = df[df["model"] == model_tag]
        if sub.empty:
            continue
        print(f"\n[{model_tag}]")
        cols = cols_del if model_tag == "delay" else cols_rej
        print(sub[cols].to_string(index=False))


def print_per_year_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return
    # Exclude the first pair (cold start) when averaging — flag separately.
    df = df.copy()
    df["is_coldstart"] = df["pair"].str.startswith("iter00->")
    agg = (df.groupby(["model", "year", "is_coldstart"])
             .agg(pct_bookings_changed=("pct_bookings_changed", "mean"),
                  fulfilment_change_pct=("fulfilment_change_pct", "mean"),
                  route_change_pct=("route_change_pct", "mean"),
                  repo_change_pct=("repo_change_pct", "mean"),
                  mean_lateness_change_days=("mean_lateness_change_days", "mean"),
                  n_pairs=("pair", "count"))
             .reset_index()
             .round(2))
    print("\n=== Per-year summary (cold-start pair separated) ===")
    print(agg.to_string(index=False))


def save_outputs(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "stability_per_pair.csv", index=False)

    # Long-form summary: model x year means over all iteration pairs
    # (including the cold-start pair iter00->iter01).
    if not df.empty:
        summary = (df.groupby(["model", "year"])
                       .agg(mean_pct_bookings_changed=("pct_bookings_changed", "mean"),
                            mean_fulfilment_change_pct=("fulfilment_change_pct", "mean"),
                            mean_route_change_pct=("route_change_pct", "mean"),
                            mean_repo_change_pct=("repo_change_pct", "mean"),
                            mean_lateness_change_days=("mean_lateness_change_days", "mean"),
                            n_pairs=("pair", "count"))
                       .reset_index()
                       .round(2))
        summary.to_csv(OUT_DIR / "stability_summary.csv", index=False)
        print(f"\nSaved: {OUT_DIR/'stability_per_pair.csv'}")
        print(f"Saved: {OUT_DIR/'stability_summary.csv'}")


def make_plots(df: pd.DataFrame) -> None:
    """Two figures: route_change_pct vs iter pair (one panel per year),
       and fulfilment_change_pct vs iter pair (same layout). Both models overlaid."""
    if df.empty:
        return
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
    from plot_style import apply_style, MODEL
    apply_style()

    df = df.copy()
    df["pair_idx"] = df["pair"].str.extract(r"iter(\d+)->").astype(int)

    def plot_metric(metric: str, ylabel: str, filename: str) -> None:
        fig, axes = plt.subplots(1, len(YEARS), figsize=(4 * len(YEARS), 3.2),
                                 sharey=True)
        if len(YEARS) == 1:
            axes = [axes]
        for ax, year in zip(axes, YEARS):
            for model_tag, marker in zip(MODELS, ["o", "s"]):
                sub = df[(df["year"] == year) & (df["model"] == model_tag)]
                if sub.empty:
                    continue
                ax.plot(sub["pair_idx"], sub[metric],
                        marker=marker, label=model_tag.capitalize(), linewidth=1.4,
                        color=MODEL.get(model_tag))
            ax.set_title(str(year))
            ax.set_xlabel("iteration pair (i vs i+1)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / filename, dpi=150)
        plt.close(fig)
        print(f"Saved: {OUT_DIR/filename}")

    plot_metric("route_change_pct",
                "Cargo routed differently (%)",
                "route_change_by_pair.png")
    plot_metric("fulfilment_change_pct",
                "Fulfilment change (% of demand)",
                "fulfilment_change_by_pair.png")
    plot_metric("repo_change_pct",
                "Empties routed differently (%)",
                "repo_change_by_pair.png")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true",
                    help="Skip runs; only analyse existing snapshots.")
    ap.add_argument("--run-only", action="store_true",
                    help="Run + snapshot but skip analysis.")
    ap.add_argument("--force-rerun", action="store_true",
                    help="Re-run even if snapshots already exist.")
    args = ap.parse_args()

    if not args.analyse:
        for model_tag, model_file in MODELS.items():
            for year in YEARS:
                if not args.force_rerun and already_snapshotted(model_tag, year):
                    print(f"  skip (snapshots exist): {model_tag}_{year}")
                    continue
                run_and_snapshot(model_tag, model_file, year)

    if args.run_only:
        return

    df = collect_all()
    print_per_pair(df)
    print_per_year_summary(df)
    save_outputs(df)
    make_plots(df)


if __name__ == "__main__":
    main()
