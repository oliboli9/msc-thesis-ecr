

import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "data")

from framework.runner import (
    advance_by_weeks, build_config_params, build_port_vessel_dicts,
    build_arcs, build_commodities, solve, ModelResults,
    load_stock_df, filter_stock_to_ports, process_stock_to_supply,
    reserve_in_transit_capacity,
    extract_carryover_demands, extract_truncated_second_legs,
    build_carryover_commodities, _merge_carryover_dicts,
    extract_delayed_commodities, build_delayed_commodities,
)

#  Experiment parameters 

START_YEAR = 2025
START_WEEK = 1
N_WEEKS    = 5     # full window length (weeks) shared by both modes
SCENARIO   = "moderate"
MIP_GAP    = 0    

VOYAGES_PATH = "data/clean/eimskip_voyages.csv"
DEMAND_PATH  = f"data/augmented/{SCENARIO}/Eimskip_data_final.csv"
CAP_PATH     = "data/raw/vessel-capacities.csv"

#  Iteration capture 

class IterCapture:
    """Holds the input/output of one rolling iteration for later diffing."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def run_iteration(i, year, week, planning_weeks, decision_weeks,
                  voyages_df, demand_df, cap_df,
                  prev_carryover, prev_truncated_legs,
                  pending_carryover, pending_truncated,
                  prev_delayed, pending_delayed,
                  stock_dir: Path,
                  raw_stock_path: str) -> IterCapture:
    """One iteration body — mirrors framework/runner.py main()."""
    label = f"{year}w{week}"

    cfg = build_config_params(year, week, planning_weeks, decision_weeks, voyages_df)
    cfg["mip_gap"] = MIP_GAP
    cfg["verbose_unfulfilled"] = False

    port_dict, vessel_lines = build_port_vessel_dicts(voyages_df, cfg["calendar_weeks"])
    arcs_df = build_arcs(voyages_df, cap_df, port_dict, cfg)
    commodities_df = build_commodities(demand_df, voyages_df, port_dict, vessel_lines, cfg)

    # Stock — read from the harness-private stock dir for iter > 0
    if i == 0:
        raw_stock = pd.read_csv(raw_stock_path)
    else:
        prev_year, prev_week = advance_by_weeks(START_YEAR, START_WEEK, (i - 1) * decision_weeks)
        raw_stock = pd.read_csv(stock_dir / f"{prev_year}w{prev_week}.csv")

    filtered_stk = filter_stock_to_ports(raw_stock, set(port_dict.keys()))
    arcs_df = reserve_in_transit_capacity(filtered_stk, arcs_df, voyages_df, cfg, verbose=False)

    # All physical stock (including full VESSEL rows) becomes carrier supply.
    # CARRY_ commodities below provide the demand to route those containers to
    # their original destinations; supply for them comes from this stock
    # pathway, not from carryover supply additions.
    stock_for_supply = filtered_stk

    initial_sup, vessel_late_rows = process_stock_to_supply(
        stock_for_supply, arcs_df, cfg["stock_cutoff_hour"],
        cfg["epoch_dt"], voyages_df, verbose=False
    )

    # Inject carryover demands (NOT their supply additions — supply already
    # comes from stock above). Avoids double-counting carriers.
    carryover_to_inject = _merge_carryover_dicts(prev_carryover, pending_carryover)
    new_pending_carry = {}
    if i > 0 and carryover_to_inject:
        carryover_df, _carry_sup_unused, deferred = build_carryover_commodities(
            carryover_to_inject, arcs_df, year, week,
            target_epoch_dt=cfg["epoch_dt"],
            horizon_end=cfg["horizon_end"],
        )
        if not carryover_df.empty:
            commodities_df = pd.concat([carryover_df, commodities_df], ignore_index=True)
        new_pending_carry = deferred
    else:
        new_pending_carry = carryover_to_inject

    # Inject truncated second legs
    legs_to_inject = _merge_carryover_dicts(prev_truncated_legs, pending_truncated)
    new_pending_trunc = {}
    if i > 0 and legs_to_inject:
        leg_df, _, deferred_legs = build_carryover_commodities(
            legs_to_inject, arcs_df, year, week,
            target_epoch_dt=cfg["epoch_dt"],
            horizon_end=cfg["horizon_end"],
        )
        if not leg_df.empty:
            leg_df["Commodity"] = leg_df["Commodity"].str.replace("CARRY_", "TLEG_", n=1)
            commodities_df = pd.concat([leg_df, commodities_df], ignore_index=True)
        new_pending_trunc = deferred_legs
    else:
        new_pending_trunc = legs_to_inject

    # Inject delayed (unfulfilled future from prior iter)
    delayed_to_inject = list(prev_delayed) + list(pending_delayed)
    new_pending_delayed: list = []
    if i > 0 and delayed_to_inject:
        delay_df, deferred_delay = build_delayed_commodities(
            delayed_to_inject, commodities_df, arcs_df,
            target_epoch_dt=cfg["epoch_dt"],
            horizon_end=cfg["horizon_end"],
        )
        if not delay_df.empty:
            commodities_df = pd.concat([delay_df, commodities_df], ignore_index=True)
        new_pending_delayed = deferred_delay
    else:
        new_pending_delayed = delayed_to_inject

    t0 = time.time()
    results = solve(arcs_df, commodities_df, initial_sup, cfg)
    solve_time = time.time() - t0

    if results.status != "OPTIMAL":
        raise RuntimeError(f"Iteration {i+1} solve status: {results.status}")

    # Save stock to harness-private dir for next iteration
    stock_dir.mkdir(parents=True, exist_ok=True)
    stock_out = results.stock_df
    if vessel_late_rows:
        stock_out = pd.concat([stock_out, pd.DataFrame(vessel_late_rows)], ignore_index=True)
    if "ArrivalTime" in stock_out.columns:
        stock_out = stock_out.copy()
        stock_out["ArrivalTime"] = stock_out["ArrivalTime"].apply(
            lambda h: (cfg["epoch_dt"] + timedelta(hours=int(float(h)))).isoformat()
            if pd.notna(h) and str(h).strip() not in ("", "nan") else ""
        )
    stock_out.to_csv(stock_dir / f"{label}.csv", index=False)

    cap = IterCapture(
        i=i, label=label, year=year, week=week, cfg=cfg,
        commodities_df=commodities_df.copy(),
        initial_supply=dict(initial_sup),
        carryover_in=dict(carryover_to_inject),
        truncated_in=dict(legs_to_inject),
        delayed_in=list(delayed_to_inject),
        results=results,
        solve_time=solve_time,
    )

    # Extract for next iteration — the harness drives the loop
    cap.next_carryover = extract_carryover_demands(results, commodities_df, cfg)
    cap.next_truncated = extract_truncated_second_legs(results, commodities_df, cfg)
    cap.next_delayed   = extract_delayed_commodities(results, commodities_df, cfg)
    cap.pending_carry  = new_pending_carry
    cap.pending_trunc  = new_pending_trunc
    cap.pending_delayed = new_pending_delayed
    return cap


def run_mode(planning_weeks_fn, decision_weeks: int, n_iters: int,
             voyages_df, demand_df, cap_df, stock_dir: Path,
             raw_stock_path: str, label: str) -> list[IterCapture]:
    print(f"\n{'#'*70}\n# {label}\n{'#'*70}")
    captures = []
    prev_carry, prev_trunc, pend_carry, pend_trunc = {}, {}, {}, {}
    prev_delay: list = []
    pend_delay: list = []
    for i in range(n_iters):
        year, week = advance_by_weeks(START_YEAR, START_WEEK, i * decision_weeks)
        plan_w = planning_weeks_fn(i)
        dec_w  = min(decision_weeks, plan_w)
        print(f"\n[{label}] iter {i+1}/{n_iters}: {year}w{week:02d}  "
              f"plan={plan_w}w  decision={dec_w}w")
        cap = run_iteration(
            i, year, week, plan_w, dec_w,
            voyages_df, demand_df, cap_df,
            prev_carry, prev_trunc, pend_carry, pend_trunc,
            prev_delay, pend_delay,
            stock_dir, raw_stock_path,
        )
        captures.append(cap)
        prev_carry  = cap.next_carryover
        prev_trunc  = cap.next_truncated
        pend_carry  = cap.pending_carry
        pend_trunc  = cap.pending_trunc
        prev_delay  = cap.next_delayed
        pend_delay  = cap.pending_delayed
    return captures


# ── Diff utilities ───────────────────────────────────────────────────────────

def slice_mode_a_by_week(cap_a: IterCapture, week_idx: int):
    """Return (commodities subset, fulfillment subset, flows subset) for week_idx
    of mode A's single solve. week_idx is 0-based; week 0 = first week.
    """
    cfg = cap_a.cfg
    h_lo = week_idx * 168
    h_hi = (week_idx + 1) * 168
    comms = cap_a.commodities_df
    week_comms = comms[(comms["DepartureTime"] >= h_lo) & (comms["DepartureTime"] < h_hi)].copy()
    ful = cap_a.results.fulfillment_df
    week_ful = ful[ful["Commodity"].isin(week_comms["Commodity"])].copy()
    flows = cap_a.results.flows_df
    week_flows = flows[(flows["DepHour"].astype(int) >= h_lo) &
                       (flows["DepHour"].astype(int) < h_hi)].copy()
    return week_comms, week_ful, week_flows


def diff_per_week(cap_a: IterCapture, captures_b: list[IterCapture]):
    print(f"\n{'='*70}\nPER-WEEK FULFILMENT  (raw demand only — no CARRY_/TLEG_)\n{'='*70}")
    print(f"{'week':>5}  {'A demand':>10} {'A fulfill':>10}  "
          f"{'B demand':>10} {'B fulfill':>10}  {'Δ fulfill':>10}")
    print("-" * 70)
    a_total = b_total = a_ful_t = b_ful_t = 0
    for k, cap_b in enumerate(captures_b):
        # A: week k slice (raw commodities only — CARRY_ doesn't exist in mode A)
        a_comms, a_ful, _ = slice_mode_a_by_week(cap_a, k)
        a_demand = int(a_ful["Demand"].sum())
        a_fulfilled = int(a_ful["Fulfilled"].sum())

        # B: iter k decision-window slice — exclude CARRY_/TLEG_ for fair compare
        cutoff = cap_b.cfg["stock_cutoff_hour"]
        comms_b = cap_b.commodities_df
        is_carry = comms_b["Commodity"].str.startswith(("CARRY_", "TLEG_"))
        dw_mask = (comms_b["DepartureTime"] < cutoff) & ~is_carry
        b_dw = set(comms_b.loc[dw_mask, "Commodity"])
        ful_b = cap_b.results.fulfillment_df
        b_dw_ful = ful_b[ful_b["Commodity"].isin(b_dw)]
        b_demand = int(b_dw_ful["Demand"].sum())
        b_fulfilled = int(b_dw_ful["Fulfilled"].sum())

        delta = b_fulfilled - a_fulfilled
        print(f"{k+1:>5}  {a_demand:>10} {a_fulfilled:>10}  "
              f"{b_demand:>10} {b_fulfilled:>10}  {delta:>+10}")
        a_total += a_demand; a_ful_t += a_fulfilled
        b_total += b_demand; b_ful_t += b_fulfilled
    print("-" * 70)
    print(f"{'tot':>5}  {a_total:>10} {a_ful_t:>10}  "
          f"{b_total:>10} {b_ful_t:>10}  {b_ful_t - a_ful_t:>+10}")
    if a_ful_t:
        pct = 100.0 * (b_ful_t - a_ful_t) / a_ful_t
        print(f"\nMode B vs Mode A: {pct:+.2f}% on cumulative fulfilment")


def diff_iter0_identity(cap_a: IterCapture, cap_b0: IterCapture):
    """At iter 0 both modes see the same commodities and same arcs (no carryover).
    The week-1 slice of mode A's solution should match mode B iter-0's locked
    week-1 commodity-level fulfilment exactly (modulo solver non-uniqueness)."""
    print(f"\n{'='*70}\nITER-0 IDENTITY CHECK  (A week-1 vs B iter-0 week-1)\n{'='*70}")

    a_comms, a_ful, a_flows = slice_mode_a_by_week(cap_a, 0)
    cutoff = cap_b0.cfg["stock_cutoff_hour"]
    comms_b = cap_b0.commodities_df
    dw_mask = (comms_b["DepartureTime"] < cutoff)
    b_dw_comms = set(comms_b.loc[dw_mask, "Commodity"])

    a_dw_comms = set(a_comms["Commodity"])
    only_a = a_dw_comms - b_dw_comms
    only_b = b_dw_comms - a_dw_comms
    common = a_dw_comms & b_dw_comms
    print(f"  Decision-window commodities: A={len(a_dw_comms)}  B={len(b_dw_comms)}  "
          f"only-A={len(only_a)}  only-B={len(only_b)}  common={len(common)}")

    if only_a or only_b:
        print(f"    [WARN] commodity sets differ — ports/network mismatch. "
              f"Investigate: only-A sample {list(only_a)[:5]}, only-B sample {list(only_b)[:5]}")

    # Per-commodity fulfilment diff on common set
    a_ful_map = a_ful.set_index("Commodity")["Fulfilled"].to_dict()
    b_ful_map = cap_b0.results.fulfillment_df.set_index("Commodity")["Fulfilled"].to_dict()
    diffs = []
    for k in common:
        if abs(a_ful_map.get(k, 0) - b_ful_map.get(k, 0)) > 0.5:
            diffs.append((k, a_ful_map.get(k, 0), b_ful_map.get(k, 0)))

    if not diffs:
        print(f"  [OK] All {len(common)} common commodities have matching fulfilment "
              f"in week-1 of both modes.")
    else:
        print(f"  [DIVERGE] {len(diffs)} commodities with different fulfilment:")
        for k, a, b in diffs[:10]:
            print(f"    {k}: A={a:.0f}  B={b:.0f}  diff={b - a:+.0f}")
        if len(diffs) > 10:
            print(f"    ... and {len(diffs) - 10} more")

    a_obj = float(a_ful["Fulfilled"].sum())
    b_obj_dw = float(sum(b_ful_map.get(k, 0) for k in b_dw_comms))
    print(f"  A week-1 total fulfilled: {a_obj:.0f}")
    print(f"  B iter-0 DW total fulfilled: {b_obj_dw:.0f}  diff={b_obj_dw - a_obj:+.0f}")


def _normalize_stock(stock_df: pd.DataFrame, epoch_dt) -> dict:
    """Bucket a stock DataFrame into a comparable dict.

    Returns {(loc_kind, key, ct, cs, owner, fullEmpty): count}, where
    loc_kind ∈ {"port", "vessel", "with_customer"}:
      - port:          key = port code; container is at port
      - vessel:        key = (last_loc, next_loc, arr_wall_iso); in transit
      - with_customer: key = (port, return_wall_iso); empty out for delivery

    Times are converted to wall-clock ISO so two stock dicts produced from
    different epochs are comparable.
    """
    out: dict = defaultdict(int)
    if stock_df is None or stock_df.empty:
        return out
    for _, r in stock_df.iterrows():
        owner = str(r["Owner"]).strip().upper()
        if owner != "CARRIER":
            continue
        ct = str(r["ContainerType"]).strip()
        try:
            cs = float(r["ContainerSize"])
        except (TypeError, ValueError):
            continue
        try:
            count = int(round(float(r["count"])))
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        loc = str(r.get("Location", "")).strip()
        fe  = str(r.get("FullEmpty", "Empty")).strip().title()
        wc  = str(r.get("With customer", "")).strip().lower() == "true"

        if loc.upper() == "VESSEL":
            last_loc = str(r.get("Last location", "")).strip()
            next_loc = str(r.get("Next location", "")).strip()
            arr_t = r.get("ArrivalTime")
            if pd.isna(arr_t) or str(arr_t).strip() in ("", "nan"):
                arr_iso = ""
            else:
                # ArrivalTime in iter stock CSVs is already ISO; in mode A's
                # raw stock it is an epoch hour int — coerce both to ISO.
                from datetime import timedelta as _td
                s = str(arr_t).strip()
                try:
                    arr_iso = pd.to_datetime(s).isoformat()
                except (ValueError, TypeError):
                    try:
                        arr_iso = (epoch_dt + _td(hours=int(float(s)))).isoformat()
                    except (TypeError, ValueError):
                        arr_iso = s
            key = ("vessel", (last_loc, next_loc, arr_iso), ct, cs, owner.title(), fe)
        elif wc:
            arr_t = r.get("ArrivalTime")
            from datetime import timedelta as _td
            s = "" if pd.isna(arr_t) else str(arr_t).strip()
            try:
                ret_iso = pd.to_datetime(s).isoformat() if s else ""
            except (ValueError, TypeError):
                try:
                    ret_iso = (epoch_dt + _td(hours=int(float(s)))).isoformat()
                except (TypeError, ValueError):
                    ret_iso = s
            key = ("with_customer", (loc, ret_iso), ct, cs, owner.title(), fe)
        else:
            key = ("port", loc, ct, cs, owner.title(), fe)
        out[key] += count
    return out


def _summarize_state(state: dict) -> dict:
    """Aggregate a state dict to (kind, port, fullEmpty) → total count."""
    agg: dict = defaultdict(int)
    for key, cnt in state.items():
        kind = key[0]
        if kind == "vessel":
            _, (last_loc, next_loc, _arr), _ct, _cs, _own, fe = key
            agg[(kind, f"{last_loc}->{next_loc}", fe)] += cnt
        elif kind == "with_customer":
            _, (port, _ret), _ct, _cs, _own, fe = key
            agg[(kind, port, fe)] += cnt
        else:
            _, port, _ct, _cs, _own, fe = key
            agg[(kind, port, fe)] += cnt
    return agg


def diff_per_iter_state(stock_dir_a: Path, captures_b: list[IterCapture],
                        voyages_df, demand_df, cap_df, raw_stock_path: str):
    """For each iteration boundary, compare the state mode A produces at hour
    k*168 against the state mode B iter k+1 actually loads.

    Re-solves mode A at cutoff = k*168 (model is identical, stock extraction
    differs) and compares the resulting stock_df to mode B iter k's *output*
    stock_df.
    """
    print(f"\n{'='*70}\nPER-ITER STATE DEVIATION  (mode A projected vs mode B actual)\n{'='*70}")
    # Reuse the mode A planning horizon for each cutoff solve.
    n = len(captures_b)
    for k in range(1, n):  # check cutoffs at end of week 1..n-1 (input to iter k+1)
        cutoff_weeks = k
        # Solve mode A again with this cutoff (deterministic model)
        print(f"\n--- end of week {k} (input to mode B iter {k+1}) ---")
        cap_alt = run_mode(
            planning_weeks_fn=lambda i: N_WEEKS,
            decision_weeks=cutoff_weeks,
            n_iters=1,
            voyages_df=voyages_df, demand_df=demand_df, cap_df=cap_df,
            stock_dir=stock_dir_a / f"alt-cutoff-w{k}", raw_stock_path=raw_stock_path,
            label=f"Mode A (cutoff at week {k})",
        )[0]

        # Mode A's projected stock at hour k*168 — read the saved CSV
        a_stock_path = stock_dir_a / f"alt-cutoff-w{k}" / f"{cap_alt.label}.csv"
        if not a_stock_path.exists():
            print(f"  [WARN] mode A stock for cutoff w{k} missing: {a_stock_path}")
            continue
        a_stock_df = pd.read_csv(a_stock_path)
        a_state = _normalize_stock(a_stock_df, cap_alt.cfg["epoch_dt"])

        # Mode B's actual input state for iter k+1 = iter k's output stock CSV
        b_stock_path = Path("results/compare-horizon-modes/stock-B") / f"{captures_b[k-1].label}.csv"
        if not b_stock_path.exists():
            print(f"  [WARN] mode B stock for iter {k} missing: {b_stock_path}")
            continue
        b_stock_df = pd.read_csv(b_stock_path)
        b_state = _normalize_stock(b_stock_df, captures_b[k-1].cfg["epoch_dt"])

        # Aggregate by (kind, port, fullEmpty)
        a_agg = _summarize_state(a_state)
        b_agg = _summarize_state(b_state)

        all_keys = sorted(set(a_agg) | set(b_agg))
        a_tot = sum(a_agg.values())
        b_tot = sum(b_agg.values())
        print(f"  Mode A projected: {a_tot} carrier containers in stock state")
        print(f"  Mode B actual:    {b_tot} carrier containers in stock state")
        print(f"  Δ totals:         {b_tot - a_tot:+d}")

        diffs = []
        for key in all_keys:
            a = a_agg.get(key, 0)
            b = b_agg.get(key, 0)
            if a != b:
                diffs.append((key, a, b, b - a))
        if not diffs:
            print(f"  [OK] state matches across all (kind, port, fullEmpty) buckets")
        else:
            print(f"  [DIVERGE] {len(diffs)} bucket(s) differ:")
            # Print top 15 by |Δ|
            diffs.sort(key=lambda t: -abs(t[3]))
            for (kind, port, fe), a, b, d in diffs[:15]:
                print(f"    {kind:>14}  {port:<25}  {fe:<5}  A={a:>5}  B={b:>5}  Δ={d:+d}")
            if len(diffs) > 15:
                tail = sum(abs(d[3]) for d in diffs[15:])
                print(f"    ... and {len(diffs)-15} more (sum |Δ| = {tail})")


def diff_state_transfer(captures_b: list[IterCapture]):
    """Per-iteration state-transfer diagnostics."""
    print(f"\n{'='*70}\nSTATE-TRANSFER DIAGNOSTICS  (mode B, between iterations)\n{'='*70}")

    for i in range(len(captures_b) - 1):
        cap_i  = captures_b[i]
        cap_n  = captures_b[i + 1]
        print(f"\n--- iter {i+1} → iter {i+2} ---")

        # 1. Commodities unfulfilled in iter i (looking at planning horizon
        #    BEYOND its decision window, i.e., still re-attemptable later).
        comms_i = cap_i.commodities_df
        ful_i = cap_i.results.fulfillment_df.set_index("Commodity")["Fulfilled"].to_dict()
        cutoff_i = cap_i.cfg["stock_cutoff_hour"]

        # commodities depending in iter i's planning beyond decision window
        future_comms = comms_i[
            (comms_i["DepartureTime"] >= cutoff_i) &
            ~comms_i["Commodity"].str.startswith(("CARRY_", "TLEG_"))
        ].copy()
        future_comms["Fulfilled"] = future_comms["Commodity"].map(ful_i).fillna(0)
        unfilled = future_comms[future_comms["Fulfilled"] < future_comms["Count"] - 1e-6]
        n_unfilled = len(unfilled)
        n_unfilled_units = int((unfilled["Count"] - unfilled["Fulfilled"]).sum())

        # 2. Same commodities in iter i+1? Match by stable signature
        # (Year, Week, Origin, Destination, ContainerType, ContainerSize, Owner)
        # since DLAY_ commodities re-emit raw bookings under synthetic IDs.
        comms_n = cap_n.commodities_df.copy()
        sig_cols = ["Year", "Week", "Origin", "Destination",
                    "ContainerType", "ContainerSize", "Owner"]
        n_sigs = set(zip(*[comms_n[c].astype(str) for c in sig_cols]))
        unf_sigs = list(zip(*[unfilled[c].astype(str) for c in sig_cols]))
        absent_mask = [s not in n_sigs for s in unf_sigs]
        absent = unfilled[absent_mask]
        print(f"  Iter {i+1} unfulfilled future commodities (depart > decision cutoff): "
              f"{n_unfilled} groups, {n_unfilled_units} containers")
        print(f"    Of these, {len(absent)} groups absent from iter {i+2}'s commodity list "
              f"({int((absent['Count'] - absent['Fulfilled']).sum())} containers — "
              f"silently dropped after fix)")

        # 3. CARRY_/TLEG_ time-shift
        # For each CARRY_/TLEG_ in iter i+1, look up the originating commodity
        # in iter i and compare its actual arrival hour from flows_df vs the
        # newly-assigned departure hour.
        carry_comms_n = cap_n.commodities_df[
            cap_n.commodities_df["Commodity"].str.startswith(("CARRY_", "TLEG_"))
        ]
        if len(carry_comms_n) == 0:
            print(f"  No CARRY_/TLEG_ injected into iter {i+2}.")
            continue

        # Cap_i.results.flows_df gives us actual arrival hours of crossing flows.
        # Build: (orig_commodity, arr_port) → arr_hour
        flows_i = cap_i.results.flows_df
        cutoff = cap_i.cfg["stock_cutoff_hour"]
        crossing = flows_i[
            ~flows_i["Commodity"].str.startswith("EMPTY_") &
            (flows_i["DepPort"] != flows_i["ArrPort"]) &
            (flows_i["DepHour"].astype(int) < cutoff) &
            (flows_i["ArrHour"].astype(int) > cutoff) &
            (flows_i["Flow"] > 0)
        ]
        # arr_hour relative to iter_i epoch — convert to wall datetime for comparison
        epoch_i = cap_i.cfg["epoch_dt"]
        epoch_n = cap_n.cfg["epoch_dt"]
        epoch_offset_h = int((epoch_n - epoch_i).total_seconds() / 3600)

        # build expected (arr_port, ct, cs, eo) → (arr_hour_relative_to_n)
        expected_arrival = defaultdict(list)
        for _, r in crossing.iterrows():
            arr_h_in_n = int(r["ArrHour"]) - epoch_offset_h
            expected_arrival[r["ArrPort"]].append(arr_h_in_n)

        # Compare
        shifts = []
        for _, c in carry_comms_n.iterrows():
            origin = c["Origin"]
            new_dep = int(c["DepartureTime"])
            arrivals_at_origin = expected_arrival.get(origin, [])
            if not arrivals_at_origin:
                shifts.append((c["Commodity"], None, new_dep, None))
                continue
            # closest expected arrival
            best = min(arrivals_at_origin, key=lambda h: abs(h - new_dep))
            shifts.append((c["Commodity"], best, new_dep, new_dep - best))

        finite = [s for s in shifts if s[3] is not None]
        if finite:
            avg_abs = sum(abs(s[3]) for s in finite) / len(finite)
            max_abs = max(abs(s[3]) for s in finite)
            print(f"  CARRY_/TLEG_ time-shift: {len(finite)} commodities, "
                  f"avg |Δ|={avg_abs:.1f}h, max |Δ|={max_abs}h")
            big = [s for s in finite if abs(s[3]) > 24]
            if big:
                print(f"    {len(big)} have |Δ|>24h (significant). Examples:")
                for k, exp, got, d in big[:5]:
                    print(f"      {k}: expected dep≈{exp}h, got dep={got}h, Δ={d:+d}h")
        if any(s[1] is None for s in shifts):
            print(f"    [WARN] {sum(1 for s in shifts if s[1] is None)} CARRY_/TLEG_ "
                  f"have no matching arrival flow in iter {i+1}'s solution")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    voyages_df = pd.read_csv(VOYAGES_PATH, parse_dates=["etaDateTime", "etdDateTime"])
    demand_df  = pd.read_csv(DEMAND_PATH)
    cap_df     = pd.read_csv(CAP_PATH, sep=r"\s+")
    raw_stock_path = f"data/raw/Stock_0101{START_YEAR}.csv"

    # Use harness-private stock dirs to avoid colliding with results/stock/
    out_dir = Path("results/compare-horizon-modes")
    out_dir.mkdir(parents=True, exist_ok=True)
    stock_dir_a = out_dir / "stock-A"
    stock_dir_b = out_dir / "stock-B"

    # ── Mode A: single solve ─────────────────────────────────────────────────
    captures_a = run_mode(
        planning_weeks_fn=lambda i: N_WEEKS,
        decision_weeks=N_WEEKS,
        n_iters=1,
        voyages_df=voyages_df, demand_df=demand_df, cap_df=cap_df,
        stock_dir=stock_dir_a, raw_stock_path=raw_stock_path,
        label=f"Mode A: 1 iter, plan={N_WEEKS}w decision={N_WEEKS}w",
    )
    cap_a = captures_a[0]

    # ── Mode B: rolling, shrinking horizon ───────────────────────────────────
    captures_b = run_mode(
        planning_weeks_fn=lambda i: N_WEEKS - i,
        decision_weeks=1,
        n_iters=N_WEEKS,
        voyages_df=voyages_df, demand_df=demand_df, cap_df=cap_df,
        stock_dir=stock_dir_b, raw_stock_path=raw_stock_path,
        label=f"Mode B: {N_WEEKS} iters, plan shrinking, decision=1w",
    )

    # ── Diffs ────────────────────────────────────────────────────────────────
    diff_iter0_identity(cap_a, captures_b[0])
    diff_per_week(cap_a, captures_b)
    diff_state_transfer(captures_b)
    diff_per_iter_state(stock_dir_a, captures_b, voyages_df, demand_df, cap_df, raw_stock_path)

    # Persist a JSON summary for follow-up
    import json
    summary = {
        "mode_a": {
            "total_demand": int(cap_a.results.fulfillment_df["Demand"].sum()),
            "total_fulfilled": int(cap_a.results.fulfillment_df["Fulfilled"].sum()),
            "solve_time_s": round(cap_a.solve_time, 2),
        },
        "mode_b": [
            {
                "iter": c.i + 1,
                "label": c.label,
                "demand": int(c.results.fulfillment_df["Demand"].sum()),
                "fulfilled": int(c.results.fulfillment_df["Fulfilled"].sum()),
                "n_carry_in": len(c.carryover_in),
                "n_trunc_in": len(c.truncated_in),
                "solve_time_s": round(c.solve_time, 2),
            }
            for c in captures_b
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
