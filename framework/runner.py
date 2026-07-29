import sys
import time
import importlib.util
from collections import defaultdict
from datetime import date, timedelta, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,"data")
from clean.eimskip_network import PORT_NAMES, MAIN_LINES 

LINE_LABEL = {l: l.replace(" Line", "") for l in MAIN_LINES}
MAIN_LINES_SET = set(MAIN_LINES)


START_YEAR             = 2025   # ISO year of first iteration
START_WEEK             = 1      # ISO week of first iteration
PLANNING_HORIZON_WEEKS = 3      # weeks the model optimises over per iteration
DECISION_HORIZON_WEEKS = 1      # weeks that are "locked in" per iteration
NUM_ITERATIONS         = 10      # total rolling-horizon iterations
TRUNCATE_HORIZON_AT_END = False # cap H per iter so lookahead never exceeds the
                                # remaining (NUM_ITERATIONS-i)*DECISION_HORIZON_WEEKS
SCENARIO               = "baseline"  # "baseline" | "light" | "moderate" | "heavy"
                                     # (see pipeline/augment_demand.py --all)
VERBOSE_STOCK          = False  # print per-row VESSEL stock injection diagnostics
VERBOSE_UNFULFILLED    = True  # print detailed unfulfilled demand diagnosis

# Optimisation model:
#   "models/max-demand-fulfilment.py"  — strict O→D node fulfilment (baseline)
#   "models/min-lateness-fulfilment.py" — fulfil-as-soon-as-possible with lateness penalty
MODEL_FILE             = "models/max-demand-fulfilment.py"

# Allow overrides from sweep scripts via environment variables.
import os as _os
SCENARIO   = _os.environ.get("RUNNER_SCENARIO",   SCENARIO)
MODEL_FILE = _os.environ.get("RUNNER_MODEL_FILE", MODEL_FILE)
_env_iters = _os.environ.get("RUNNER_NUM_ITERATIONS")
if _env_iters:
    NUM_ITERATIONS = int(_env_iters)
_env_horizon = _os.environ.get("RUNNER_PLANNING_HORIZON_WEEKS")
if _env_horizon:
    PLANNING_HORIZON_WEEKS = int(_env_horizon)
_env_decision = _os.environ.get("RUNNER_DECISION_HORIZON_WEEKS")
if _env_decision:
    DECISION_HORIZON_WEEKS = int(_env_decision)
_env_year = _os.environ.get("RUNNER_START_YEAR")
if _env_year:
    START_YEAR = int(_env_year)
_env_week = _os.environ.get("RUNNER_START_WEEK")
if _env_week:
    START_WEEK = int(_env_week)


FORECAST_SIGMA = float(_os.environ.get("RUNNER_FORECAST_SIGMA", 0.0))
FORECAST_SEED  = int(_os.environ.get("RUNNER_FORECAST_SEED", 0))


DISABLE_VESSEL = _os.environ.get("RUNNER_DISABLE_VESSEL", "").strip().upper()
DISABLE_ITERS  = {int(x) for x in _os.environ.get("RUNNER_DISABLE_ITERS", "").split(",") if x.strip()}


RESULTS_DIR = _os.environ.get("RUNNER_RESULTS_DIR", "results").rstrip("/")



def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_arcs_mod  = _load("arcs",       "pipeline/2_construct_arcs_voyages.py")
_comm_mod  = _load("commodities", "pipeline/3_construct_commodities_voyages.py")
_model_mod = _load("model",       MODEL_FILE)

build_arcs        = _arcs_mod.build_arcs
build_commodities = _comm_mod.build_commodities
solve             = _model_mod.solve
ModelResults      = _model_mod.ModelResults


# Helpers

def _iso_monday(year: int, week: int) -> date:
    """Return the date of Monday in the given ISO year/week."""
    jan4 = date(year, 1, 4)
    return jan4 - timedelta(days=jan4.weekday()) + timedelta(weeks=week - 1)


def advance_by_weeks(year: int, week: int, n: int) -> tuple[int, int]:
    """Advance (year, week) by n ISO weeks."""
    monday = _iso_monday(year, week) + timedelta(weeks=n)
    iso    = monday.isocalendar()
    return iso[0], iso[1]


def build_calendar_weeks(start_year: int, start_week: int, n_weeks: int) -> list:
    """Return list of (year, week) tuples for n_weeks starting at start_year/start_week."""
    weeks = []
    for i in range(n_weeks):
        monday = _iso_monday(start_year, start_week) + timedelta(weeks=i)
        iso    = monday.isocalendar()
        weeks.append((iso[0], iso[1]))
    return weeks


def compute_epoch_dt(voyages_df: pd.DataFrame,
                     calendar_weeks: list,
                     horizon_monday: date) -> datetime:

    _ = (voyages_df, calendar_weeks)  
    return datetime(horizon_monday.year, horizon_monday.month, horizon_monday.day)


def build_config_params(start_year: int,
                        start_week: int,
                        planning_weeks: int,
                        decision_weeks: int,
                        voyages_df: pd.DataFrame) -> dict:

    calendar_weeks = build_calendar_weeks(start_year, start_week, planning_weeks)
    horizon_monday = _iso_monday(start_year, start_week)
    epoch_dt       = compute_epoch_dt(voyages_df, calendar_weeks, horizon_monday)
    H              = planning_weeks * 168
    stock_cutoff   = decision_weeks * 168
    return {
        "epoch_hour_offset": 0,
        "H":                 H,
        "horizon_end":       H,
        "stock_cutoff_hour": stock_cutoff,
        "epoch_dt":          epoch_dt,
        "calendar_weeks":    calendar_weeks,
        "horizon_monday":    horizon_monday,
        "verbose_unfulfilled": VERBOSE_UNFULFILLED,
        "lambda_cost":           float(_os.environ.get("RUNNER_LAMBDA_COST", 0.01)),
        "mu_lateness":           float(_os.environ.get("RUNNER_MU_LATENESS", 0.1)),
        "mu_reefer":             float(_os.environ.get("RUNNER_MU_REEFER", 0.0)),
        "max_delay_days":        int(_os.environ.get("RUNNER_MAX_DELAY_DAYS", 0)),
        "fractional_lateness":   _os.environ.get("RUNNER_FRACTIONAL_LATENESS", "0") == "1",
        "terminal_empty_reward": float(_os.environ.get("RUNNER_TERMINAL_EMPTY_REWARD", 0.0)),
        "terminal_empty_ports":  [p.strip() for p in _os.environ.get("RUNNER_TERMINAL_EMPTY_PORTS", "").split(",") if p.strip()],
        "idle_empty_penalty":    float(_os.environ.get("RUNNER_IDLE_EMPTY_PENALTY", 0.0)),
        "idle_empty_ports":      [p.strip() for p in _os.environ.get("RUNNER_IDLE_EMPTY_PORTS", "").split(",") if p.strip()],
    }


# Demand uncertainty

def perturb_future_demand(commodities_df: pd.DataFrame, cfg: dict,
                          sigma: float, seed: int, iteration: int,
                          verbose: bool = True) -> pd.DataFrame:
    """Corrupt the model's *forecast* of future-week demand.

    The decision week (week 0 of the planning horizon) is left at its true
    realised value; demand departing in week w >= 1 is scaled by (1 + eps) with
    eps ~ Normal(0, sigma * w), so forecast error grows linearly the further
    ahead the week is. The perturbation is mean-preserving in expectation and
    re-drawn every iteration (seed folds in `iteration`), so a given calendar
    week's forecast is refreshed — and ultimately revealed exactly — as it rolls
    toward the decision window.

    sigma <= 0 returns the frame unchanged (perfect foresight / deterministic
    rolling horizon, identical to the baseline run).
    """
    if sigma <= 0 or commodities_df.empty:
        return commodities_df

    out = commodities_df.copy()
    dep = out["DepartureTime"].to_numpy(dtype=float)
    week_ahead = np.maximum(np.floor(dep / 168.0).astype(int), 0)

    rng = np.random.default_rng((seed, iteration))
    counts = out["Count"].to_numpy(dtype=float)
    eps = rng.normal(0.0, sigma * week_ahead)
    eps[week_ahead == 0] = 0.0            # decision week is always true
    new_counts = np.maximum(0, np.round(counts * (1.0 + eps))).astype(int)

    out["Count"] = new_counts
    before = int(commodities_df["Count"].sum())


    n_dropped = int((new_counts == 0).sum())
    out = out[out["Count"] > 0].reset_index(drop=True)

    if verbose:
        after  = int(out["Count"].sum())
        n_pert = int((week_ahead > 0).sum())
        print(f"  Forecast noise (sigma={sigma}, seed={seed}, iter={iteration}): "
              f"perturbed {n_pert} future-week rows, dropped {n_dropped} to zero, "
              f"demand {before} -> {after} ({after-before:+d})")

    return out


# Vessel unavailable stress test

def disable_vessel(voyages_df: pd.DataFrame, code: str, iters: set,
                   iteration: int) -> pd.DataFrame:
    """Remove a vessel's voyages for a disrupted iteration.

    `code` is a 3-char vessel prefix and `iters` a set of 1-based iteration
    numbers in which the vessel is unavailable. When this iteration is disrupted
    the vessel's voyage rows are dropped, so its sailings, capacity, network
    connectivity and commodity feasibility all vanish consistently for that
    iteration only; later iterations see the vessel again (the caller's
    voyages_df is never mutated).

    No-op (returns the frame unchanged) when no vessel is named or this
    iteration is not disrupted — baseline runs are untouched.
    """
    if not code or (iteration + 1) not in iters:
        return voyages_df

    mask = voyages_df["voyage"].str[:3].str.upper() == code
    print(f"  DISRUPTION: vessel {code} unavailable in iteration {iteration + 1} "
          f"({int(mask.sum())} voyage rows removed)")
    return voyages_df[~mask].copy()


#  Port/vessel dict construction 

def build_full_port_vessel_dicts(voyages_df: pd.DataFrame) -> tuple[dict, dict]:
    """Port/vessel dicts spanning ALL voyages in the dataset.

    Used for commodity filtering so the set of bookings admitted to the model
    is horizon-independent: a booking's leg-2 vessel/port being only visible
    in a future week shouldn't prevent the booking from being constructed.
    The horizon-bounded variant is still used for arc construction and stock
    port filtering, so the model only optimises over the active H window.
    """
    v = voyages_df[voyages_df["tradeRouteName"].isin(MAIN_LINES_SET)].copy()
    v["vessel"] = v["voyage"].str[:3].str.upper()

    port_lines = (
        v.groupby("portID")["tradeRouteName"]
        .apply(lambda s: tuple(sorted({LINE_LABEL[l] for l in s if l in LINE_LABEL})))
        .to_dict()
    )
    port_dict = {
        port: (PORT_NAMES.get(port, port),) + lines
        for port, lines in sorted(port_lines.items())
    }
    vessel_lines = (
        v.groupby("vessel")["tradeRouteName"]
        .apply(lambda s: tuple(sorted({LINE_LABEL[l] for l in s if l in LINE_LABEL})))
        .to_dict()
    )
    return port_dict, vessel_lines


def build_port_vessel_dicts(voyages_df: pd.DataFrame,
                            calendar_weeks: list) -> tuple[dict, dict]:

    first_year, first_week = calendar_weeks[0]
    last_year,  last_week  = calendar_weeks[-1]
    horizon_start = pd.Timestamp(_iso_monday(first_year, first_week))
    horizon_end   = pd.Timestamp(_iso_monday(last_year,  last_week)) + pd.Timedelta(days=7)

    in_window = (
        (voyages_df["etdDateTime"] >= horizon_start) &
        (voyages_df["etdDateTime"] <  horizon_end)
    )
    voyages_in_horizon = voyages_df.loc[in_window, "voyage"].unique()

    v = voyages_df[
        voyages_df["voyage"].isin(voyages_in_horizon) &
        voyages_df["tradeRouteName"].isin(MAIN_LINES_SET)
    ].copy()
    v["vessel"] = v["voyage"].str[:3].str.upper()

    port_lines = (
        v.groupby("portID")["tradeRouteName"]
        .apply(lambda s: tuple(sorted({LINE_LABEL[l] for l in s if l in LINE_LABEL})))
        .to_dict()
    )
    port_dict = {
        port: (PORT_NAMES.get(port, port),) + lines
        for port, lines in sorted(port_lines.items())
    }

    vessel_lines = (
        v.groupby("vessel")["tradeRouteName"]
        .apply(lambda s: tuple(sorted({LINE_LABEL[l] for l in s if l in LINE_LABEL})))
        .to_dict()
    )

    return port_dict, vessel_lines


# Stock

def load_stock_df(iteration: int,
                  start_year: int,
                  start_week: int,
                  decision_horizon_weeks: int) -> pd.DataFrame:
    """Load stock CSV for the given iteration."""
    if iteration == 0:
        path = f"data/raw/Stock_0101{start_year}.csv"
    else:
        prev_year, prev_week = advance_by_weeks(
            start_year, start_week, (iteration - 1) * decision_horizon_weeks
        )
        path = f"{RESULTS_DIR}/stock/{prev_year}w{prev_week}.csv"
    return pd.read_csv(path)


def filter_stock_to_ports(stock_df: pd.DataFrame, valid_ports: set) -> pd.DataFrame:
    """Keep only stock rows relevant to the current network's ports."""
    port_mask = (
        (stock_df["Location"] != "VESSEL") &
        stock_df["Location"].isin(valid_ports)
    )
    vessel_mask = (
        (stock_df["Location"] == "VESSEL") &
        (stock_df["Last location"].isin(valid_ports) |
         stock_df["Next location"].isin(valid_ports))
    )
    return stock_df[port_mask | vessel_mask].copy()



def _find_vessel_arrival(voyages_df, epoch_dt, last_loc, next_loc):

    best = (None, None)
    for voyage, group in voyages_df.groupby("voyage"):
        route = group.sort_values("etaDateTime")
        ports = list(route["portID"])
        if last_loc not in ports or next_loc not in ports:
            continue
        last_idx = ports.index(last_loc)
        # next_loc must come after last_loc in the route
        try:
            next_idx = ports[last_idx + 1:].index(next_loc) + last_idx + 1
        except ValueError:
            continue
        dep_row = route.iloc[last_idx]
        arr_row = route.iloc[next_idx]
        # The vessel must have departed last_loc before epoch_dt
        if dep_row["etdDateTime"] >= epoch_dt:
            continue
        dep_h = (dep_row["etdDateTime"] - epoch_dt).total_seconds() / 3600
        arr_h = (arr_row["etaDateTime"] - epoch_dt).total_seconds() / 3600
        if best[1] is None or arr_h > best[1]:
            best = (int(dep_h), int(arr_h))
    return best


def _find_voyage_route(voyages_df, epoch_dt, last_loc, next_loc):
    best = None  # (arr_h, voyage, route_slice)
    for voyage, group in voyages_df.groupby("voyage"):
        route = group.sort_values("etaDateTime").reset_index(drop=True)
        ports = list(route["portID"])
        if last_loc not in ports or next_loc not in ports:
            continue
        last_idx = ports.index(last_loc)
        try:
            next_idx = ports[last_idx + 1:].index(next_loc) + last_idx + 1
        except ValueError:
            continue
        dep_row = route.iloc[last_idx]
        arr_row = route.iloc[next_idx]
        if dep_row["etdDateTime"] >= epoch_dt:
            continue
        arr_h = (arr_row["etaDateTime"] - epoch_dt).total_seconds() / 3600
        if best is None or arr_h > best[0]:
            best = (arr_h, voyage, route.iloc[last_idx:next_idx + 1])

    if best is None:
        return None

    _, voyage, route_slice = best
    stops = []
    for _, r in route_slice.iterrows():
        etd_h = (r["etdDateTime"] - epoch_dt).total_seconds() / 3600
        eta_h = (r["etaDateTime"] - epoch_dt).total_seconds() / 3600
        stops.append((r["portID"], int(etd_h), int(eta_h)))
    vessel_code = str(voyage)[:3].upper()
    return vessel_code, stops


def _teu_for_size(cs) -> int:
    try:
        sz = float(cs)
    except (TypeError, ValueError):
        return 2
    if sz == 20.0:
        return 1
    if sz in (40.0, 45.0):
        return 2
    return 2


def reserve_in_transit_capacity(stock_df: pd.DataFrame,
                                arcs_df: pd.DataFrame,
                                voyages_df: pd.DataFrame,
                                cfg: dict,
                                verbose: bool = False) -> pd.DataFrame:
    if stock_df.empty or "Location" not in stock_df.columns:
        return arcs_df

    epoch_dt = cfg["epoch_dt"]
    H        = cfg["H"]

    sail_mask = arcs_df["DepPort"] != arcs_df["ArrPort"]
    sail_idx  = arcs_df.index[sail_mask]

    per_vessel_reserved: dict = defaultdict(int)
    total_reserved = 0
    rows_processed = 0
    rows_skipped   = 0

    for _, row in stock_df.iterrows():
        if str(row.get("Location", "")).strip().upper() != "VESSEL":
            continue
        if str(row.get("Owner", "")).strip().upper() != "CARRIER":
            continue
        last_loc = str(row.get("Last location", "")).strip()
        next_loc = str(row.get("Next location", "")).strip()
        if not last_loc or not next_loc or last_loc == "nan" or next_loc == "nan":
            continue
        try:
            count = int(row["count"])
        except (TypeError, ValueError, KeyError):
            continue
        if count <= 0:
            continue
        cs = row.get("ContainerSize")
        teu_per = _teu_for_size(cs)
        teu_total = count * teu_per

        result = _find_voyage_route(voyages_df, epoch_dt, last_loc, next_loc)
        if result is None:
            rows_skipped += 1
            continue
        vessel_code, stops = result

        for i in range(len(stops) - 1):
            p_i, etd_i, _   = stops[i]
            p_j, _,    eta_j = stops[i + 1]
            # Only segments at least partially in horizon stand a chance of matching
            if eta_j <= 0 or etd_i >= H:
                continue
            mask = (
                sail_mask &
                (arcs_df["Vessel"]  == vessel_code) &
                (arcs_df["DepPort"] == p_i) &
                (arcs_df["ArrPort"] == p_j)
            )
            cands = arcs_df[mask]
            if cands.empty:
                continue
            # Pick arc with DepHour closest to segment etd
            j = (cands["DepHour"] - etd_i).abs().idxmin()
            new_cap = max(float(arcs_df.at[j, "Capacity"]) - teu_total, 0.0)
            arcs_df.at[j, "Capacity"] = new_cap
            per_vessel_reserved[vessel_code] += teu_total
            total_reserved += teu_total

        rows_processed += 1

    if verbose and total_reserved > 0:
        print(f"  In-transit capacity reserved: {total_reserved} TEU "
              f"across {rows_processed} stock rows ({rows_skipped} skipped)")
        for v in sorted(per_vessel_reserved):
            print(f"    {v}: {per_vessel_reserved[v]} TEU")

    _ = sail_idx  
    return arcs_df


def process_stock_to_supply(stock_df: pd.DataFrame,
                             arcs_df: pd.DataFrame,
                             stock_cutoff_hour: int = 0,
                             epoch_dt=None,
                             voyages_df=None,
                             verbose: bool = False) -> tuple:
    from datetime import timedelta

    def _to_dt(h):
        return (epoch_dt + timedelta(hours=int(h))).strftime("%d %b %H:%M") if epoch_dt else str(h)

    # Build earliest node time per port
    nodes = set(zip(
        pd.concat([arcs_df["DepPort"], arcs_df["ArrPort"]]),
        pd.concat([arcs_df["DepHour"].astype(int), arcs_df["ArrHour"].astype(int)])
    ))
    earliest_time_per_port: dict[str, int] = {}
    for port, h in nodes:
        if port not in earliest_time_per_port or h < earliest_time_per_port[port]:
            earliest_time_per_port[port] = h

    # Build node times per port (for closest-node matching)
    port_node_times: dict[str, list[int]] = defaultdict(list)
    for port, h in nodes:
        port_node_times[port].append(h)

    # Build vessel arrival hour lookup from sailing arcs (direct legs)
    sailing_arcs = arcs_df[arcs_df["DepPort"] != arcs_df["ArrPort"]]
    vessel_arrival_hour: dict[tuple, int] = {}
    for _, arc in sailing_arcs.sort_values("ArrHour").iterrows():
        key = (arc["DepPort"], arc["ArrPort"])
        if key not in vessel_arrival_hour:
            vessel_arrival_hour[key] = int(arc["ArrHour"])

    supply_at_port: dict[tuple, int] = defaultdict(int)   # (port, ct, cs) → count
    supply_at_node: dict[tuple, int] = defaultdict(int)   # (node, ct, cs) → count
    vessel_late_rows: list = []   # VESSEL rows with arr_h > stock_cutoff_hour

    # Drop/redirect bookkeeping — printed at end so issues aren't silent
    drop_log: dict[str, int] = defaultdict(int)         # reason → container count
    drop_examples: dict[str, list] = defaultdict(list)  # reason → up to 3 sample rows
    def _log_drop(reason: str, n: int, row=None):
        drop_log[reason] += n
        if row is not None and len(drop_examples[reason]) < 3:
            drop_examples[reason].append(
                f"{str(row.get('Location','')).strip()}|"
                f"{str(row.get('Last location','')).strip()}→"
                f"{str(row.get('Next location','')).strip()}|"
                f"{str(row.get('FullEmpty','')).strip()}|"
                f"{str(row.get('ContainerType','')).strip()}/"
                f"{str(row.get('ContainerSize','')).strip()}|"
                f"{str(row.get('Owner','')).strip()}|n={n}"
            )

    def _inject_vessel(next_loc, arr_h, last_loc, ct, cs, count):
        """Inject a VESSEL container at (next_loc, arr_h), snapping to closest node.

        If arr_h > cutoff, the container bypasses the model entirely and is
        carried forward via vessel_late_rows for the next iteration.
        """
        if stock_cutoff_hour and arr_h > stock_cutoff_hour:
            vessel_late_rows.append({
                "Location": "VESSEL",
                "Last location": last_loc,
                "Next location": next_loc,
                "ArrivalTime": arr_h,
                "FullEmpty": "Empty",
                "With customer": "False",
                "ContainerType": ct,
                "ContainerSize": float(cs),
                "Owner": "CARRIER",
                "count": count,
            })
            return  
        if next_loc in port_node_times:
            valid_times = [t for t in port_node_times[next_loc]
                           if not stock_cutoff_hour or t <= stock_cutoff_hour]
            if valid_times:
                future_nodes = [t for t in valid_times if t >= arr_h]
                if future_nodes:
                    closest_h = min(future_nodes)
                else:
                    closest_h = max(valid_times)
            else:
                closest_h = min(port_node_times[next_loc])
            supply_at_node[((next_loc, closest_h), ct, cs)] += count
        else:
            supply_at_port[(next_loc, ct, cs)] += count

    for _, row in stock_df.iterrows():
        try:
            row_count = int(row["count"])
        except (TypeError, ValueError, KeyError):
            row_count = 0
        if str(row["Owner"]).strip().upper() != "CARRIER":
            _log_drop("non-carrier owner (untracked)", row_count, row)
            continue

        location      = str(row["Location"]).strip()
        last_location = str(row.get("Last location", "")).strip()
        next_location = str(row.get("Next location", "")).strip()

        raw_type = str(row["ContainerType"]).strip()
        ct = raw_type

        try:
            cs = int(float(row["ContainerSize"]))
        except (ValueError, TypeError):
            _log_drop("unparseable container size", row_count, row)
            continue
        count = int(row["count"])

        # Containers "with customer" — return at ArrivalTime
        if str(row.get("With customer", "")).strip().lower() == "true":
            raw_at = row.get("ArrivalTime")
            if pd.notna(raw_at) and str(raw_at).strip() not in ("", "nan"):
                try:
                    parsed_dt = pd.to_datetime(str(raw_at).strip())
                    arr_h = int((parsed_dt - epoch_dt).total_seconds() / 3600)
                    # If return is after cutoff, carry forward only 
                    if stock_cutoff_hour and arr_h > stock_cutoff_hour:
                        vessel_late_rows.append({
                            "Location": location,
                            "Last location": "", "Next location": "",
                            "ArrivalTime": arr_h,
                            "FullEmpty": "Empty",
                            "With customer": "True",
                            "ContainerType": ct,
                            "ContainerSize": float(cs),
                            "Owner": "CARRIER",
                            "count": count,
                        })
                        continue
                    if location in port_node_times:
                        valid_times = [t for t in port_node_times[location]
                                       if not stock_cutoff_hour or t <= stock_cutoff_hour]
                        if valid_times:
                            future_nodes = [t for t in valid_times if t >= arr_h]
                            if future_nodes:
                                closest_h = min(future_nodes)
                            else:
                                closest_h = max(valid_times)
                        else:
                            closest_h = min(port_node_times[location])
                        supply_at_node[((location, closest_h), ct, cs)] += count
                    else:
                        supply_at_port[(location, ct, cs)] += count
                    continue
                except Exception:
                    pass
            if location in earliest_time_per_port:
                _log_drop("with-customer: ArrivalTime missing/unparseable, injected at earliest node",
                          count, row)
                supply_at_port[(location, ct, cs)] += count
            else:
                _log_drop("with-customer: location not in network", count, row)
            continue

        if location.upper() == "VESSEL":
            if not next_location or next_location == "nan":
                # No next location — inject at last location instead
                if last_location in earliest_time_per_port:
                    _log_drop("VESSEL no Next location, injected at Last", count, row)
                    supply_at_port[(last_location, ct, cs)] += count
                else:
                    _log_drop("VESSEL no Next + Last not in network", count, row)
                continue

            # 1. Check for ArrivalTime from previous iteration's stock
            raw_at = row.get("ArrivalTime")
            if pd.notna(raw_at) and str(raw_at).strip() not in ("", "nan"):
                at_str = str(raw_at).strip()
                try:
                    parsed_dt = pd.to_datetime(at_str)
                    arr_h = int((parsed_dt - epoch_dt).total_seconds() / 3600)
                    _inject_vessel(next_location, arr_h, last_location, ct, cs, count)
                    continue
                except Exception:
                    pass  # fall through to arc lookup

            # 2. Direct arc lookup
            arr_h = vessel_arrival_hour.get((last_location, next_location))
            if arr_h is not None:
                _inject_vessel(next_location, arr_h, last_location, ct, cs, count)
                continue

            # 3. Route-finding: trace vessel routes for indirect legs
            dep_h, arr_h = _find_vessel_arrival(voyages_df, epoch_dt, last_location, next_location) if voyages_df is not None else (None, None)
            if arr_h is not None and arr_h >= 0:
                if verbose:
                    print(f"  VESSEL {last_location}→{next_location}: traced via route, "
                          f"dep hour {dep_h} ({_to_dt(dep_h)}), arr hour {arr_h} ({_to_dt(arr_h)})")
                _inject_vessel(next_location, arr_h, last_location, ct, cs, count)
                continue

            # 4. Fallback: already arrived or no route — inject at earliest node
            dest = next_location
            if dest not in earliest_time_per_port:
                if dest.startswith("IS") and "IS REY" in earliest_time_per_port:
                    _log_drop(f"VESSEL Next={next_location} not in network, redirected to IS REY",
                              count, row)
                    dest = "IS REY"
                else:
                    _log_drop(f"VESSEL Next={next_location} not in network, dropped",
                              count, row)
                    if verbose:
                        print(f"  Warning: VESSEL row skipped — {next_location} not in network.")
                    continue
            if arr_h is not None and arr_h < 0:
                _log_drop("VESSEL already arrived (arr_h<0), injected at earliest node",
                          count, row)
                if verbose:
                    print(f"  VESSEL {last_location}→{dest}: already arrived "
                          f"(arr hour {arr_h}, {_to_dt(arr_h)}), injecting at earliest node.")
            else:
                _log_drop(f"VESSEL no route {last_location}→{dest}, injected at earliest node",
                          count, row)
                if verbose:
                    print(f"  Warning: no route {last_location}→{dest}; "
                          f"injecting VESSEL stock at earliest node.")
            supply_at_port[(dest, ct, cs)] += count
        else:
            if not location or location not in earliest_time_per_port:
                # Icelandic port not in network → redirect to IS REY
                if location.startswith("IS") and "IS REY" in earliest_time_per_port:
                    _log_drop(f"port row {location} not in network, redirected to IS REY",
                              count, row)
                    supply_at_port[("IS REY", ct, cs)] += count
                else:
                    _log_drop(f"port row {location or '<empty>'} not in network, dropped",
                              count, row)
                continue
            supply_at_port[(location, ct, cs)] += count

    # Build supply dict
    supply: dict = {}
    for port, earliest_h in earliest_time_per_port.items():
        node = (port, earliest_h)
        for (p, ct, cs), amount in supply_at_port.items():
            if p == port and amount:
                key = (node, ct, cs, "Carrier")
                supply[key] = supply.get(key, 0) + amount

    for (node, ct, cs), amount in supply_at_node.items():
        if node in nodes and amount:
            key = (node, ct, cs, "Carrier")
            supply[key] = supply.get(key, 0) + amount

    # Late VESSEL rows (arr_h > cutoff) deferred to next iteration's stock CSV
    if vessel_late_rows:
        n_late = sum(int(r.get("count", 0)) for r in vessel_late_rows)
        print(f"  Deferred {n_late} containers across {len(vessel_late_rows)} VESSEL rows "
              f"(arrive after stock cutoff)")

    if drop_log:
        print("  Stock row drops/redirects:")
        for reason in sorted(drop_log, key=lambda r: -drop_log[r]):
            n = drop_log[reason]
            print(f"    [{n:>5}]  {reason}")
            if verbose:
                for ex in drop_examples[reason]:
                    print(f"             e.g. {ex}")

    return supply, vessel_late_rows


# Carryover demand

def extract_carryover_demands(results: ModelResults,
                              commodities_df: pd.DataFrame,
                              cfg: dict) -> dict:
    from datetime import timedelta as _td
    cutoff = cfg["stock_cutoff_hour"]
    epoch_dt = cfg["epoch_dt"]

    crossing = results.flows_df[
        ~results.flows_df["Commodity"].str.startswith("EMPTY_") &
        (results.flows_df["DepPort"] != results.flows_df["ArrPort"]) &
        (results.flows_df["DepHour"].astype(int) < cutoff) &
        (results.flows_df["ArrHour"].astype(int) > cutoff) &
        (results.flows_df["Flow"] > 0)
    ].copy()

    if crossing.empty:
        return {}

    comm_lookup = (
        commodities_df
        .set_index("Commodity")[["Destination", "ContainerType", "ContainerSize", "Owner"]]
        .to_dict("index")
    )

    carryover: dict[tuple, int] = defaultdict(int)
    for _, row in crossing.iterrows():
        k = row["Commodity"]
        if k not in comm_lookup:
            continue
        spec = comm_lookup[k]
        arr_port = row["ArrPort"]
        arr_h = int(row["ArrHour"])
        arr_wall = (epoch_dt + _td(hours=arr_h)).isoformat()
        key = (
            arr_port,
            spec["Destination"],
            spec["ContainerType"],
            float(spec["ContainerSize"]),
            spec["Owner"],
            arr_wall,
        )
        carryover[key] += int(row["Flow"])

    return dict(carryover)


def build_iter0_intransit_carryover(stock_df: pd.DataFrame,
                                    commodities_df: pd.DataFrame,
                                    arcs_df: pd.DataFrame,
                                    voyages_df: pd.DataFrame,
                                    cfg: dict) -> tuple[dict, pd.DataFrame]:
    if stock_df.empty or commodities_df.empty:
        return {}, commodities_df

    epoch_dt = cfg["epoch_dt"]

    # Direct-arc arrival lookup, mirrors process_stock_to_supply
    sailing = arcs_df[arcs_df["DepPort"] != arcs_df["ArrPort"]]
    vessel_arrival_hour: dict[tuple, int] = {}
    for _, arc in sailing.sort_values("ArrHour").iterrows():
        key = (arc["DepPort"], arc["ArrPort"])
        if key not in vessel_arrival_hour:
            vessel_arrival_hour[key] = int(arc["ArrHour"])

    # Mutable working copy of commodities — we'll decrement Count and drop empties
    work = commodities_df.copy()
    work["Count"] = work["Count"].astype(int)
    # Owner case in commodities_df is upper ("CARRIER"); normalise for matching
    work["_owner_u"] = work["Owner"].astype(str).str.upper()
    work["_cs_f"]    = work["ContainerSize"].astype(float)

    carryover: dict[tuple, int] = defaultdict(int)
    matched_total = 0
    unmatched_total = 0
    matched_groups = 0
    unmatched_rows: list[str] = []

    for _, row in stock_df.iterrows():
        if str(row.get("Location", "")).strip().upper() != "VESSEL":
            continue
        if str(row.get("FullEmpty", "")).strip().lower() != "full":
            continue
        if str(row.get("Owner", "")).strip().upper() != "CARRIER":
            continue
        last_loc = str(row.get("Last location", "")).strip()
        next_loc = str(row.get("Next location", "")).strip()
        if not last_loc or not next_loc or next_loc == "nan":
            continue
        try:
            count = int(row["count"])
        except (TypeError, ValueError, KeyError):
            continue
        if count <= 0:
            continue
        ct = str(row["ContainerType"]).strip()
        try:
            cs = float(row["ContainerSize"])
        except (TypeError, ValueError):
            continue

        # Resolve arrival hour at next_loc → wall-clock ISO
        arr_h = vessel_arrival_hour.get((last_loc, next_loc))
        if arr_h is None and voyages_df is not None:
            _, arr_h = _find_vessel_arrival(voyages_df, epoch_dt, last_loc, next_loc)
        if arr_h is None:
            arr_h = 0
        arr_wall_iso = (epoch_dt + timedelta(hours=int(arr_h))).isoformat()

        # Match against bookings: Origin=last_loc, Destination=next_loc, type/size/owner
        cand_mask = (
            (work["Origin"] == last_loc) &
            (work["Destination"] == next_loc) &
            (work["ContainerType"] == ct) &
            (work["_cs_f"] == cs) &
            (work["_owner_u"] == "CARRIER") &
            (work["Count"] > 0)
        )
        cands = work[cand_mask].sort_values("DepartureTime")

        remaining = count
        for idx, brow in cands.iterrows():
            if remaining <= 0:
                break
            take = min(int(brow["Count"]), remaining)
            orig_dest = brow.get("FinalDestination") or brow["Destination"]
            key = (next_loc, orig_dest, ct, cs, "Carrier", arr_wall_iso)
            carryover[key] += take
            work.at[idx, "Count"] = int(brow["Count"]) - take
            remaining -= take
            matched_total += take
            matched_groups += 1

        if remaining > 0:
            unmatched_total += remaining
            unmatched_rows.append(
                f"{last_loc}→{next_loc}  {ct}/{int(cs)}ft  n={remaining}"
            )

    work = work[work["Count"] > 0].drop(columns=["_owner_u", "_cs_f"]).reset_index(drop=True)

    if matched_total or unmatched_total:
        print(f"  Iter-0 in-transit carryover: matched {matched_total} containers "
              f"to {matched_groups} bookings, {unmatched_total} unmatched "
              f"(empty injected at Next location only — no booking found)")
        if unmatched_rows:
            print(f"  Unmatched in-transit stock rows ({len(unmatched_rows)}):")
            for line in unmatched_rows:
                print(f"    {line}")

    return dict(carryover), work


def extract_truncated_second_legs(results: ModelResults,
                                  commodities_df: pd.DataFrame,
                                  cfg: dict | None = None) -> dict:
    from datetime import timedelta as _td

    if "FinalDestination" not in commodities_df.columns:
        return {}

    truncated = commodities_df[
        commodities_df["Truncated"].fillna(False).astype(bool) &
        commodities_df["FinalDestination"].notna() &
        (commodities_df["Destination"] != commodities_df["FinalDestination"])
    ]
    if truncated.empty:
        return {}

    fulfilled_lookup = results.fulfillment_df.set_index("Commodity")["Fulfilled"].to_dict()
    epoch_dt = cfg["epoch_dt"] if cfg else None

    # Per-commodity latest arrival hour at the transshipment port
    flows = results.flows_df
    arrivals_by_k: dict[str, int] = {}
    if not flows.empty:
        trunc_ks = set(truncated["Commodity"])
        sub = flows[flows["Commodity"].isin(trunc_ks) & (flows["Flow"] > 0)]
        for k_, port_, arr_h_ in zip(sub["Commodity"], sub["ArrPort"], sub["ArrHour"]):
            tk = truncated[truncated["Commodity"] == k_]
            if tk.empty:
                continue
            transship_port = tk.iloc[0]["Destination"]
            if port_ != transship_port:
                continue
            cur = arrivals_by_k.get(k_)
            if cur is None or int(arr_h_) > cur:
                arrivals_by_k[k_] = int(arr_h_)

    second_legs: dict[tuple, int] = defaultdict(int)
    for _, row in truncated.iterrows():
        k = row["Commodity"]
        fulfilled = fulfilled_lookup.get(k, 0)
        if fulfilled < 1e-6:
            continue
        arr_h = arrivals_by_k.get(k)
        if arr_h is None:
            # Fallback: original ArrivalTime from the commodity row
            try:
                arr_h = int(row["ArrivalTime"])
            except (TypeError, ValueError, KeyError):
                arr_h = 0
        arr_wall = ((epoch_dt + _td(hours=arr_h)).isoformat()
                    if epoch_dt is not None else str(arr_h))
        key = (
            row["Destination"],
            row["FinalDestination"],
            row["ContainerType"],
            float(row["ContainerSize"]),
            row["Owner"],
            arr_wall,
        )
        second_legs[key] += int(fulfilled)

    if second_legs:
        n = sum(second_legs.values())
        print(f"  Truncated second legs to carry forward: {len(second_legs)} groups, {n} containers")

    return dict(second_legs)


def _merge_carryover_dicts(*dicts: dict) -> dict:

    merged: dict = {}
    for d in dicts:
        for key, count in d.items():
            merged[key] = merged.get(key, 0) + count
    return merged


def build_carryover_commodities(carryover: dict,
                                arcs_df: pd.DataFrame,
                                year: int = 0,
                                week: int = 0,
                                target_epoch_dt=None,
                                horizon_end: int | None = None) -> tuple[pd.DataFrame, dict, dict]:

    from datetime import datetime as _dt

    # Port node times
    port_node_times: dict[str, list[int]] = defaultdict(list)
    for _, arc in arcs_df.iterrows():
        port_node_times[arc["DepPort"]].append(int(arc["DepHour"]))
        port_node_times[arc["ArrPort"]].append(int(arc["ArrHour"]))

    carryover_rows = []
    supply_additions: dict = {}
    deferred: dict = {}

    for c_idx, (key, count) in enumerate(carryover.items()):
        # Backwards-compat: accept old 5-tuple keys (no wall-clock time).
        if len(key) == 6:
            arr_port, orig_dest, ct, cs, eo, arr_wall_iso = key
        else:
            arr_port, orig_dest, ct, cs, eo = key
            arr_wall_iso = None

        eo = eo.strip().title() if isinstance(eo, str) else eo
        if arr_port not in port_node_times or orig_dest not in port_node_times:
            d_key = (arr_port, orig_dest, ct, cs, eo, arr_wall_iso) if arr_wall_iso \
                    else (arr_port, orig_dest, ct, cs, eo)
            deferred[d_key] = deferred.get(d_key, 0) + count
            continue

        # Compute earliest feasible departure for this carryover in the new
        # epoch: re-base wall-clock arrival, then snap up to nearest node ≥ it.
        port_times_sorted = sorted(set(port_node_times[arr_port]))
        if arr_wall_iso and target_epoch_dt is not None:
            try:
                arr_wall_dt = _dt.fromisoformat(arr_wall_iso)
                rebased_h = int((arr_wall_dt - target_epoch_dt).total_seconds() / 3600)
            except (ValueError, TypeError):
                rebased_h = port_times_sorted[0]
        else:
            rebased_h = port_times_sorted[0]
        # Snap up to the first port node at or after rebased_h. If none, the
        # vessel has effectively already passed — fall back to earliest node
        # (model will route as best it can).
        future_nodes = [t for t in port_times_sorted if t >= rebased_h]
        dep_h = future_nodes[0] if future_nodes else port_times_sorted[0]

        # Inject matching empty at carryover origin node
        if eo == "Carrier":
            node = (arr_port, dep_h)
            sup_key = (node, ct, cs, eo)
            supply_additions[sup_key] = supply_additions.get(sup_key, 0) + count

        # Skip demand if already arriving at final destination
        if arr_port == orig_dest:
            continue


        dest_times_sorted = sorted(set(port_node_times[orig_dest]))
        if horizon_end is not None:
            in_horizon_arrivals = [t for t in dest_times_sorted
                                   if dep_h < t <= horizon_end]
        else:
            in_horizon_arrivals = [t for t in dest_times_sorted if t > dep_h]
        if in_horizon_arrivals:
            arr_h = max(in_horizon_arrivals)
        else:
            # No in-horizon arrival node downstream of dep_h — defer this
            # carryover to a later iteration when the destination has nodes.
            d_key = (arr_port, orig_dest, ct, cs, eo, arr_wall_iso) if arr_wall_iso \
                    else (arr_port, orig_dest, ct, cs, eo)
            deferred[d_key] = deferred.get(d_key, 0) + count
            # Roll back the supply addition we may have just added.
            if eo == "Carrier":
                node = (arr_port, dep_h)
                sup_key = (node, ct, cs, eo)
                supply_additions[sup_key] = supply_additions.get(sup_key, 0) - count
                if supply_additions[sup_key] <= 0:
                    supply_additions.pop(sup_key, None)
            continue

        carryover_rows.append({
            "Commodity":     f"CARRY_{c_idx}_{arr_port}_{orig_dest}",
            "Year":          year,
            "Week":          week,
            "Origin":        arr_port,
            "DepartureTime": dep_h,
            "Destination":   orig_dest,
            "ArrivalTime":   arr_h,
            "ContainerType": ct,
            "ContainerSize": cs,
            "Owner":         eo,
            "Count":         count,
            "Truncated":     False,
            "FinalDestination": orig_dest,
        })

    if carryover_rows:
        n_containers = sum(r["Count"] for r in carryover_rows)
        print(f"  Added {len(carryover_rows)} carryover commodity groups "
              f"({n_containers} containers in transit)")

    if deferred:
        lanes = {(k[0], k[1]) for k in deferred}
        n_def = sum(deferred.values())
        print(f"  Deferred {n_def} carryover containers across {len(lanes)} lane(s) "
              f"whose port(s) are absent from this horizon — will retry next iteration")

    return (pd.DataFrame(carryover_rows) if carryover_rows else pd.DataFrame(),
            supply_additions, deferred)




def extract_unfulfilled_synth_carryover(results: ModelResults,
                                        commodities_df: pd.DataFrame,
                                        cfg: dict) -> dict:

    from datetime import timedelta as _td

    if commodities_df is None or commodities_df.empty:
        return {}
    if results.fulfillment_df is None or results.fulfillment_df.empty:
        return {}

    epoch_dt = cfg["epoch_dt"]
    fulf = (
        results.fulfillment_df
        .set_index("Commodity")[["Demand", "Fulfilled"]]
        .to_dict("index")
    )

    is_synth = commodities_df["Commodity"].astype(str).str.startswith(("CARRY_", "TLEG_"))
    synth = commodities_df[is_synth]
    if synth.empty:
        return {}

    carry: dict[tuple, int] = defaultdict(int)
    for _, row in synth.iterrows():
        k = row["Commodity"]
        f = fulf.get(k, {})
        unmet = int(round(float(f.get("Demand", 0)) - float(f.get("Fulfilled", 0))))
        if unmet <= 0:
            continue
        dep_h = int(row["DepartureTime"])
        dep_wall = (epoch_dt + _td(hours=dep_h)).isoformat()
        orig_dest = row.get("FinalDestination") or row["Destination"]
        key = (
            row["Origin"],
            orig_dest,
            str(row["ContainerType"]),
            float(row["ContainerSize"]),
            "Carrier",
            dep_wall,
        )
        carry[key] += unmet

    if carry:
        n = sum(carry.values())
        print(f"  Unfulfilled CARRY_/TLEG_ rolled forward: {n} containers across "
              f"{len(carry)} groups")
    return dict(carry)


def extract_delayed_commodities(results: ModelResults, 
                                commodities_df: pd.DataFrame,
                                cfg: dict) -> list[dict]:

    from datetime import timedelta as _td

    cutoff = cfg["stock_cutoff_hour"]
    epoch_dt = cfg["epoch_dt"]

    if commodities_df is None or commodities_df.empty:
        return []

    is_synth = commodities_df["Commodity"].str.startswith(("CARRY_", "TLEG_", "DLAY_"))
    future_mask = commodities_df["DepartureTime"].astype(int) >= cutoff
    candidates = commodities_df[~is_synth & future_mask]
    if candidates.empty:
        return []

    delayed: list[dict] = []
    for _, row in candidates.iterrows():
        demand = int(round(float(row["Count"])))
        if demand <= 0:
            continue

        dep_h = int(row["DepartureTime"])
        arr_h = int(row["ArrivalTime"])
        dep_wall = (epoch_dt + _td(hours=dep_h)).isoformat()
        arr_wall = (epoch_dt + _td(hours=arr_h)).isoformat()

        delayed.append({
            "Origin":           row["Origin"],
            "Destination":      row["Destination"],
            "FinalDestination": row.get("FinalDestination", row["Destination"]),
            "Truncated":        bool(row.get("Truncated", False)),
            "ContainerType":    row["ContainerType"],
            "ContainerSize":    float(row["ContainerSize"]),
            "Owner":            row["Owner"],
            "OrigYear":         int(row["Year"]) if pd.notna(row.get("Year")) else 0,
            "OrigWeek":         int(row["Week"]) if pd.notna(row.get("Week")) else 0,
            "Count":            demand,
            "DepWall":          dep_wall,
            "ArrWall":          arr_wall,
        })

    if delayed:
        n = sum(d["Count"] for d in delayed)
        print(f"  Future-window commodities to carry forward (DLAY candidates): "
              f"{len(delayed)} groups, {n} containers")
    return delayed


def _delay_signature(d) -> tuple:

    return (
        int(d["OrigYear"]),
        int(d["OrigWeek"]),
        str(d["Origin"]),
        str(d["Destination"]),
        str(d["ContainerType"]),
        float(d["ContainerSize"]),
        str(d["Owner"]),
    )


def build_delayed_commodities(delayed: list[dict],
                              commodities_df: pd.DataFrame,
                              arcs_df: pd.DataFrame,
                              target_epoch_dt,
                              horizon_end: int | None = None) -> tuple[pd.DataFrame, list[dict]]:

    from datetime import datetime as _dt

    if not delayed:
        return pd.DataFrame(), []

    # Port set for this iteration
    ports_in_horizon = set(arcs_df["DepPort"]).union(set(arcs_df["ArrPort"]))

    # Index fresh bookings by signature (sum of counts, list of indices)
    fresh_idx_by_sig: dict[tuple, list[int]] = defaultdict(list)
    if commodities_df is not None and not commodities_df.empty:
        is_synth = commodities_df["Commodity"].str.startswith(("CARRY_", "TLEG_", "DLAY_"))
        for idx, r in commodities_df[~is_synth].iterrows():
            sig = (
                int(r["Year"]) if pd.notna(r.get("Year")) else 0,
                int(r["Week"]) if pd.notna(r.get("Week")) else 0,
                str(r["Origin"]),
                str(r["Destination"]),
                str(r["ContainerType"]),
                float(r["ContainerSize"]),
                str(r["Owner"]),
            )
            fresh_idx_by_sig[sig].append(idx)

    port_node_times: dict[str, list[int]] = defaultdict(list)
    for _, arc in arcs_df.iterrows():
        port_node_times[arc["DepPort"]].append(int(arc["DepHour"]))
        port_node_times[arc["ArrPort"]].append(int(arc["ArrHour"]))

    rows_to_inject: list[dict] = []
    deferred: list[dict] = []
    deduped_against_fresh = 0

    for d_idx, d in enumerate(delayed):
        sig = _delay_signature(d)

        # 1. Defer if origin or destination absent from this horizon's network
        if d["Origin"] not in ports_in_horizon or d["Destination"] not in ports_in_horizon:
            deferred.append(d)
            continue

        # 2. De-dup: if a fresh booking with the same signature exists, the
        #    booking has reappeared — don't inject a new commodity. We trust
        #    the fresh row to represent total demand; the delayed count is
        #    already implicit in it.
        if sig in fresh_idx_by_sig:
            deduped_against_fresh += d["Count"]
            continue

        # 3. Re-base wall-clock times into this iteration's epoch
        try:
            dep_wall = _dt.fromisoformat(d["DepWall"])
            arr_wall = _dt.fromisoformat(d["ArrWall"])
        except (ValueError, TypeError):
            deferred.append(d)
            continue
        dep_h = int((dep_wall - target_epoch_dt).total_seconds() / 3600)
        arr_h = int((arr_wall - target_epoch_dt).total_seconds() / 3600)

        # 4. Snap dep_h up to the next available origin node ≥ dep_h.
        port_times = sorted(set(port_node_times.get(d["Origin"], [])))
        if not port_times:
            deferred.append(d)
            continue
        future_nodes = [t for t in port_times if t >= dep_h]
        snapped_dep = future_nodes[0] if future_nodes else port_times[0]
        if snapped_dep > dep_h:
            shift = snapped_dep - dep_h
            arr_h += shift
            dep_h = snapped_dep

        # 5. Pick an in-horizon arrival node at the destination so the model
        # doesn't classify this commodity as over-horizon and absorb its full
        # cargo (which would silently lose the loaded empty).
        dest_times = sorted(set(port_node_times.get(d["Destination"], [])))
        if horizon_end is not None and dest_times:
            in_horizon = [t for t in dest_times if dep_h < t <= horizon_end]
            if in_horizon:
                arr_h = max(in_horizon)
            else:
                # No in-horizon arrival node downstream of dep_h — defer.
                deferred.append(d)
                continue
        elif dest_times:
            ge_dep = [t for t in dest_times if t > dep_h]
            if ge_dep:
                arr_h = max(ge_dep)

        if arr_h <= dep_h:
            arr_h = dep_h + 1

        rows_to_inject.append({
            "Commodity":        f"DLAY_{d_idx}_{d['Origin']}_{d['Destination']}",
            "Year":             d["OrigYear"],
            "Week":             d["OrigWeek"],
            "Origin":           d["Origin"],
            "DepartureTime":    dep_h,
            "Destination":      d["Destination"],
            "ArrivalTime":      arr_h,
            "ContainerType":    d["ContainerType"],
            "ContainerSize":    d["ContainerSize"],
            "Owner":            d["Owner"],
            "Count":            d["Count"],
            "Truncated":        d["Truncated"],
            "FinalDestination": d["FinalDestination"],
        })

    if rows_to_inject:
        n = sum(r["Count"] for r in rows_to_inject)
        print(f"  Added {len(rows_to_inject)} delayed commodity groups "
              f"({n} containers re-emitted as DLAY_)")
    if deduped_against_fresh:
        print(f"  De-duped {deduped_against_fresh} delayed containers against "
              f"fresh bookings already present in this horizon")
    if deferred:
        n = sum(d["Count"] for d in deferred)
        print(f"  Deferred {n} delayed containers — port(s) absent this iteration")

    return (pd.DataFrame(rows_to_inject) if rows_to_inject else pd.DataFrame(),
            deferred)


# Results

def save_results(results: ModelResults, label: str, arcs_df: pd.DataFrame, epoch_dt,
                 vessel_late_rows: list | None = None) -> pd.DataFrame:
    from datetime import timedelta


    skip = {s.strip() for s in _os.environ.get("RUNNER_SKIP_RESULTS", "").split(",") if s.strip()}
    skip.discard("stock")

    for subdir in ["stock", "model-flows", "vessel-utilisation", "demand-fulfilment"]:
        if subdir in skip:
            continue
        Path(f"{RESULTS_DIR}/{subdir}").mkdir(parents=True, exist_ok=True)

    stock_df = results.stock_df
    if vessel_late_rows:
        stock_df = pd.concat(
            [stock_df, pd.DataFrame(vessel_late_rows)], ignore_index=True
        )

    # Convert ArrivalTime from epoch hours to datetime string for VESSEL rows
    if "ArrivalTime" in stock_df.columns:
        stock_df["ArrivalTime"] = stock_df["ArrivalTime"].apply(
            lambda h: (epoch_dt + timedelta(hours=int(float(h)))).isoformat()
            if pd.notna(h) and str(h).strip() not in ("", "nan") else ""
        )
    stock_df.to_csv(f"{RESULTS_DIR}/stock/{label}.csv", index=False)
    if "model-flows" not in skip:
        results.flows_df.to_csv(f"{RESULTS_DIR}/model-flows/{label}.csv", index=False)
    if "demand-fulfilment" not in skip:
        results.fulfillment_df.to_csv(f"{RESULTS_DIR}/demand-fulfilment/{label}.csv", index=False)

    #  Vessel utilisation 
    def teu_of(commodity):
        for part in str(commodity).split("_"):
            try:
                sz = float(part)
                if sz in (20, 20.0):
                    return 1
                if sz in (40, 40.0, 45, 45.0):
                    return 2
            except ValueError:
                continue
        return 2

    def to_dt(h):
        return (epoch_dt + timedelta(hours=int(h))).strftime("%d %b %H:%M")

    flows = results.flows_df.copy()
    flows["TEU"]      = flows["Flow"] * flows["Commodity"].apply(teu_of)
    flows["is_empty"] = flows["Commodity"].str.startswith("EMPTY_")
    sail_flows = flows[flows["DepPort"] != flows["ArrPort"]].copy()

    if not sail_flows.empty:
        sail_arcs = arcs_df[
            (arcs_df["Vessel"] != "WAIT") & (~arcs_df["Line"].isin(["TRANS"]))
        ][["Line", "Vessel", "DepPort", "DepHour", "ArrPort", "ArrHour", "Capacity"]].copy()

        def lookup_capacity(dep_port, arr_port, vessel, dep_hour):
            matches = sail_arcs[
                (sail_arcs["DepPort"] == dep_port) &
                (sail_arcs["ArrPort"] == arr_port) &
                (sail_arcs["Vessel"]  == vessel)
            ]
            if matches.empty:
                return None, None
            closest = matches.iloc[(matches["DepHour"] - dep_hour).abs().argmin()]
            return closest["Capacity"], closest["Line"]

        arc_keys = ["DepPort", "DepHour", "ArrPort", "ArrHour", "Vessel"]
        cargo = (
            sail_flows[~sail_flows["is_empty"]]
            .groupby(arc_keys)["TEU"].sum().rename("CargoTEU")
        )
        empty = (
            sail_flows[sail_flows["is_empty"]]
            .groupby(arc_keys)["TEU"].sum().rename("EmptyTEU")
        )
        df = pd.concat([cargo, empty], axis=1).reset_index().fillna(0)
        df["TotalTEU"] = df["CargoTEU"] + df["EmptyTEU"]

        caps, lines = zip(*[
            lookup_capacity(r.DepPort, r.ArrPort, r.Vessel, r.DepHour)
            for _, r in df.iterrows()
        ])
        df["Capacity"] = caps
        df["Line"]     = lines

        df = df.dropna(subset=["Capacity"]).copy()
        df["Capacity"]   = df["Capacity"].astype(float)
        df["UnusedTEU"]  = (df["Capacity"] - df["TotalTEU"]).clip(lower=0)
        df["CargoFill%"] = (df["CargoTEU"] / df["Capacity"] * 100).round(1)
        df["TotalFill%"] = (df["TotalTEU"] / df["Capacity"] * 100).round(1)
        df["DepDate"]    = df["DepHour"].apply(to_dt)

        util = df[["Line", "Vessel", "DepPort", "DepHour", "DepDate", "ArrPort", "ArrHour",
                   "Capacity", "CargoTEU", "EmptyTEU", "UnusedTEU", "CargoFill%", "TotalFill%"]]
        util = util.sort_values(["Vessel", "DepHour"]).reset_index(drop=True)
        if "vessel-utilisation" not in skip:
            util.to_csv(f"{RESULTS_DIR}/vessel-utilisation/{label}.csv", index=False)

    print(f"  Results saved to {RESULTS_DIR}/*/{label}.csv")
    return stock_df


#  Carrier conservation check 

def check_carrier_conservation(initial_supply: dict, final_stock_df: pd.DataFrame,
                               vessel_late_rows: list | None = None):
    """Verify total Carrier containers are conserved and print per-port breakdown."""
    start_total = sum(v for (node, ct, cs, eo), v in initial_supply.items()
                      if eo == "Carrier")
    if vessel_late_rows:
        start_total += sum(r["count"] for r in vessel_late_rows
                           if str(r.get("Owner", "")).upper() == "CARRIER")

    carrier_end = final_stock_df[final_stock_df["Owner"].str.upper() == "CARRIER"]
    end_total   = int(carrier_end["count"].sum())

    if abs(start_total - end_total) > 1:
        print(f"\nWARNING: Carrier containers changed: {start_total} → {end_total} "
              f"(diff = {end_total - start_total:+d})")
    else:
        print(f"\nCarrier conservation OK: {start_total} containers")

    # Per-port breakdown (logic from analysis/stock-summary.py)
    if final_stock_df.empty:
        return

    end_df = final_stock_df.copy()
    end_df["Owner"] = end_df["Owner"].str.upper()
    c = end_df[end_df["Owner"] == "CARRIER"]
    end_at  = c[c["Location"] != "VESSEL"].groupby("Location")["count"].sum()
    end_arr = c[c["Location"] == "VESSEL"].groupby("Next location")["count"].sum()

    ports = sorted(set(end_at.index) | set(end_arr.index))
    if not ports:
        return

    print(f"  {'Port':<12}  {'At port':>9}  {'Arriving':>9}  {'Total':>9}")
    print("  " + "-" * 44)
    for port in ports:
        ea = int(end_at.get(port, 0))
        er = int(end_arr.get(port, 0))
        print(f"  {port:<12}  {ea:>9,}  {er:>9,}  {ea+er:>9,}")
    print(f"  {'TOTAL':<12}  {int(end_at.sum()):>9,}  {int(end_arr.sum()):>9,}  "
          f"{int(end_at.sum()) + int(end_arr.sum()):>9,}")


# Main  

def main():
    print("Loading data...")
    voyages_df = pd.read_csv(
        "data/clean/eimskip_voyages.csv",
        parse_dates=["etaDateTime", "etdDateTime"]
    )
    demand_path = f"data/augmented/{SCENARIO}/Eimskip_data_final.csv"
    demand_df = pd.read_csv(demand_path)
    print(f"Demand source: {demand_path}  (scenario={SCENARIO}, model={MODEL_FILE})")
    cap_df    = pd.read_csv("data/raw/vessel-capacities.csv", sep=r"\s+")

    iter_summaries: list = []        # per-iteration stats for end-of-run summary
    prev_carryover:      dict = {}   # carryover from previous iteration
    prev_truncated_legs: dict = {}   # truncated second legs from previous iteration
    pending_carryover:   dict = {}   # carryover deferred because port(s) were absent
    pending_truncated:   dict = {}   # truncated legs deferred for the same reason
    prev_delayed:        list = []   # unfulfilled future commodities to retry
    pending_delayed:     list = []   # delayed commodities deferred (port absent)
    prev_cfg:            dict = {}

    for i in range(NUM_ITERATIONS):
        year, week = advance_by_weeks(START_YEAR, START_WEEK, i * DECISION_HORIZON_WEEKS)
        label      = f"{year}w{week}"

        if TRUNCATE_HORIZON_AT_END:
            max_remaining = (NUM_ITERATIONS - i) * DECISION_HORIZON_WEEKS
            effective_planning = min(PLANNING_HORIZON_WEEKS, max_remaining)
        else:
            effective_planning = PLANNING_HORIZON_WEEKS
        effective_decision = min(DECISION_HORIZON_WEEKS, effective_planning)

        print(f"\n{'='*60}")
        print(f"Iteration {i+1}/{NUM_ITERATIONS}: {year} W{week:02d}  "
              f"(planning {effective_planning}w, decision {effective_decision}w)")
        print(f"{'='*60}")

        # 1. Build config params
        cfg = build_config_params(
            year, week, effective_planning, effective_decision, voyages_df
        )
        print(f"  Epoch start: {cfg['epoch_dt']:%d %b %Y}  |  "
              f"H={cfg['H']}h  |  cutoff={cfg['stock_cutoff_hour']}h")

        voyages_iter = disable_vessel(voyages_df, DISABLE_VESSEL, DISABLE_ITERS, i)

        # 2. Build port/vessel dicts + arcs (the time-space network)
        _t_net = time.perf_counter()
        port_dict, vessel_lines = build_port_vessel_dicts(voyages_iter, cfg["calendar_weeks"])

        full_port_dict, full_vessel_lines = build_full_port_vessel_dicts(voyages_df)
        print(f"  Network: {len(port_dict)} ports, {len(vessel_lines)} vessels "
              f"(commodity-filter network: {len(full_port_dict)}/{len(full_vessel_lines)})")

        # 3. Build arcs
        arcs_df = build_arcs(voyages_iter, cap_df, port_dict, cfg)
        _network_build_s = time.perf_counter() - _t_net

        # 4. Build commodities (historical demand for this horizon)
        _t_comm = time.perf_counter()
        commodities_df = build_commodities(demand_df, voyages_df, full_port_dict, full_vessel_lines, cfg)
        _commodity_build_s = time.perf_counter() - _t_comm
        print(f"  NETWORK_BUILD_S={_network_build_s:.3f}")
        print(f"  COMMODITY_BUILD_S={_commodity_build_s:.3f}")

        commodities_df = perturb_future_demand(
            commodities_df, cfg, FORECAST_SIGMA, FORECAST_SEED, i,
            verbose=VERBOSE_UNFULFILLED,
        )

        # 5. Load and process stock
        raw_stock    = load_stock_df(i, START_YEAR, START_WEEK, DECISION_HORIZON_WEEKS)
        filtered_stk = filter_stock_to_ports(raw_stock, set(port_dict.keys()))


        arcs_df = reserve_in_transit_capacity(
            filtered_stk, arcs_df, voyages_df, cfg, verbose=VERBOSE_STOCK
        )

        stock_for_supply = filtered_stk

        initial_sup, vessel_late_rows = process_stock_to_supply(
            stock_for_supply, arcs_df, cfg["stock_cutoff_hour"], cfg["epoch_dt"],
            voyages_df, verbose=VERBOSE_STOCK
        )
        carrier_start = sum(v for (node, ct, cs, eo), v in initial_sup.items()
                            if eo == "Carrier")
        print(f"  Initial Carrier supply: {carrier_start} containers")

        # 6. Prepend carryover demands. 
        if i == 0:
            iter0_carry, commodities_df = build_iter0_intransit_carryover(
                filtered_stk, commodities_df, arcs_df, voyages_df, cfg
            )
            prev_carryover = iter0_carry
        carryover_to_inject = _merge_carryover_dicts(prev_carryover, pending_carryover)
        if carryover_to_inject:
            carryover_df, _carryover_sup_unused, deferred = build_carryover_commodities(
                carryover_to_inject, arcs_df, year, week,
                target_epoch_dt=cfg["epoch_dt"],
                horizon_end=cfg["horizon_end"],
            )
            if not carryover_df.empty:
                commodities_df = pd.concat([carryover_df, commodities_df], ignore_index=True)
            pending_carryover = deferred
        else:
            pending_carryover = carryover_to_inject 
        legs_to_inject = _merge_carryover_dicts(prev_truncated_legs, pending_truncated)
        if i > 0 and legs_to_inject:
            leg_df, _, deferred_legs = build_carryover_commodities(
                legs_to_inject, arcs_df, year, week,
                target_epoch_dt=cfg["epoch_dt"],
                horizon_end=cfg["horizon_end"],
            )
            if not leg_df.empty:
                # Rename to TLEG_ prefix to distinguish from CARRY_
                leg_df["Commodity"] = leg_df["Commodity"].str.replace("CARRY_", "TLEG_", n=1)
                commodities_df = pd.concat([leg_df, commodities_df], ignore_index=True)
            pending_truncated = deferred_legs
        else:
            pending_truncated = legs_to_inject

  
        delayed_to_inject = list(prev_delayed) + list(pending_delayed)
        if i > 0 and delayed_to_inject:
            delay_df, deferred_delay = build_delayed_commodities(
                delayed_to_inject, commodities_df, arcs_df,
                target_epoch_dt=cfg["epoch_dt"],
                horizon_end=cfg["horizon_end"],
            )
            if not delay_df.empty:
                commodities_df = pd.concat([delay_df, commodities_df], ignore_index=True)
            pending_delayed = deferred_delay
        else:
            pending_delayed = delayed_to_inject

        # 7. Save intermediate arcs and commodities
        end_year, end_week = advance_by_weeks(year, week, PLANNING_HORIZON_WEEKS - 1)
        if PLANNING_HORIZON_WEEKS == 1:
            horizon_stem = f"{year}w{week}"
        elif end_year == year:
            horizon_stem = f"{year}w{week}-w{end_week}"
        else:
            horizon_stem = f"{year}w{week}-{end_year}w{end_week}"
        Path("data/processed/arcs").mkdir(parents=True, exist_ok=True)
        Path("data/processed/commodities").mkdir(parents=True, exist_ok=True)
        arcs_df.to_csv(f"data/processed/arcs/{horizon_stem}.csv", index=False)
        commodities_df.to_csv(f"data/processed/commodities/{horizon_stem}.csv", index=False)
        print(f"  Saved arcs/commodities → data/processed/*/{ horizon_stem}.csv")

        # 8. Solve
        print(f"  Solving: {len(arcs_df)} arcs, {len(commodities_df)} commodities...")
        results = solve(arcs_df, commodities_df, initial_sup, cfg)

        if results.status != "OPTIMAL":
            print(f"  Solve failed with status: {results.status}")
            break

        # 9. Extract carryover for next iteration
        prev_carryover = extract_carryover_demands(results, commodities_df, cfg)

        unfilled_synth = extract_unfulfilled_synth_carryover(results, commodities_df, cfg)
        for k, v in unfilled_synth.items():
            prev_carryover[k] = prev_carryover.get(k, 0) + v
        prev_truncated_legs = extract_truncated_second_legs(results, commodities_df, cfg)
        prev_delayed = extract_delayed_commodities(results, commodities_df, cfg)
        prev_cfg = cfg

        # 10. Save results
        final_stock_df = save_results(
            results, label, arcs_df, cfg["epoch_dt"], vessel_late_rows
        )

        # 11. Checks
        check_carrier_conservation(initial_sup, final_stock_df, vessel_late_rows)

        # Unfulfilled demand summary — decision window only

        cutoff = cfg["stock_cutoff_hour"]
        dw_comms = set(
            commodities_df.loc[commodities_df["DepartureTime"] < cutoff, "Commodity"]
        )
        dw_ful = results.fulfillment_df[results.fulfillment_df["Commodity"].isin(dw_comms)]
  
        is_synth_dw = dw_ful["Commodity"].astype(str).str.startswith(("CARRY_", "TLEG_", "DLAY_"))
        synth_unfilled = dw_ful[is_synth_dw & (dw_ful["Fulfilled"] < dw_ful["Demand"])]
        dw_unfulfilled = dw_ful[~is_synth_dw & (dw_ful["Fulfilled"] < dw_ful["Demand"])]
        dw_unmet = int((dw_unfulfilled["Demand"] - dw_unfulfilled["Fulfilled"]).sum())
        in_transit_unmet = int((synth_unfilled["Demand"] - synth_unfilled["Fulfilled"]).sum())
 
        dw_customer = dw_ful[~is_synth_dw]
        dw_demand = int(dw_customer["Demand"].sum())
        dw_fulfilled = int(dw_customer["Fulfilled"].sum())

        dw_rate = dw_fulfilled / dw_demand * 100 if dw_demand > 0 else 0
        print(f"\n  Decision window: {dw_fulfilled}/{dw_demand} fulfilled "
              f"+ {in_transit_unmet} in transit ({dw_rate:.1f}%), "
              f"{len(dw_unfulfilled)} unfulfilled commodities, "
              f"{dw_unmet} containers unmet")

        iter_summaries.append({
            "iter":             i + 1,
            "label":            label,
            "arcs":             len(arcs_df),
            "commodities":      len(commodities_df),
            "dw_demand":        dw_demand,
            "dw_fulfilled":     dw_fulfilled,
            "in_transit":       in_transit_unmet,
            "dw_unmet":         dw_unmet,
            "dw_unfilled_n":    len(dw_unfulfilled),
            "dw_rate":          dw_rate,
        })

    print(f"\n{'='*60}")
    print("Runner complete.")

    if len(iter_summaries) > 1:
        print(f"\n{'='*78}")
        print("Rolling-horizon summary")
        print(f"{'='*78}")
        header = (f"{'Iter':>4} {'Window':>10} {'Arcs':>6} {'Comms':>7} "
                  f"{'Demand':>8} {'Fulfilled':>10} {'InTransit':>10} "
                  f"{'Unmet':>7} {'Unfilled':>9} {'Rate%':>7}")
        print(header)
        print("-" * len(header))
        for s in iter_summaries:
            print(f"{s['iter']:>4} {s['label']:>10} {s['arcs']:>6} "
                  f"{s['commodities']:>7} {s['dw_demand']:>8} "
                  f"{s['dw_fulfilled']:>10} {s['in_transit']:>10} "
                  f"{s['dw_unmet']:>7} {s['dw_unfilled_n']:>9} "
                  f"{s['dw_rate']:>7.1f}")
        print("-" * len(header))
        tot_dem  = sum(s["dw_demand"]     for s in iter_summaries)
        tot_ful  = sum(s["dw_fulfilled"]  for s in iter_summaries)
        tot_trans = sum(s["in_transit"]   for s in iter_summaries)
        tot_unmet = sum(s["dw_unmet"]     for s in iter_summaries)
        tot_unfilled_n = sum(s["dw_unfilled_n"] for s in iter_summaries)
        tot_rate = (tot_ful / tot_dem * 100) if tot_dem > 0 else 0
        print(f"{'TOT':>4} {'':>10} {'':>6} {'':>7} "
              f"{tot_dem:>8} {tot_ful:>10} {tot_trans:>10} "
              f"{tot_unmet:>7} {tot_unfilled_n:>9} {tot_rate:>7.1f}")
        print(f"\n  Decision-window fulfillment across {len(iter_summaries)} iterations: "
              f"{tot_ful}/{tot_dem} fulfilled + {tot_trans} in transit "
              f"({tot_rate:.1f}%), {tot_unmet} containers unmet "
              f"across {tot_unfilled_n} commodities.")

    return iter_summaries


if __name__ == "__main__":
    main()
