"""
Build time-space arcs from the historical sailing schedule
Run automatically by runner.py


"""

import sys
import pandas as pd
from datetime import datetime, timedelta

MAIN_LINES           = {"Red Line", "Green Line", "Yellow Line", "Blue Line"}
ICELAND_PORTS        = {"IS REY", "IS GRT", "IS RFJ", "IS VES"}
TRANSSHIPMENT_PORTS  = {"IS REY", "FO THO", "DK AAR", "NL RTM"}
TRANS_COST           = 1.0


def build_arcs(voyages_df: pd.DataFrame,
               capacities_df: pd.DataFrame,
               port_dict: dict,
               cfg: dict) -> pd.DataFrame:

    EPOCH_HOUR_OFFSET = cfg["epoch_hour_offset"]
    H                 = cfg["H"]
    EPOCH_DT          = cfg["epoch_dt"]
    CALENDAR_WEEKS    = cfg["calendar_weeks"]

    vessel_capacity = {row["Vessel"].strip(): int(row["TEUs"])
                       for _, row in capacities_df.iterrows()}

    def to_epoch_hour(dt) -> int:
        return int((dt.to_pydatetime().replace(tzinfo=None) - EPOCH_DT).total_seconds() / 3600)



    vdf = voyages_df[voyages_df["tradeRouteName"].isin(MAIN_LINES)].copy()
    vdf = vdf[vdf["portID"].isin(port_dict)].copy()

    horizon_start = pd.Timestamp(EPOCH_DT)
    horizon_end   = horizon_start + timedelta(hours=H)


    voyages_in_horizon = vdf[
        (vdf["etdDateTime"] >= horizon_start) &
        (vdf["etdDateTime"] <  horizon_end)
    ]["voyage"].unique()

    vdf = vdf[vdf["voyage"].isin(voyages_in_horizon)].copy()
    vdf["vessel"] = vdf["voyage"].str[:3]

    #  Sailing arcs 

    sailing_arcs = []
    port_times_per_line: dict[str, dict[str, set]] = {}
    arrivals_trans   = []
    departures_trans = []

    for (voyage, line), group in vdf.groupby(["voyage", "tradeRouteName"]):
        vessel   = group["vessel"].iloc[0]
        capacity = vessel_capacity.get(vessel)
        if capacity is None:
            continue

        stops = group.sort_values("etaDateTime").reset_index(drop=True)
        pt    = port_times_per_line.setdefault(line, {})

        for _, s in stops.iterrows():
            port = s["portID"]
            eta  = to_epoch_hour(s["etaDateTime"])
            etd  = to_epoch_hour(s["etdDateTime"])
            pt.setdefault(port, set()).update([eta, etd])

        voyage_entered_horizon = False

        for i in range(len(stops) - 1):
            dep_port = stops.loc[i,   "portID"]
            arr_port = stops.loc[i+1, "portID"]

            if dep_port == arr_port:
                continue

            dep_hour = to_epoch_hour(stops.loc[i,   "etdDateTime"])
            arr_hour = to_epoch_hour(stops.loc[i+1, "etaDateTime"])

            if dep_hour >= arr_hour:
                continue

            if EPOCH_HOUR_OFFSET <= dep_hour < EPOCH_HOUR_OFFSET + H:
                voyage_entered_horizon = True

            if dep_hour < EPOCH_HOUR_OFFSET:
                continue
            if dep_hour >= EPOCH_HOUR_OFFSET + H and not voyage_entered_horizon:
                continue

            sailing_arcs.append({
                "Line":     line,
                "Vessel":   vessel,
                "DepPort":  dep_port,
                "DepHour":  dep_hour,
                "ArrPort":  arr_port,
                "ArrHour":  arr_hour,
                "Capacity": capacity,
                "Cost":     1.0,
            })

            if arr_port in TRANSSHIPMENT_PORTS:
                arrivals_trans.append((arr_port, arr_hour, line, vessel))
            if dep_port in TRANSSHIPMENT_PORTS:
                departures_trans.append((dep_port, dep_hour, line, vessel))

    # Wait arcs 

    wait_arcs = []

    for line, ports in port_times_per_line.items():
        for port, times in ports.items():
            valid_times = sorted(t for t in times if t >= EPOCH_HOUR_OFFSET)
            for t1, t2 in zip(valid_times, valid_times[1:]):
                cost = 0.0
                wait_arcs.append({
                    "Line":     line,
                    "Vessel":   "WAIT",
                    "DepPort":  port,
                    "DepHour":  t1,
                    "ArrPort":  port,
                    "ArrHour":  t2,
                    "Capacity": 10000,
                    "Cost":     cost,
                })

    #  Transshipment arcs 

    trans_arcs = []

    for port in TRANSSHIPMENT_PORTS:
        port_arr = [(h, l, v) for (p, h, l, v) in arrivals_trans   if p == port and h >= EPOCH_HOUR_OFFSET]
        port_dep = [(h, l, v) for (p, h, l, v) in departures_trans if p == port and h >= EPOCH_HOUR_OFFSET]

        for arr_hour, arr_line, arr_vessel in port_arr:
            for dep_hour, dep_line, dep_vessel in port_dep:
                if arr_line == dep_line:
                    continue
                if dep_hour <= arr_hour:
                    continue
                trans_arcs.append({
                    "Line":     "TRANS",
                    "Vessel":   f"{arr_vessel}->{dep_vessel}",
                    "DepPort":  port,
                    "DepHour":  arr_hour,
                    "ArrPort":  port,
                    "ArrHour":  dep_hour,
                    "Capacity": min(vessel_capacity.get(arr_vessel, 10000),
                                    vessel_capacity.get(dep_vessel, 10000)),
                    "Cost":     TRANS_COST,
                })

    #  Seed arcs 

    seed_arcs = []
    earliest_per_port_line: dict[tuple, int] = {}
    for arc in sailing_arcs:
        key = (arc["DepPort"], arc["Line"])
        h   = arc["DepHour"]
        if key not in earliest_per_port_line or h < earliest_per_port_line[key]:
            earliest_per_port_line[key] = h

    for (port, line), earliest_h in earliest_per_port_line.items():
        if earliest_h <= EPOCH_HOUR_OFFSET:
            continue
        seed_arcs.append({
            "Line":     line,
            "Vessel":   "WAIT",
            "DepPort":  port,
            "DepHour":  EPOCH_HOUR_OFFSET,
            "ArrPort":  port,
            "ArrHour":  earliest_h,
            "Capacity": 10000,
            "Cost":     0.0,
        })


    arcs_df = (
        pd.DataFrame(sailing_arcs + wait_arcs + trans_arcs + seed_arcs)
        .drop_duplicates()
        .sort_values(["Line", "Vessel", "DepHour"])
        .reset_index(drop=True)
    )

    n_sail = len(sailing_arcs)
    n_wait = len(wait_arcs)
    n_trans = len(trans_arcs)
    n_seed = len(seed_arcs)
    print(f"  Arcs: {len(arcs_df)} total "
          f"({n_sail} sailing + {n_wait} wait + {n_trans} transshipment + {n_seed} seed)")

    return arcs_df




if __name__ == "__main__":
    sys.path.insert(0, "pipeline")
    from config import NUM_WEEKS, H, EPOCH_HOUR_OFFSET, HORIZON_MONDAY, CALENDAR_WEEKS, ARCS_PATH

    sys.path.insert(0, "data")
    from processed.eimskip_routes_generated import port_dict

    voyages_df = pd.read_csv(
        "data/raw/Eimskip_voyages.csv",
        parse_dates=["etaDateTime", "etdDateTime"]
    )
    cap_df = pd.read_csv("data/raw/vessel-capacities.csv", sep=r"\s+")


    _week_mask = False
    for _yr, _wk in CALENDAR_WEEKS:
        _week_mask = _week_mask | ((voyages_df["year"] == _yr) & (voyages_df["week"] == _wk))
    _week_etds = voyages_df[_week_mask]["etdDateTime"]
    if not _week_etds.empty:
        _epoch_dt = _week_etds.min().normalize().to_pydatetime()
    else:
        _epoch_dt = datetime(HORIZON_MONDAY.year, HORIZON_MONDAY.month, HORIZON_MONDAY.day)

    cfg = {
        "epoch_hour_offset": EPOCH_HOUR_OFFSET,
        "H": H,
        "horizon_end": EPOCH_HOUR_OFFSET + H,
        "stock_cutoff_hour": H,
        "epoch_dt": _epoch_dt,
        "calendar_weeks": CALENDAR_WEEKS,
        "horizon_monday": HORIZON_MONDAY,
    }

    arcs_df = build_arcs(voyages_df, cap_df, port_dict, cfg)
    ARCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    arcs_df.to_csv(ARCS_PATH, index=False)
    print(f"Saved to {ARCS_PATH}  ({NUM_WEEKS} weeks)")
