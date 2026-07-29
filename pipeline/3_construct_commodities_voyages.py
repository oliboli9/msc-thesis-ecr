"""
Build commodities from the historical sailing schedule
Run automatically by runner.py


"""

import os
import sys
import pandas as pd
from datetime import datetime


MAIN_LINES = {"Red Line", "Green Line", "Yellow Line", "Blue Line"}
NO_VOYAGE2 = {"", "000", "nan", "NAN", " 000"}


def classify_container_type(code: str) -> str:
    """3rd character == 'R' → Reefer, anything else → Dry."""
    code = str(code).strip()
    return "Reefer" if len(code) >= 3 and code[2].upper() == "R" else "Dry"


def build_commodities(demand_df: pd.DataFrame,
                      voyages_df: pd.DataFrame,
                      port_dict: dict,
                      vessel_lines: dict,
                      cfg: dict) -> pd.DataFrame:
    EPOCH_HOUR_OFFSET = cfg["epoch_hour_offset"]
    H                 = cfg["H"]
    EPOCH_DT          = cfg["epoch_dt"]
    CALENDAR_WEEKS    = cfg["calendar_weeks"]
    horizon_end       = EPOCH_HOUR_OFFSET + H

    def to_epoch_hour(dt) -> int:
        return int((dt.to_pydatetime().replace(tzinfo=None) - EPOCH_DT).total_seconds() / 3600)


    vraw = voyages_df[
        voyages_df["tradeRouteName"].isin(MAIN_LINES) &
        voyages_df["portID"].isin(port_dict)
    ].copy()

    voyage_stops: dict[str, list[tuple]] = {}
    for _, row in vraw.iterrows():
        v = row["voyage"]
        voyage_stops.setdefault(v, []).append((
            row["portID"],
            to_epoch_hour(row["etaDateTime"]),
            to_epoch_hour(row["etdDateTime"]),
        ))
    for v in voyage_stops:
        voyage_stops[v].sort(key=lambda x: x[1])

    vessel_stops: dict[str, list[tuple]] = {}
    for voy, stops in voyage_stops.items():
        vc = voy[:3].upper()
        for stop in stops:
            vessel_stops.setdefault(vc, []).append(stop)
    for vc in vessel_stops:
        vessel_stops[vc] = sorted(set(vessel_stops[vc]), key=lambda x: x[1])


    horizon_end_dt = EPOCH_DT + pd.Timedelta(hours=horizon_end)
    active_voyages = set(
        voyages_df.loc[
            (voyages_df["etdDateTime"] >= EPOCH_DT) &
            (voyages_df["etdDateTime"] <  horizon_end_dt),
            "voyage",
        ].astype(str).unique()
    )
    df_filtered = demand_df[demand_df["Voyage 1"].astype(str).isin(active_voyages)].copy()

    if os.environ.get("INCLUDE_EMPTY_DEMAND") == "1":

        fe_mask = pd.Series(True, index=df_filtered.index)
    else:

        fe_mask = ((df_filtered["Full/Empty"] == "F") |
                   (df_filtered["Owner"].isin(["LEASED", "SHIPPER"])))
    df_full = df_filtered[
        fe_mask & (df_filtered["ContainerSize"].isin([20, 40, 45]))
    ].copy()


    def vessel_of(voyage_code) -> str:
        if pd.isna(voyage_code):
            return ""
        return str(voyage_code).strip()[:3].upper()

    v1_ok = df_full["Voyage 1"].apply(vessel_of).isin(vessel_lines)

    dropped_mask_v = ~v1_ok
    if dropped_mask_v.any():
        print(f"  Dropped {dropped_mask_v.sum()} rows with Voyage 1 vessel not in vessel_lines")
        if cfg.get("verbose_pipeline", False):
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 240):
                print(df_full[dropped_mask_v].to_string())
    df_full = df_full[v1_ok].copy()


    def assign_voyage_times(row) -> pd.Series:
        voyage1 = row.get("Voyage 1")
        load1   = row.get("Load 1")

        if pd.isna(voyage1) or pd.isna(load1) or str(voyage1) not in voyage_stops:
            return pd.Series([pd.NA, pd.NA, pd.NA, False, pd.NA])

        stops1 = voyage_stops[str(voyage1)]

        dep_idx = next(
            (i for i, (p, eta, etd) in enumerate(stops1) if p == load1),
            None
        )
        if dep_idx is None:
            return pd.Series([pd.NA, pd.NA, pd.NA, False, pd.NA])

        dep_hour = stops1[dep_idx][2]   # etd

        voyage2 = str(row.get("Voyage 2", "")).strip()
        has_v2  = voyage2 not in NO_VOYAGE2


        if has_v2 and voyage2 not in voyage_stops:
            discharge1 = row.get("Discharge 1")
            discharge2 = row.get("Discharge 2")
            if pd.isna(discharge1) or pd.isna(discharge2):
                return pd.Series([pd.NA, pd.NA, pd.NA, False, pd.NA])
            arr_idx = next(
                (i for i, (p, eta, etd) in enumerate(stops1)
                 if i > dep_idx and p == discharge1),
                None
            )
            if arr_idx is None:
                return pd.Series([pd.NA, pd.NA, pd.NA, False, pd.NA])
            arr_hour = stops1[arr_idx][1]
            if arr_hour <= dep_hour:
                return pd.Series([pd.NA, pd.NA, pd.NA, False, pd.NA])
            return pd.Series([int(dep_hour), int(arr_hour), discharge1, True, discharge2])

        if has_v2 and voyage2 in voyage_stops:
            discharge2 = row.get("Discharge 2")
            discharge1 = row.get("Discharge 1")
            if pd.isna(discharge2):
                return pd.Series([pd.NA, pd.NA, pd.NA, False, pd.NA])

            stops2 = voyage_stops[voyage2]

            v1_arr_at_trans_idx = next(
                (i for i, (p, eta, etd) in enumerate(stops1) if i > dep_idx and p == discharge1),
                None
            )
            if v1_arr_at_trans_idx is None:
                return pd.Series([pd.NA, pd.NA, pd.NA, False, pd.NA])
            v1_arr_at_trans = stops1[v1_arr_at_trans_idx][1]

            v2_dep_at_trans = next(
                (etd for p, eta, etd in stops2 if p == discharge1 and etd > v1_arr_at_trans),
                None
            )

            if v2_dep_at_trans is None:
                v2_dep_at_trans = next(
                    (etd for p, eta, etd in stops2 if p == discharge1),
                    None
                )

            if v2_dep_at_trans is not None and v2_dep_at_trans >= horizon_end:
                arr_hour  = v1_arr_at_trans
                discharge = discharge1
                truncated = True
                final_dest = discharge2
            elif v2_dep_at_trans is None:
                arr_hour  = v1_arr_at_trans
                discharge = discharge1
                truncated = True
                final_dest = discharge2
            else:
                v2_trans_dep_idx = next(
                    (i for i, (p, eta, etd) in enumerate(stops2)
                     if p == discharge1 and etd == v2_dep_at_trans),
                    None
                )
                arr_idx = next(
                    (i for i, (p, eta, etd) in enumerate(stops2)
                     if i > (v2_trans_dep_idx or 0) and p == discharge2),
                    None
                )
                if arr_idx is None:
                    return pd.Series([pd.NA, pd.NA, pd.NA, False, pd.NA])
                arr_hour  = stops2[arr_idx][1]
                discharge = discharge2
                truncated = False
                final_dest = discharge2
        else:
            discharge = row.get("Discharge 1")
            if pd.isna(discharge):
                return pd.Series([pd.NA, pd.NA, pd.NA, False, pd.NA])
            arr_idx = next(
                (i for i, (p, eta, etd) in enumerate(stops1) if i > dep_idx and p == discharge),
                None
            )
            if arr_idx is None:
                vc = str(voyage1)[:3].upper()
                full_stops = vessel_stops.get(vc, [])
                full_dep_idx = next(
                    (i for i, (p, eta, etd) in enumerate(full_stops)
                     if p == load1 and etd == dep_hour),
                    None
                )
                if full_dep_idx is not None:
                    full_arr_idx = next(
                        (i for i, (p, eta, etd) in enumerate(full_stops)
                         if i > full_dep_idx and p == discharge),
                        None
                    )
                    if full_arr_idx is not None:
                        arr_hour  = full_stops[full_arr_idx][1]
                        truncated = False
                        if arr_hour > dep_hour:
                            return pd.Series([int(dep_hour), int(arr_hour), discharge, truncated, discharge])
                return pd.Series([pd.NA, pd.NA, pd.NA, False, pd.NA])
            arr_hour  = stops1[arr_idx][1]
            truncated = False
            final_dest = discharge

        if arr_hour <= dep_hour:
            return pd.Series([pd.NA, pd.NA, pd.NA, False, pd.NA])

        return pd.Series([int(dep_hour), int(arr_hour), discharge, truncated, final_dest])

    df_full[["DepartureTimeNum", "ArrivalTimeNum", "ResolvedDest", "Truncated", "FinalDestination"]] = df_full.apply(
        assign_voyage_times, axis=1
    )

    dropped_mask_t = df_full["DepartureTimeNum"].isna() | df_full["ArrivalTimeNum"].isna()
    if dropped_mask_t.any():
        print(f"  Dropped {dropped_mask_t.sum()} rows with unresolvable voyage times")
        if cfg.get("verbose_pipeline", False):
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 240):
                print(df_full[dropped_mask_t].to_string())
    df_full = df_full.dropna(subset=["DepartureTimeNum", "ArrivalTimeNum"])

    truncated_count = df_full["Truncated"].sum()
    if truncated_count:
        print(f"  Truncated {truncated_count} rows to transshipment port")


    mask_ports = df_full["Load 1"].isin(port_dict) & df_full["ResolvedDest"].isin(port_dict)
    if (~mask_ports).any():
        print(f"  Dropped {(~mask_ports).sum()} rows with Load/Discharge port not in port_dict")
        if cfg.get("verbose_pipeline", False):
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 240):
                print(df_full[~mask_ports].to_string())
    df_full    = df_full[mask_ports].copy()


    dropped_mask_y = df_full["DepartureTimeNum"] < 0
    if dropped_mask_y.any():
        print(f"  Dropped {dropped_mask_y.sum()} rows departing before epoch start")
        if cfg.get("verbose_pipeline", False):
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 240):
                print(df_full[dropped_mask_y].to_string())
    df_full = df_full[~dropped_mask_y].copy()

    dropped_mask_h = df_full["DepartureTimeNum"] >= horizon_end
    if dropped_mask_h.any():
        print(f"  Dropped {dropped_mask_h.sum()} rows departing outside horizon")
        if cfg.get("verbose_pipeline", False):
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 240):
                print(df_full[dropped_mask_h].to_string())
    df_full = df_full[~dropped_mask_h].copy()


    final = pd.DataFrame({
        "Year":             df_full["Year"],
        "Week":             df_full["Week"],
        "Origin":           df_full["Load 1"],
        "DepartureTime":    df_full["DepartureTimeNum"].astype(int),
        "Destination":      df_full["ResolvedDest"],
        "ArrivalTime":      df_full["ArrivalTimeNum"].astype(int),
        "ContainerType":    df_full["ContainerType"],
        "ContainerSize":    df_full["ContainerSize"],
        "Owner":            df_full["Owner"],
        "ContractType":     df_full.get("ContractType", pd.Series([""] * len(df_full), index=df_full.index)),
        "Count":            df_full["Total"],
        "Truncated":        df_full["Truncated"].astype(bool),
        "FinalDestination": df_full["FinalDestination"],
    }).reset_index(drop=True)


    final["Commodity"] = [
        f"k{i}_{row.ContainerType}_{int(row.ContainerSize)}_{row.Owner}"
        for i, row in final.iterrows()
    ]

    print(f"  Commodities: {len(final)} rows ({len(CALENDAR_WEEKS)} weeks)")
    return final



if __name__ == "__main__":
    sys.path.insert(0, "pipeline")
    from config import CALENDAR_WEEKS, NUM_WEEKS, EPOCH_HOUR_OFFSET, H, HORIZON_MONDAY, COMMODITIES_PATH
    from datetime import date

    sys.path.insert(0, "data")
    from processed.eimskip_routes_generated import port_dict, vessel_lines

    voyages_df = pd.read_csv(
        "data/raw/Eimskip_voyages.csv",
        parse_dates=["etaDateTime", "etdDateTime"]
    )
    demand_df = pd.read_csv("data/clean/Eimskip_data_final.csv")

    # Compute epoch_dt
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

    commodities_df = build_commodities(demand_df, voyages_df, port_dict, vessel_lines, cfg)
    COMMODITIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    commodities_df.to_csv(COMMODITIES_PATH, index=False)
    print(f"Saved to {COMMODITIES_PATH}  ({NUM_WEEKS} weeks)")
