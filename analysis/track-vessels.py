"""
Per-vessel timeline tracker.

Reads results/vessel-utilisation/{label}.csv and results/model-flows/{label}.csv
for the rolling-horizon iterations and prints, for each vessel:
  - starting state (location + first-departure time)
  - each port call: gross containers loaded and unloaded (TEU)
  - per-leg utilisation (capacity, cargo TEU, empty TEU, fill %)

Usage:
    python analysis/track-vessels.py
    python analysis/track-vessels.py --start-year 2025 --start-week 1 \
                                     --iterations 3 --decision-weeks 1

Defaults match the runner's typical configuration: 2025 W01, 3 iterations,
1-week decision horizon.
"""

from __future__ import annotations
import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def teu_of(commodity: str) -> int:
    for part in str(commodity).split("_"):
        try:
            sz = float(part)
            if sz == 20.0:
                return 1
            if sz in (40.0, 45.0):
                return 2
        except ValueError:
            continue
    return 2


def iso_monday(year: int, week: int) -> date:
    jan4 = date(year, 1, 4)
    return jan4 - timedelta(days=jan4.weekday()) + timedelta(weeks=week - 1)


def build_labels(start_year: int, start_week: int, iterations: int,
                 decision_weeks: int) -> list[str]:
    labels = []
    for i in range(iterations):
        monday = iso_monday(start_year, start_week) + timedelta(weeks=i * decision_weeks)
        iso = monday.isocalendar()
        labels.append(f"{iso[0]}w{iso[1]}")
    return labels


def load_vessel_capacities() -> dict:
    """Map Vessel code → nominal TEU capacity from data/raw/vessel-capacities.csv."""
    cap = pd.read_csv(REPO / "data/raw/vessel-capacities.csv", sep=r"\s+")
    return dict(zip(cap["Vessel"], cap["TEUs"].astype(float)))


def load_legs(labels: list[str], decision_hours: int):
    """Stitch decision-window legs across iterations into one absolute timeline.

    Restores nominal vessel capacity (from vessel-capacities.csv) and exposes
    the in-transit reservation as PreloadedTEU = nominal - effective.
    """
    nominal_cap = load_vessel_capacities()

    util_pieces, flow_pieces = [], []
    for idx, label in enumerate(labels):
        offset = idx * decision_hours
        util_path = REPO / f"results/vessel-utilisation/{label}.csv"
        flow_path = REPO / f"results/model-flows/{label}.csv"
        if not util_path.exists() or not flow_path.exists():
            raise FileNotFoundError(
                f"Missing results for label {label!r}. "
                f"Run framework/runner.py with matching parameters first."
            )
        util = pd.read_csv(util_path)
        util = util[util["DepHour"] < decision_hours].copy()
        util["Iter"] = idx + 1
        util["AbsDepHour"] = util["DepHour"] + offset
        util["AbsArrHour"] = util["ArrHour"] + offset
        util["EffectiveCapacity"] = util["Capacity"]
        util["NominalCapacity"]   = util["Vessel"].map(nominal_cap).fillna(util["Capacity"])
        util["PreloadedTEU"]      = (util["NominalCapacity"] - util["EffectiveCapacity"]).clip(lower=0)
        util_pieces.append(util)

        flows = pd.read_csv(flow_path)
        sail = flows[
            (flows["DepPort"] != flows["ArrPort"]) &
            (flows["DepHour"] < decision_hours)
        ].copy()
        sail["Iter"] = idx + 1
        flow_pieces.append(sail)

    legs = pd.concat(util_pieces, ignore_index=True)
    flows = pd.concat(flow_pieces, ignore_index=True)
    flows["TEU"] = flows["Flow"] * flows["Commodity"].apply(teu_of)
    return legs, flows


def commodities_on_arc(flows: pd.DataFrame, leg) -> dict:
    """Return {commodity: TEU} for the model-flows row matching this leg."""
    m = flows[
        (flows["Iter"] == leg["Iter"]) &
        (flows["Vessel"] == leg["Vessel"]) &
        (flows["DepPort"] == leg["DepPort"]) &
        (flows["ArrPort"] == leg["ArrPort"]) &
        (flows["DepHour"] == leg["DepHour"]) &
        (flows["ArrHour"] == leg["ArrHour"])
    ]
    return dict(zip(m["Commodity"], m["TEU"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-year",     type=int, default=2025)
    ap.add_argument("--start-week",     type=int, default=1)
    ap.add_argument("--iterations",     type=int, default=3)
    ap.add_argument("--decision-weeks", type=int, default=1)
    args = ap.parse_args()

    decision_hours = args.decision_weeks * 168
    labels = build_labels(args.start_year, args.start_week,
                          args.iterations, args.decision_weeks)
    print(f"# Tracking vessels across iterations: {labels}")
    print(f"# Decision horizon = {args.decision_weeks}w ({decision_hours}h)")

    legs, flows = load_legs(labels, decision_hours)
    legs = legs.sort_values(["Vessel", "AbsDepHour"]).reset_index(drop=True)

    for vessel, vlegs in legs.groupby("Vessel"):
        vlegs = vlegs.sort_values("AbsDepHour").reset_index(drop=True)
        line = vlegs["Line"].iloc[0]
        print("\n" + "=" * 78)
        print(f"VESSEL {vessel}  ({line})")
        print("=" * 78)

        first = vlegs.iloc[0]
        nominal = float(first["NominalCapacity"])
        preload_first = float(first["PreloadedTEU"])
        if preload_first > 0:
            start_note = (f"onboard at hour 0 = {preload_first:.0f} TEU "
                          f"(in-transit cargo from prior horizon)")
        else:
            start_note = ("onboard at hour 0 = 0 TEU "
                          "(model loads cargo at first port from port stock)")
        print(f"Start: at {first['DepPort']} (first departure {first['DepDate']}, "
              f"hour {int(first['AbsDepHour'])}); nominal capacity {nominal:.0f} TEU; "
              f"{start_note}")

        prev_comms: dict = {}
        prev_arr_port = None
        prev_arr_hour = None

        for i, leg in vlegs.iterrows():
            cur_comms = commodities_on_arc(flows, leg)
            cur_total = sum(cur_comms.values())
            port = leg["DepPort"]

            if i == 0:
                loaded, unloaded = cur_total, 0
                print(f"\n  Port call @ {port}  "
                      f"(depart hour {int(leg['AbsDepHour'])}, {leg['DepDate']})")
                print(f"    Loaded:   {loaded:>5.0f} TEU")
                print(f"    Unloaded: {unloaded:>5.0f} TEU")
            else:
                if prev_arr_port != port:
                    # Iteration boundary or non-contiguous voyage segment.
                    if prev_comms:
                        prev_total = sum(prev_comms.values())
                        print(f"\n  Port call @ {prev_arr_port}  "
                              f"(arrive hour {int(prev_arr_hour)})  "
                              f"[iteration boundary — vessel rests/repositions]")
                        print(f"    Loaded:   {0:>5.0f} TEU")
                        print(f"    Unloaded: {prev_total:>5.0f} TEU")
                    print(f"\n  Port call @ {port}  "
                          f"(depart hour {int(leg['AbsDepHour'])}, {leg['DepDate']})  "
                          f"[iteration boundary]")
                    print(f"    Loaded:   {cur_total:>5.0f} TEU")
                    print(f"    Unloaded: {0:>5.0f} TEU")
                else:
                    keys = set(prev_comms) | set(cur_comms)
                    loaded, unloaded = 0.0, 0.0
                    for k in keys:
                        a = prev_comms.get(k, 0.0)  # TEU arriving on V
                        b = cur_comms.get(k, 0.0)   # TEU departing on V
                        if a > b:
                            unloaded += a - b
                        elif b > a:
                            loaded += b - a
                    print(f"\n  Port call @ {port}  "
                          f"(arrive hour {int(prev_arr_hour)}, "
                          f"depart hour {int(leg['AbsDepHour'])}, {leg['DepDate']})")
                    print(f"    Loaded:   {loaded:>5.0f} TEU")
                    print(f"    Unloaded: {unloaded:>5.0f} TEU")

            nom  = float(leg["NominalCapacity"])
            pre  = float(leg["PreloadedTEU"])
            cargo = float(leg["CargoTEU"])
            emp   = float(leg["EmptyTEU"])
            fill  = (cargo + emp + pre) / nom * 100 if nom else 0.0
            preload_str = f"  preloaded {pre:.0f}" if pre > 0 else ""
            print(f"  Leg → {leg['ArrPort']}  "
                  f"(arr hour {int(leg['AbsArrHour'])}):  "
                  f"cargo {cargo:.0f}  empty {emp:.0f}{preload_str}  "
                  f"capacity {nom:.0f}  fill {fill:.1f}%")

            prev_comms = cur_comms
            prev_arr_port = leg["ArrPort"]
            prev_arr_hour = leg["AbsArrHour"]

        if prev_comms:
            final = sum(prev_comms.values())
            print(f"\n  Final port call @ {prev_arr_port}  "
                  f"(arrive hour {int(prev_arr_hour)})")
            print(f"    Unloaded: {final:>5.0f} TEU  (end of decision window)")

        nom = vlegs["NominalCapacity"].astype(float)
        true_fill = ((vlegs["CargoTEU"] + vlegs["EmptyTEU"] + vlegs["PreloadedTEU"])
                     / nom * 100).mean()
        cargo_avg = (vlegs["CargoTEU"] / nom * 100).mean()
        print(f"\n  Summary: {len(vlegs)} legs, "
              f"avg total fill {true_fill:.1f}% (cargo {cargo_avg:.1f}%)")


if __name__ == "__main__":
    main()
