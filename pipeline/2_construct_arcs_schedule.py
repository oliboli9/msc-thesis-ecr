"""
Build time-space arcs from the weekly sailing schedule
Run automatically by runner.py
"""

import pandas as pd
from config import NUM_WEEKS, H, EPOCH_HOUR_OFFSET, ARCS_PATH

INCLUDE_WRAP_ARC = False

TRANS_COST = 1.0
ICELAND_PORTS = {"IS REY", "IS GRT", "IS RFJ", "IS VES"}
TRANSSHIPMENT_PORTS = {"IS REY", "FO THO", "DK AAR", "NL RTM"}

DAY_INDEX = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

BASE_SCHEDULE = {
    "Red Line": [
        (1, "Mon", "DE BRV", "6:00",  "20:00"),
        (1, "Wed", "SE HEL", "7:00",  "23:00"),
        (1, "Thu", "DK AAR", "6:00",  "23:00"),
        (2, "Mon", "IS REY", "6:00",  "20:00"),
        (2, "Thu", "GL GOH", "18:00", None),
        (2, "Sat", "GL GOH", None,    "6:00"),
        (3, "Tue", "IS REY", "20:00", None),
        (3, "Wed", "IS REY", None,    "15:00"),
        (3, "Thu", "IS RFJ", "15:00", "21:00"),
        (3, "Fri", "FO THO", "16:00", "23:00"),
    ],
    "Blue Line": [
        (1, "Tue", "IS REY", None,    "6:00"),
        (1, "Tue", "IS GRT", "8:00",  None),
        (1, "Wed", "IS GRT", None,    "18:00"),
        (1, "Wed", "IS REY", "20:00", None),
        (1, "Thu", "IS REY", None,    "18:00"),
        (1, "Sun", "NL RTM", "19:00", None),
        (2, "Wed", "NL RTM", None,    "18:00"),
        (2, "Thu", "GB TEE", "14:00", "23:00"),
        (2, "Sun", "IS REY", "14:00", None),
    ],
    "Yellow Line": [
        (1, "Tue", "IS REY", "12:00", None),
        (1, "Wed", "IS REY", None,    "19:00"),
        (1, "Thu", "IS VES", "5:00",  "11:00"),
        (1, "Fri", "FO THO", "16:00", "22:00"),
        (1, "Sun", "GB IMM", "14:00", None),
        (2, "Mon", "GB IMM", None,    "1:00"),
        (2, "Wed", "NO FRK", "06:00", "18:00"),
        (2, "Wed", "PL SZZ", "10:00", "23:00"),
        (2, "Fri", "DK AAR", "6:00",  "15:00"),
        (2, "Sun", "FO THO", "18:00", "23:00"),
    ],
    "Green Line": [
        (1, "Tue", "US PWM", "8:00",  None),
        (1, "Wed", "US PWM", None,    "13:00"),
        (1, "Thu", "CA HAL", "13:00", "19:00"),
        (1, "Sat", "CA NWP", "6:00",  "15:00"),
        (2, "Fri", "IS REY", "8:00",  None),
        (3, "Tue", "IS REY", None,    "22:00"),
    ],
}

VESSEL_ROTATION = {
    "Red Line":    ["TUK", "BRU", "DET"],
    "Green Line":  ["PIC", "AVA", "BAK"],
    "Blue Line":   ["JOR", "HER"],
    "Yellow Line": ["SEL", "SKO"],
}

EXCLUSIONS = {
    ("Green Line",  "AVA", "CA NWP"),
    ("Yellow Line", "SEL", "PL SZZ"),
    ("Yellow Line", "SKO", "NO FRK"),
}

def to_hour(week: int, day: str, time_str: str) -> int:
    """Return LOCAL hour (0-based from horizon start)."""
    h, m = map(int, time_str.split(":"))
    return (week - 1) * 7 * 24 + DAY_INDEX[day] * 24 + h


def expand_schedule(base: dict, weeks: int) -> dict:
    expanded = {}
    base_weeks = {line: max(e[0] for e in events) for line, events in base.items()}
    for line, events in base.items():
        cycle_len = base_weeks[line]
        expanded_events = []
        for w in range(weeks):
            cycle, week_in_cycle = divmod(w, cycle_len)
            week_in_cycle += 1
            for week, day, port, arr, dep in events:
                if week == week_in_cycle:
                    new_week = week + cycle * cycle_len
                    if new_week <= weeks:
                        expanded_events.append((new_week, day, port, arr, dep))
        expanded[line] = expanded_events
    return expanded, base_weeks


def build_line_arcs(line, events, rotation, base_weeks, vessel_capacity):
    arcs = []
    port_times = {}

    # Extend by one extra cycle so sailings that depart within H but arrive
    # after H still have an arc.  Departures >= H are excluded below.
    H_EXT = H + base_weeks * 7 * 24
    extended_weeks = NUM_WEEKS + base_weeks

    for vessel_idx, vessel in enumerate(rotation):
        vessel_events = []
        offset = vessel_idx

        for abs_week in range(1, extended_weeks + 1):
            base_week = ((abs_week - 1 + offset) % base_weeks) + 1

            for week, day, port, arr_str, dep_str in BASE_SCHEDULE[line]:
                if week != base_week:
                    continue
                if (line, vessel, port) in EXCLUSIONS:
                    print(f"    Excluded: {line} / {vessel} / {port}")
                    continue

                arr_hour = to_hour(abs_week, day, arr_str) if arr_str else None
                dep_hour = to_hour(abs_week, day, dep_str) if dep_str else None

                if arr_hour is not None and (arr_hour < 0 or arr_hour >= H_EXT):
                    print(f"    Drop (out of range): {line} / {vessel} / {port} arr h{arr_hour}")
                    continue
                if dep_hour is not None and (dep_hour < 0 or dep_hour >= H_EXT):
                    print(f"    Drop (out of range): {line} / {vessel} / {port} dep h{dep_hour}")
                    continue

                vessel_events.append((port, arr_hour, dep_hour))

        vessel_events.sort(key=lambda e: e[1] if e[1] is not None else (e[2] or 0))

        # Sailing arcs — convert local hours to epoch hours for output.
        # Only include arcs whose departure is within the planning horizon (< H).
        # Arrivals may extend beyond H (ship finishes its voyage after week N).
        for (dep_port, _, dep_hour), (next_port, arr_hour, _) in zip(
            vessel_events, vessel_events[1:]
        ):
            if dep_hour is not None and arr_hour is not None and dep_hour < arr_hour and dep_hour < H:
                arcs.append(
                    ((dep_port, dep_hour + EPOCH_HOUR_OFFSET),
                     (next_port, arr_hour + EPOCH_HOUR_OFFSET),
                     vessel, vessel_capacity[vessel], 1.0)
                )

        if INCLUDE_WRAP_ARC and vessel_events:
            first_port, first_arr, _ = vessel_events[0]
            last_port, _, last_dep = vessel_events[-1]
            if last_dep is not None and first_arr is not None:
                arcs.append(
                    ((last_port, last_dep + EPOCH_HOUR_OFFSET),
                     (first_port, first_arr + EPOCH_HOUR_OFFSET),
                     vessel, vessel_capacity[vessel], 1.0)
                )

        # Only register times within the planning horizon for wait arcs
        for port, arr_hour, dep_hour in vessel_events:
            if arr_hour is not None and arr_hour < H:
                port_times.setdefault(port, set()).add(arr_hour)
            if dep_hour is not None and dep_hour < H:
                port_times.setdefault(port, set()).add(dep_hour)

    port_capacity = max(vessel_capacity[v] for v in rotation)

    for port, times in port_times.items():
        sorted_times = sorted(times)
        n = len(sorted_times)
        index_pairs = (
            [(i, j) for i in range(n) for j in range(n) if i != j]
            if INCLUDE_WRAP_ARC
            else [(i, j) for i in range(n) for j in range(i + 1, n)]
        )
        for i, j in index_pairs:
            t1, t2 = sorted_times[i], sorted_times[j]
            delta = t2 - t1
            if delta > 0:
                cost = 0.0 if port in ICELAND_PORTS else round(delta / 24, 2)
                arcs.append(((port, t1 + EPOCH_HOUR_OFFSET),
                             (port, t2 + EPOCH_HOUR_OFFSET),
                             "WAIT", port_capacity, cost))

    return arcs


def build_transshipment_arcs(time_space_arcs, vessel_rotation):
    arrivals, departures = [], []
    for line, arcs in time_space_arcs.items():
        for dep_node, arr_node, vessel, _, _ in arcs:
            if dep_node[0] != arr_node[0]:
                arrivals.append((arr_node[0], arr_node[1], line, vessel))
                departures.append((dep_node[0], dep_node[1], line, vessel))

    trans_arcs = {line: [] for line in time_space_arcs}
    for port in TRANSSHIPMENT_PORTS:
        port_arrivals   = [a for a in arrivals   if a[0] == port]
        port_departures = [d for d in departures if d[0] == port]
        for arr_port, arr_hour, arr_line, arr_vessel in port_arrivals:
            for dep_port, dep_hour, dep_line, dep_vessel in port_departures:
                if arr_line == dep_line:
                    continue
                if arr_vessel in vessel_rotation.get(dep_line, []):
                    continue
                if dep_hour > arr_hour:
                    trans_arcs[dep_line].append(
                        ((arr_port, arr_hour), (dep_port, dep_hour),
                         f"{arr_vessel}->{dep_vessel}", 10000, TRANS_COST)
                    )
                elif INCLUDE_WRAP_ARC:
                    delta = (dep_hour - arr_hour) % H
                    if delta > 0:
                        trans_arcs[dep_line].append(
                            ((arr_port, arr_hour), (dep_port, dep_hour),
                             f"{arr_vessel}->{dep_vessel}", 10000, TRANS_COST)
                        )
    return trans_arcs


# Main 

vessel_df = pd.read_csv("data/raw/vessel-capacities.csv", sep=r"\s+")
vessel_capacity = {row["Vessel"].strip(): int(row["TEUs"]) for _, row in vessel_df.iterrows()}

schedule, base_weeks = expand_schedule(BASE_SCHEDULE, NUM_WEEKS)

time_space_arcs = {
    line: build_line_arcs(line, events, VESSEL_ROTATION[line], base_weeks[line], vessel_capacity)
    for line, events in schedule.items()
}

for line, arcs in build_transshipment_arcs(time_space_arcs, VESSEL_ROTATION).items():
    time_space_arcs[line].extend(arcs)

rows = [
    {
        "Line": line, "Vessel": vessel,
        "DepPort": dep[0], "DepHour": dep[1],
        "ArrPort": arr[0], "ArrHour": arr[1],
        "Capacity": capacity, "Cost": cost,
    }
    for line, arcs in time_space_arcs.items()
    for dep, arr, vessel, capacity, cost in arcs
]

arcs_df = (
    pd.DataFrame(rows)
    .drop_duplicates()
    .sort_values(["Line", "Vessel", "DepHour"])
    .reset_index(drop=True)
)

ARCS_PATH.parent.mkdir(parents=True, exist_ok=True)
arcs_df.to_csv(ARCS_PATH, index=False)
print(f"Saved {len(arcs_df)} arcs to {ARCS_PATH} ({NUM_WEEKS} weeks, epoch offset {EPOCH_HOUR_OFFSET}h)")
