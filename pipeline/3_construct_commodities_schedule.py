"""
Build commodities from the weekly sailing schedule
Run automatically by runner.py


"""
import pandas as pd
from config import START_YEAR, START_WEEK, NUM_WEEKS, EPOCH_HOUR_OFFSET, CALENDAR_WEEKS, H, COMMODITIES_PATH

# Set to True to keep all raw container type codes; False for Dry / Reefer only.
SHOW_ALL_CONTAINER_TYPES = True

def classify_container_type(code: str) -> str:
    """3rd character == 'R' → Reefer, anything else → Dry."""
    code = str(code).strip()
    return "Reefer" if len(code) >= 3 and code[2].upper() == "R" else "Dry"

# 1. LOAD DATA
df = pd.read_csv("data/clean/Eimskip_data_final.csv")

# Filter to the configured horizon 
_mask = False
for _yr, _wk in CALENDAR_WEEKS:
    _mask = _mask | ((df["Year"] == _yr) & (df["Week"] == _wk))
df_filtered = df[_mask].copy()
print(f"  Loaded {len(df)} rows; {len(df_filtered)} in horizon {CALENDAR_WEEKS}")

df_full = df_filtered[
    (df_filtered["Full/Empty"] == "F") &
    (df_filtered["ContainerSize"].isin([20, 40, 45]))
].copy()

if not SHOW_ALL_CONTAINER_TYPES:
    df_full["ContainerType"] = df_full["ContainerType"].apply(classify_container_type)
dropped_filter = len(df_filtered) - len(df_full)
if dropped_filter:
    n_empty = (df_filtered["Full/Empty"] != "F").sum()
    n_size  = df_filtered[df_filtered["Full/Empty"] == "F"]["ContainerSize"].isin([20, 40, 45])
    n_size  = (~n_size).sum()
    print(f"  Dropped {dropped_filter} rows in initial filter  (empty containers: {n_empty}, invalid size: {n_size})")

# 2. SCHEDULE
# Tuples are: (week, day, port, ARR, DEP)
schedule = {
    "Red Line":[
        (1,"Mon","DE BRV","6:00","20:00"),
        (1,"Wed","SE HEL","7:00","23:00"),
        (1,"Thu","DK AAR","6:00","23:00"),
        (2,"Mon","IS REY","6:00","20:00"),
        (2,"Thu","GL GOH","18:00",None),
        (2,"Sat","GL GOH",None,"6:00"),
        (3,"Tue","IS REY","20:00",None),
        (3,"Wed","IS REY",None,"15:00"),
        (3,"Thu","IS RFJ","15:00","21:00"),
        (3,"Fri","FO THO","16:00","23:00"),
    ],
    "Blue Line":[
        (1,"Tue","IS REY",None,"6:00"),
        (1,"Tue","IS GRT","8:00",None),
        (1,"Wed","IS GRT",None,"18:00"),
        (1,"Wed","IS REY","20:00",None),
        (1,"Thu","IS REY",None,"18:00"),
        (1,"Sun","NL RTM","19:00",None),
        (2,"Wed","NL RTM",None,"18:00"),
        (2,"Thu","GB TEE","14:00","23:00"),
        (2,"Sun","IS REY","14:00",None),
    ],
    "Yellow Line":[
        (1,"Tue","IS REY","12:00",None),
        (1,"Wed","IS REY",None,"19:00"),
        (1,"Thu","IS VES","5:00","11:00"),
        (1,"Fri","FO THO","16:00","22:00"),
        (1,"Sun","GB IMM","14:00",None),
        (2,"Mon","GB IMM",None,"1:00"),
        (2,"Wed","NO FRK","06:00","18:00"),
        (2,"Wed","PL SZZ","10:00","23:00"),
        (2,"Fri","DK AAR","6:00","15:00"),
        (2,"Sun","FO THO","18:00","23:00"),
    ],
    "Green Line":[
        (1,"Tue","US PWM","8:00",None),
        (1,"Wed","US PWM",None,"13:00"),
        (1,"Thu","CA HAL","13:00","19:00"),
        (1,"Sat","CA NWP","6:00","15:00"),
        (2,"Fri","IS REY","8:00",None),
        (3,"Tue","IS REY",None,"22:00")
    ]
}

day_idx = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
idx_day = {v:k for k,v in day_idx.items()}

def to_hours(week, day, time):
    if pd.isna(time) or time is None:
        return None
    h, m = map(int, time.split(":"))
    return (week-1)*7*24 + day_idx[day]*24 + h + m/60

def hours_to_weekday_hour(hours):
    week = int(hours // (7*24)) + 1
    rem = hours % (7*24)
    day = idx_day[int(rem // 24)]
    hr = int(rem % 24)
    mn = int(round((rem % 1)*60))
    return f"W{week} {day} {hr:02d}:{mn:02d}"

# 3. BUILD A SCHEDULE DATAFRAME
# Unpack as (week, day, port, arr, dep) 
rows = []
for line, stops in schedule.items():
    for w, d, port, arr, dep in stops:
        rows.append({"line": line, "week": w, "day": d, "port": port, "dep": dep, "arr": arr})
sched_df = pd.DataFrame(rows)

sched_df["dep_h"] = sched_df.apply(lambda r: to_hours(r["week"], r["day"], r["dep"]), axis=1)
sched_df["arr_h"] = sched_df.apply(lambda r: to_hours(r["week"], r["day"], r["arr"]), axis=1)

# 4.  FIND EARLIEST DEPARTURE AND ARRIVAL
def assign_line_times(row, base_week=START_WEEK):
    line = row["Line 1"] if pd.notna(row["Line 1"]) else row["Line 2"]
    if pd.isna(line):
        return pd.Series([pd.NA, pd.NA])

    stops = schedule.get(line, [])


    stop_hours = []
    for w, d, p, arr_str, dep_str in stops:
        stop_hours.append({
            "port": p,
            "arr": to_hours(w, d, arr_str) if arr_str else None,
            "dep": to_hours(w, d, dep_str) if dep_str else None,
        })

    base_weeks = max(w for w, d, p, arr_str, dep_str in stops)
    cycle_hours = base_weeks * 7 * 24

    # Find departure at origin
    origin_idx = next(
        (i for i, s in enumerate(stop_hours)
         if s["port"] == row["Load 1"] and s["dep"] is not None),
        None
    )
    if origin_idx is None:
        print(f"    Drop: origin port '{row['Load 1']}' not found on {line}")
        return pd.Series([pd.NA, pd.NA])

    dep_time = stop_hours[origin_idx]["dep"]

    # Search for destination after origin, wrapping into next cycle if needed.
    # We extend the stop list by one extra cycle so destinations that appear
    # earlier in the schedule than the origin are found at origin+N stops.
    dest_ports = {row["Discharge 1"], row["Discharge 2"]} - {None, pd.NA, float('nan')}
    dest_ports = {p for p in dest_ports if isinstance(p, str)}

    arr_time = None

    # First pass: look for destination strictly after origin in same cycle
    for i in range(origin_idx + 1, len(stop_hours)):
        s = stop_hours[i]
        if s["port"] in dest_ports and s["arr"] is not None:
            arr_time = s["arr"]
            break

    # Second pass: destination is earlier in the schedule (next cycle wrap)
    if arr_time is None:
        for i in range(0, origin_idx + 1):
            s = stop_hours[i]
            if s["port"] in dest_ports and s["arr"] is not None:
                arr_time = s["arr"] + cycle_hours
                break

    if arr_time is None:
        return pd.Series([dep_time, pd.NA])

    # arrival must be after departure
    while arr_time < dep_time:
        arr_time += cycle_hours

    # Shift to actual week in data, then convert to global epoch hours
    week_shift = (row["Week"] - base_week) * 7 * 24
    dep_time += week_shift + EPOCH_HOUR_OFFSET
    arr_time += week_shift + EPOCH_HOUR_OFFSET

    return pd.Series([dep_time, arr_time])

df_full[["DepartureTimeNum","ArrivalTimeNum"]] = df_full.apply(assign_line_times, axis=1)

before = len(df_full)
df_full = df_full.dropna(subset=["DepartureTimeNum", "ArrivalTimeNum"])
dropped = before - len(df_full)
if dropped:
    print(f"  Dropped {dropped} rows with unresolvable schedule times")

# Drop commodities that depart outside the horizon
horizon_end = EPOCH_HOUR_OFFSET + H
before_h = len(df_full)
df_full = df_full[df_full["DepartureTimeNum"] < horizon_end].copy()
dropped_h = before_h - len(df_full)
if dropped_h:
    print(f"  Dropped {dropped_h} rows with departure outside horizon (>= h{horizon_end})")

# 5. FINAL CSV
final = pd.DataFrame({
    "Year":          df_full["Year"],
    "Week":          df_full["Week"],
    "Origin":        df_full["Load 1"],
    "DepartureTime": df_full["DepartureTimeNum"],
    "Destination":   df_full["Discharge 2"].combine_first(df_full["Discharge 1"]),
    "ArrivalTime":   df_full["ArrivalTimeNum"],
    "ContainerType": df_full["ContainerType"],
    "ContainerSize": df_full["ContainerSize"],
    "Owner":         df_full["Owner"],
    "Count":         df_full["Total"]
})

COMMODITIES_PATH.parent.mkdir(parents=True, exist_ok=True)
final.to_csv(COMMODITIES_PATH, index=False)
print(f"Finished: {COMMODITIES_PATH}  ({len(final)} rows, {NUM_WEEKS} weeks, epoch offset {EPOCH_HOUR_OFFSET}h)")