from datetime import date, timedelta
from pathlib import Path

EPOCH_YEAR = 2023
EPOCH_WEEK = 1      # ISO week 1 of 2023 → Monday 2 Jan 2023 = hour 0

START_YEAR = 2025
START_WEEK = 1      # ISO week number of the first week of the horizon
NUM_WEEKS  = 3      # length of the horizon in weeks

def _iso_monday(year: int, week: int) -> date:
    """Return the date of Monday in the given ISO year/week."""
    jan4 = date(year, 1, 4)          # Jan 4 is always in ISO week 1
    return jan4 - timedelta(days=jan4.weekday()) + timedelta(weeks=week - 1)

EPOCH_MONDAY   = _iso_monday(EPOCH_YEAR, EPOCH_WEEK)    # 2023-01-02
HORIZON_MONDAY = _iso_monday(START_YEAR, START_WEEK)

START_WEEK_OFFSET = (HORIZON_MONDAY - EPOCH_MONDAY).days // 7
EPOCH_HOUR_OFFSET = 0   # hours always start at 0 for the horizon, regardless of year

H = NUM_WEEKS * 7 * 24     # total horizon hours 

# List of (iso_year, iso_week) pairs covered by the horizon 
CALENDAR_WEEKS: list[tuple[int, int]] = []
for _i in range(NUM_WEEKS):
    _monday = HORIZON_MONDAY + timedelta(weeks=_i)
    _iso    = _monday.isocalendar()
    CALENDAR_WEEKS.append((_iso[0], _iso[1]))



def _horizon_stem() -> str:
    end_year, end_week = CALENDAR_WEEKS[-1]
    start = f"{START_YEAR}w{START_WEEK}"
    if NUM_WEEKS == 1:
        return start
    if end_year == START_YEAR:
        return f"{start}-w{end_week}"
    return f"{start}-{end_year}w{end_week}"

HORIZON_STEM     = _horizon_stem()
ARCS_PATH        = Path("data/processed/arcs")        / f"{HORIZON_STEM}.csv"
COMMODITIES_PATH = Path("data/processed/commodities") / f"{HORIZON_STEM}.csv"

if __name__ == "__main__":
    import sys
    import runpy

    voyage = "--voyage" in sys.argv

    if voyage:
        import subprocess
        print("Running voyage-based pipeline...")
        subprocess.run([sys.executable, "pipeline/1_clean_data_generate_routes.py"], check=True)
        runpy.run_path("pipeline/2_construct_arcs_voyages.py", run_name="__main__")
        runpy.run_path("pipeline/3_construct_commodities_voyages.py", run_name="__main__")
    else:
        print("Running schedule-based pipeline...")
        runpy.run_path("pipeline/2_construct_arcs_schedule.py", run_name="__main__")
        runpy.run_path("pipeline/3_construct_commodities_schedule.py", run_name="__main__")
