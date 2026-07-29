"""
Manual data cleaning step for raw demand and voyage data. (run once to create clean data)
"""

import csv
import pandas as pd

MAIN_LINES = {"Red Line", "Green Line", "Yellow Line", "Blue Line"}
EXCLUDED_PORTS = {"IS GRT", "IS REY", "IS RFJ", "IS VES"}


def is_old_is_port(port):
    return isinstance(port, str) and port.startswith("IS ") and port not in EXCLUDED_PORTS


def clean_demand_data():
    """Clean raw demand CSV → data/clean/Eimskip_data_final.csv.

    Applies:
    - Normalise " 000" voyage/discharge codes to empty strings
    - Remove Orange Line cargo
    - Normalise hybrid ICS/main-line names
    - Handle Icelandic Coastal Service (ICS) legs
    - Normalise old Iceland port codes to IS REY
    - Drop same-origin-destination rows
    """
    input_file  = "data/raw/Eimskip_data_final_LeaseTypes.csv"
    output_file = "data/clean/Eimskip_data_final.csv"

    RENAME_COLS = {
        "Container type": "ContainerType",
        "Container size": "ContainerSize",
        "Contract type":  "ContractType",
    }

    with open(input_file, newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        out_fields = [RENAME_COLS.get(f, f) for f in reader.fieldnames]
        writer = csv.DictWriter(outfile, fieldnames=out_fields)
        writer.writeheader()

        for row in reader:

            row = {RENAME_COLS.get(k, k): v for k, v in row.items()}
            # Drop rows lacking owner / container type / size 
            owner = (row.get("Owner") or "").strip().upper()
            ctype = (row.get("ContainerType") or "").strip()
            csize = (row.get("ContainerSize") or "").strip()
            if owner in ("", "NAN", "NA") or ctype in ("", "NAN", "NA") \
                    or csize in ("", "NAN", "NA"):
                continue

            # LONG and CABOTAGE leases become CARRIER as Eimskip controls these long enough that their empties are repositioned the same way as carrier-owned containers
            contract = (row.get("ContractType") or "").strip().upper()
            if owner == "LEASED" and contract in {"LONG", "CABOTAGE"}:
                row["Owner"] = "CARRIER"
                owner = "CARRIER"

            for col in ["Voyage 1", "Voyage 2", "Discharge 1", "Discharge 2", "Load 2"]:
                if row.get(col, "").strip() == "000":
                    row[col] = ""

            if "Orange" in row.get("Line 1", "") or "Orange" in row.get("Line 2", ""):
                continue  # skip Orange Line rows

            for col in ("Line 1", "Line 2"):
                val = row.get(col, "")
                if val.startswith("Icelandic Coastal Service / "):
                    row[col] = val[len("Icelandic Coastal Service / "):]

            line1 = row.get("Line 1", "")
            line2 = row.get("Line 2", "")
            ics1 = (line1 == "Icelandic Coastal Service")
            ics2 = (line2 == "Icelandic Coastal Service")

            if ics1 and ics2:
                continue  # both legs coastal 

            if ics1 and line2 in MAIN_LINES:
                # shift leg 2 into leg 1
                row["Line 1"]      = row["Line 2"]
                row["Voyage 1"]    = row["Voyage 2"]
                row["Load 1"]      = row["Load 2"]
                row["Discharge 1"] = row["Discharge 2"]
                row["Line 2"]      = ""
                row["Voyage 2"]    = ""
                row["Load 2"]      = ""
                row["Discharge 2"] = ""
            elif ics1:
                continue  

            if ics2:
                # truncate to Discharge 1, clear leg 2
                row["Line 2"]      = ""
                row["Voyage 2"]    = ""
                row["Load 2"]      = ""
                row["Discharge 2"] = ""

            load_1      = row.get("Load 1")
            discharge_1 = row.get("Discharge 1")

            if load_1 == discharge_1:
                continue 

            #  Normalise old IS ports — Leg 1 
            old_l1 = is_old_is_port(load_1)
            old_d1 = is_old_is_port(discharge_1)

            if old_l1 and old_d1:
                continue                          
            if old_l1 and not discharge_1.startswith("IS "):
                row["Load 1"] = "IS REY"
                load_1 = "IS REY"
            if old_d1 and not load_1.startswith("IS "):
                row["Discharge 1"] = "IS REY"
                discharge_1 = "IS REY"

            # Normalise old IS ports — Leg 2 
            if row.get("Voyage 2") not in (None, "", "000", " 000"):
                load_2      = row.get("Load 2", "")
                discharge_2 = row.get("Discharge 2", "")

                if not load_2 or not discharge_2:
                    print(row)
                    continue

                old_l2 = is_old_is_port(load_2)
                old_d2 = is_old_is_port(discharge_2)

                if old_l2 and old_d2:
                    continue
                if old_l2 and not discharge_2.startswith("IS "):
                    row["Load 2"] = "IS REY"
                if old_d2 and not load_2.startswith("IS "):
                    row["Discharge 2"] = "IS REY"

            port_cols = [row.get("Load 1"), row.get("Discharge 1"),
                         row.get("Load 2"), row.get("Discharge 2")]
            if any(is_old_is_port(p) for p in port_cols):
                continue

            writer.writerow(row)

    print("Step 1 complete: fixedEimskip_data_final.csv written.")


def filter_voyages():
    """Filter Eimskip_voyages.csv to main lines only → data/clean/eimskip_voyages.csv."""
    voyages_df = pd.read_csv(
        "data/raw/Eimskip_voyages.csv",
        parse_dates=["etaDateTime", "etdDateTime"]
    )
    filtered = voyages_df[voyages_df["tradeRouteName"].isin(MAIN_LINES)].copy()
    filtered.to_csv("data/clean/eimskip_voyages.csv", index=False)
    print(f"Step 2 complete: eimskip_voyages.csv written "
          f"({len(filtered):,} rows, {filtered['tradeRouteName'].nunique()} lines, "
          f"{filtered['portID'].nunique()} ports).")


if __name__ == "__main__":
    clean_demand_data()
    filter_voyages()
