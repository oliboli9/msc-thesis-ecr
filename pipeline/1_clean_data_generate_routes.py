"""
Clean data and recreate historical voyages for construct_xx_voyages pipeline
Called by voyages pipeline, do not run manually
"""

import csv
import pandas as pd


EXCLUDED_PORTS = {"IS GRT", "IS REY", "IS RFJ", "IS VES"}
MAIN_LINES     = {"Red Line", "Green Line", "Yellow Line", "Blue Line"}

def is_old_is_port(port):
    return isinstance(port, str) and port.startswith("IS ") and port not in EXCLUDED_PORTS


input_file  = "data/raw/Eimskip_data_final_LeaseTypes.csv"
output_file = "data/clean/Eimskip_data_final.csv"

with open(input_file, newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()

    for row in reader:
        for col in ["Voyage 1", "Voyage 2", "Discharge 1", "Discharge 2", "Load 2"]:
            if row.get(col, "").strip() == "000":
                row[col] = ""

        if "Orange" in row.get("Line 1", "") or "Orange" in row.get("Line 2", ""):
            continue  # Skip rows with Orange Line

        for col in ("Line 1", "Line 2"):
            val = row.get(col, "")
            if val.startswith("Icelandic Coastal Service / "):
                row[col] = val[len("Icelandic Coastal Service / "):]

        line1 = row.get("Line 1", "")
        line2 = row.get("Line 2", "")
        ics1 = (line1 == "Icelandic Coastal Service")
        ics2 = (line2 == "Icelandic Coastal Service")

        if ics1 and ics2:
            continue  

        if ics1 and line2 in MAIN_LINES:
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
            row["Line 2"]      = ""
            row["Voyage 2"]    = ""
            row["Load 2"]      = ""
            row["Discharge 2"] = ""

        load_1      = row.get("Load 1")
        discharge_1 = row.get("Discharge 1")

        if load_1 == discharge_1:
            continue

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


_MAIN_LINES_LIST = ["Red Line", "Green Line", "Yellow Line", "Blue Line"]
LINE_LABEL = {l: l.replace(" Line", "") for l in _MAIN_LINES_LIST}

PORT_NAMES = {
    "BR OUT": "Outeiro",
    "CA HAL": "Halifax",
    "CA NWP": "Argentia",
    "CA SAT": "Saint Anthony",
    "DE BRV": "Bremerhaven",
    "DK AAR": "Aarhus",
    "DK HAN": "Hanstholm",
    "FO THO": "Thorshavn",
    "GB HUL": "Hull",
    "GB IMM": "Immingham",
    "GB LER": "Lerwick",
    "GB SCR": "Scrabster",
    "GB TEE": "Teesport",
    "GL GOH": "Nuuk",
    "IS AKU": "Akureyri",
    "IS GRT": "Grundartangi",
    "IS HUS": "Husavik",
    "IS ISA": "Isafjordur",
    "IS REY": "Reykjavik",
    "IS RFJ": "Reydarfjordur",
    "IS SAU": "Sauðárkrókur",
    "IS VES": "Vestmannaeyjar",
    "LT KLJ": "Klaipeda",
    "LV RIX": "Riga",
    "NL RTM": "Rotterdam",
    "NL VEL": "Velsen",
    "NL VLI": "Vlissingen",
    "NO AES": "Alesund",
    "NO BJF": "Batsfjord",
    "NO FRK": "Fredrikstad",
    "NO HIT": "Hitra",
    "NO SLX": "Sortland",
    "NO TOS": "Tromso",
    "PL GDN": "Gdansk",
    "PL GDY": "Gdynia",
    "PL SWI": "Swinoujscie",
    "PL SZZ": "Szczecin",
    "SE HEL": "Helsingborg",
    "US PWM": "Portland (Maine)",
}


voyages = pd.read_csv(
    "data/raw/Eimskip_voyages.csv",
    parse_dates=["etaDateTime", "etdDateTime"]
)

v = voyages[voyages["tradeRouteName"].isin(_MAIN_LINES_LIST)].copy()
v["vessel"] = v["voyage"].str[:3].str.upper()

# Build port_dict from voyages 

port_lines = (
    v.groupby("portID")["tradeRouteName"]
    .apply(lambda s: tuple(sorted({LINE_LABEL[l] for l in s})))
    .to_dict()
)

port_dict = {
    port: (PORT_NAMES.get(port, port),) + lines
    for port, lines in sorted(port_lines.items())
}

# Build vessel_lines from voyages 

vessel_lines = (
    v.groupby("vessel")["tradeRouteName"]
    .apply(lambda s: tuple(sorted({LINE_LABEL[l] for l in s})))
    .to_dict()
)

years = sorted(voyages["etdDateTime"].dt.year.dropna().unique().astype(int))
print(f"  Years: {years[0]}–{years[-1]},  "
      f"{len(vessel_lines)} vessels,  {len(port_dict)} ports")

out_path = "data/processed/eimskip_routes_generated.py"

lines_out = [
    f'"""',
    f'data/processed/eimskip_routes_generated.py',
    f'Auto-generated by pipeline/1_clean_data_generate_routes.py (all years in Eimskip_voyages.csv).',
    f'Do not edit manually — re-run the script to regenerate.',
    f'"""',
    f'',
    f'port_dict = {{',
]
for port, entry in sorted(port_dict.items()):
    lines_out.append(f'    {port!r}: {entry!r},')
lines_out += [
    f'}}',
    f'',
    f'vessel_lines = {{',
]
for v_code, line in sorted(vessel_lines.items()):
    lines_out.append(f'    {v_code!r}: {line!r},')
lines_out += [
    f'}}',
    f'',
]

with open(out_path, "w") as f:
    f.write("\n".join(lines_out))

print("Step 2 complete: eimskip_routes_generated.py written.")
