import pandas as pd

file_path = "data/clean/Eimskip_data_final.csv"
df = pd.read_csv(file_path, sep=",")

# -------------------------
# Basic Cleaning
# -------------------------

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Clean string columns (remove extra spaces)
for col in ["Load 1", "Discharge 1", "Load 2", "Discharge 2"]:
    df[col] = df[col].astype(str).str.strip()

# Remove fake discharge like "000"
df = df[~df["Discharge 2"].isin(["000", "", "nan"])]

# Convert numeric columns safely
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Week"] = pd.to_numeric(df["Week"], errors="coerce")
df["Total"] = pd.to_numeric(df["Total"], errors="coerce")

# -------------------------
# Unique Origin/Destination Codes
# -------------------------

origins = pd.concat([df["Load 1"], df["Load 2"]]).dropna().unique()
destinations = pd.concat([df["Discharge 1"], df["Discharge 2"]]).dropna().unique()

all_codes = sorted(set(origins).union(set(destinations)))

print("Number of unique origin codes:", len(origins))
print("Number of unique destination codes:", len(destinations))

print("\nAll Unique Origin/Destination Codes:")
print(all_codes)

# -------------------------
# Voyage Table (Voyage 1 + Voyage 2)
# -------------------------

voyage_cols = ["Voyage 1", "Voyage 2"]

voyage_table = (
    pd.concat([
        df[["Week", voyage_cols[0]]].rename(columns={voyage_cols[0]: "Voyage"}),
        df[["Week", voyage_cols[1]]].rename(columns={voyage_cols[1]: "Voyage"})
    ])
    .dropna()
    .drop_duplicates()
    .sort_values(["Week", "Voyage"])
)

print("\nVoyage Table:")
print(voyage_table)

# -------------------------
# Unique Sailings (Week + Voyage)
# -------------------------

unique_sailings = voyage_table.drop_duplicates()

sail_count = (
    unique_sailings
    .groupby("Voyage")
    .size()
    .reset_index(name="Number_of_Sailings")
)

print("\nSail Count:")
print(sail_count)

sail_count.to_csv("data/processed/historical-sail-count.csv", index=False)

# -------------------------
# Sailing Schedule
# -------------------------

sailing_schedule = (
    unique_sailings
    .groupby("Voyage")["Week"]
    .unique()
    .apply(sorted)
)

# for voyage, weeks in sailing_schedule.items():
#     print(f"{voyage} sails in weeks: {weeks}")

# -------------------------
# Build Route Legs (Leg 1 + Leg 2)
# -------------------------

legs_1 = df[["Load 1", "Discharge 1", "Voyage 1"]].rename(
    columns={
        "Load 1": "Origin",
        "Discharge 1": "Destination",
        "Voyage 1": "Voyage"
    }
)

legs_2 = df[["Load 2", "Discharge 2", "Voyage 2"]].rename(
    columns={
        "Load 2": "Origin",
        "Discharge 2": "Destination",
        "Voyage 2": "Voyage"
    }
)

df_routes = pd.concat([legs_1, legs_2]).dropna().drop_duplicates()

print("\n=== Unique Routes ===")
print(df_routes)

# -------------------------
# Demand per OD pair (FULL only)
# -------------------------

df_full = df[df["Full/Empty"] == "F"].copy()

# Build OD pairs for both legs
od_1 = df_full[[
    "Load 1", "Discharge 1",
    "ContainerType", "ContainerSize", "Owner", "Total"
]].rename(columns={
    "Load 1": "Origin",
    "Discharge 1": "Destination"
})

od_2 = df_full[[
    "Load 2", "Discharge 2",
    "ContainerType", "ContainerSize", "Owner", "Total"
]].rename(columns={
    "Load 2": "Origin",
    "Discharge 2": "Destination"
})

df_od = pd.concat([od_1, od_2]).dropna()

od_type_demand = (
    df_od
    .groupby(["Origin", "Destination", "ContainerType", "ContainerSize", "Owner"])["Total"]
    .sum()
    .reset_index()
    .sort_values(["Origin", "Destination", "ContainerType", "ContainerSize", "Owner"])
)

print("\n=== OD Demand Split by Container Type (Full Only) ===")
print(od_type_demand)

od_type_demand.to_csv("data/processed/historical-demand.csv", index=False)