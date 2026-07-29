from collections import defaultdict
import csv
import sys
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, "pipeline")
from config import EPOCH_HOUR_OFFSET, H, CALENDAR_WEEKS, HORIZON_MONDAY, START_YEAR, START_WEEK, ARCS_PATH, COMMODITIES_PATH

HORIZON_END = EPOCH_HOUR_OFFSET + H   # absolute epoch hour at end of planning horizon

_v = pd.read_csv("data/raw/Eimskip_voyages.csv", parse_dates=["etdDateTime"])
_mask = False
for _yr, _wk in CALENDAR_WEEKS:
    _mask = _mask | ((_v["year"] == _yr) & (_v["week"] == _wk))
_etds = _v[_mask]["etdDateTime"]
EPOCH_DT = (_etds.min().normalize().to_pydatetime() if not _etds.empty
            else datetime(HORIZON_MONDAY.year, HORIZON_MONDAY.month, HORIZON_MONDAY.day))

def to_dt(h): return (EPOCH_DT + timedelta(hours=int(h))).strftime("%d %b %Y %H:%M")

# ----------------------------
# Load arc data
# ----------------------------

arcs_df = pd.read_csv(ARCS_PATH)

# Create arc identifiers
arcs_df["arc_id"] = arcs_df.index

A = arcs_df["arc_id"].tolist()

capacity = arcs_df.set_index("arc_id")["Capacity"].to_dict()
cost = arcs_df.set_index("arc_id")["Cost"].to_dict()

# Time-space nodes
arcs_df["dep_node"] = list(zip(arcs_df["DepPort"], arcs_df["DepHour"]))
arcs_df["arr_node"] = list(zip(arcs_df["ArrPort"], arcs_df["ArrHour"]))

nodes = set(arcs_df["dep_node"]).union(set(arcs_df["arr_node"]))
N = list(nodes)

# Incidence lists
out_arcs = {i: [] for i in N}
in_arcs = {i: [] for i in N}

for _, row in arcs_df.iterrows():
    a = row["arc_id"]
    i = row["dep_node"]
    j = row["arr_node"]

    out_arcs[i].append(a)
    in_arcs[j].append(a)

# ----------------------------
# Example demand data
# ----------------------------

# Commodity structure
# (origin_node, destination_node, demand)

# commodities = {
#     "k1": (("IS REY", 198), ("IS GRT", 200), 100),
#     "k2": (("IS REY", 214), ("US PWM", 368), 150)
# }

# K = list(commodities.keys())

# origin = {k: commodities[k][0] for k in K}
# dest = {k: commodities[k][1] for k in K}
# demand = {k: commodities[k][2] for k in K}

commodities_df = pd.read_csv(COMMODITIES_PATH)

# Ensure time values are integers
commodities_df["DepartureTime"] = commodities_df["DepartureTime"].astype(int)
commodities_df["ArrivalTime"] = commodities_df["ArrivalTime"].astype(int)

# Normalise owner to title case so it matches empty_owners ['Carrier','Leased','Shipper']
commodities_df["Owner"] = commodities_df["Owner"].str.title()

# Truncated flag — commodity was truncated to transshipment port because Voyage 2
# departs outside the horizon
if "Truncated" not in commodities_df.columns:
    commodities_df["Truncated"] = False
commodities_df["Truncated"] = commodities_df["Truncated"].fillna(False).astype(bool)

# Create unique commodity IDs
commodities_df["Commodity"] = [
    f"k{i}_{row.ContainerType}_{row.ContainerSize}_{row.Owner}"
    for i, row in commodities_df.iterrows()
]

teu = {}
commodity_spec = {}  # k -> (container_type, container_size)
owner_of = {}        # k -> owner ('Carrier', 'Leased', 'Shipper')
truncated_of = {}    # k -> bool: truncated to transshipment port

for _, row in commodities_df.iterrows():
    k = row["Commodity"]
    size = row["ContainerSize"]
    ctype = row["ContainerType"]

    if size == 20:
        teu[k] = 1
    elif size in [40, 45]:
        teu[k] = 2
    else:
        raise ValueError(f"Unknown container size: {size}")

    commodity_spec[k] = (ctype, size)
    owner_of[k] = row["Owner"]
    truncated_of[k] = bool(row["Truncated"])

# Container type/size combinations for empty container tracking
container_types = commodities_df["ContainerType"].unique().tolist()
container_sizes = commodities_df["ContainerSize"].unique().tolist()
container_specs = [(ct, cs) for ct in container_types for cs in container_sizes]
teu_spec = {(ct, cs): 1 if cs == 20 else 2 for ct, cs in container_specs}

# Carrier containers need full empty tracking (repositioning across the network).
# Leased containers are provided on-demand by the leasing company at the origin port —
# supply is injected directly at each origin node, no repositioning needed.
# Shipper containers are customer-owned; no empty tracking.
empty_owners = ['Carrier', 'Leased']

# Commodity set
K = commodities_df["Commodity"].tolist()

# Origin nodes
origin = {
    row["Commodity"]: (row["Origin"], row["DepartureTime"])
    for _, row in commodities_df.iterrows()
}

# Destination nodes
dest = {
    row["Commodity"]: (row["Destination"], row["ArrivalTime"])
    for _, row in commodities_df.iterrows()
}

# Demand
demand = {
    row["Commodity"]: int(row["Count"])
    for _, row in commodities_df.iterrows()
}

# ----------------------------
# Initial empty container stock
# ----------------------------


earliest_time_per_port: dict[str, int] = {}
for port, h in nodes:
    if port not in earliest_time_per_port or h < earliest_time_per_port[port]:
        earliest_time_per_port[port] = h

# Uniform empty container supply 
UNIFORM_SUPPLY = 100
IS_REY_SUPPLY  = 100

supply = {}

INITIAL_SUPPLY_PATH = Path("data/processed/initial_supply.csv")

if INITIAL_SUPPLY_PATH.exists():
    # Rolling horizon mode: use carried-forward inventory from previous week.
    print("  Using carried-forward inventory from initial_supply.csv")
    init_df = pd.read_csv(INITIAL_SUPPLY_PATH)
    for _, row in init_df.iterrows():
        port = row["Port"]
        if port not in earliest_time_per_port:
            continue
        node = (port, earliest_time_per_port[port])
        ct = row["ContainerType"]
        cs = float(row["ContainerSize"])
        eo = row["Owner"]
        if ct in container_types and cs in container_sizes and eo in empty_owners:
            key = (node, ct, cs, eo)
            supply[key] = supply.get(key, 0) + int(row["Count"])
elif START_YEAR == 2023 and START_WEEK == 1:
    sailing_arcs = arcs_df[arcs_df["DepPort"] != arcs_df["ArrPort"]]
    vessel_arrival_hour: dict[tuple, int] = {}
    for _, _arc in sailing_arcs.sort_values("ArrHour").iterrows():
        _key = (_arc["DepPort"], _arc["ArrPort"])
        if _key not in vessel_arrival_hour:
            vessel_arrival_hour[_key] = int(_arc["ArrHour"])


    supply_at_port: dict[tuple, int] = defaultdict(int)  # (port, ct, cs) count
    supply_at_node: dict[tuple, int] = defaultdict(int)  # (node, ct, cs) count

    with open("data/raw/Stock_01012023.csv", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Only Carrier-owned containers belong to Eimskip's repositionable pool.
            # ONEWAY / SHIPPER containers are not Eimskip's to reposition.
            if row["Owner"].strip().upper() != "CARRIER":
                continue

            location = row["Location"].strip()
            last_location = row["Last location"].strip()
            next_location = row["Next location"].strip()

            raw_container_type = row["ContainerType"].strip().upper()
            try:
                container_size = int(float(row["ContainerSize"]))
            except ValueError:
                continue
            count = int(row["count"])

            # Reefer if 3rd character is R, otherwise Dry
            if len(raw_container_type) >= 3 and raw_container_type[2] == "R":
                container_category = "Reefer"
            else:
                container_category = "Dry"

            if location.upper() == "VESSEL":
                if not next_location:
                    continue
                arr_h = vessel_arrival_hour.get((last_location, next_location))
                if arr_h is not None:
                    node = (next_location, arr_h)
                    supply_at_node[(node, container_category, container_size)] += count
                else:
                    if next_location in earliest_time_per_port:
                        print(f"  Warning: no arc {last_location}→{next_location}; "
                              f"injecting VESSEL stock at earliest node.")
                        supply_at_port[(next_location, container_category, container_size)] += count
                    else:
                        print(f"  Warning: VESSEL row skipped — {next_location} not in network.")
            else:
                port = location
                if not port or port not in earliest_time_per_port:
                    continue
                supply_at_port[(port, container_category, container_size)] += count

    # Build supply dictionary
    supply = {}

    for port, earliest_h in earliest_time_per_port.items():
        node = (port, earliest_h)
        for ct in container_types:
            for cs in container_sizes:
                amount = supply_at_port.get((port, ct, cs), 0)
                if amount:
                    supply[(node, ct, cs, "Carrier")] = amount

    for (node, ct, cs), amount in supply_at_node.items():
        if node in nodes and ct in container_types and cs in container_sizes:
            key = (node, ct, cs, "Carrier")
            supply[key] = supply.get(key, 0) + amount
        else:
            print(f"  Warning: VESSEL arrival node {node} not in network — stock skipped.")
else:

    for port, earliest_h in earliest_time_per_port.items():
        node = (port, earliest_h)
        amount = IS_REY_SUPPLY if port == "IS REY" else UNIFORM_SUPPLY
        for ct in container_types:
            for cs in container_sizes:
                key = (node, ct, cs, "Carrier")
                supply[key] = amount
                
print(supply)


for k in K:
    if owner_of[k] != 'Leased':
        continue
    node = origin[k]
    ct, cs = commodity_spec[k]
    key = (node, ct, cs, 'Leased')
    supply[key] = supply.get(key, 0) + demand[k]



CARRYOVER_PATH = Path("data/processed/carryover_demands.csv")

if CARRYOVER_PATH.exists():
    carry_df = pd.read_csv(CARRYOVER_PATH)

    # Port → sorted list of node times (for picking earliest/latest node)
    port_node_times: dict[str, list[int]] = {}
    for p, h in nodes:
        port_node_times.setdefault(p, []).append(h)

    n_carry_added = 0
    for c_idx, row in carry_df.iterrows():
        arr_port  = row["ArrPort"]
        orig_dest = row["OrigDest"]
        ct  = row["ContainerType"]
        cs  = float(row["ContainerSize"])
        eo  = row["Owner"]
        cnt = int(row["Count"])
        orig_id = str(row["OrigCommodityID"])

        if arr_port not in port_node_times or orig_dest not in port_node_times:
            continue  

        dep_h = min(port_node_times[arr_port])   # earliest available node at arrival port
        arr_h = max(port_node_times[orig_dest])  # latest node at destination (full week available)

        k_id = f"CARRY_{c_idx}_{orig_id[:20]}"  # unique carry-over ID

        K.append(k_id)
        origin[k_id]         = (arr_port, dep_h)
        dest[k_id]           = (orig_dest, arr_h)
        demand[k_id]         = cnt
        teu[k_id]            = 1 if cs == 20 else 2
        commodity_spec[k_id] = (ct, cs)
        owner_of[k_id]       = eo
        truncated_of[k_id]   = False


        if eo in empty_owners and ct in container_types and cs in container_sizes:
            key = ((arr_port, dep_h), ct, cs, eo)
            supply[key] = supply.get(key, 0) + cnt

        n_carry_added += 1

    if n_carry_added:
        print(f"  Added {n_carry_added} carry-over commodity groups "
              f"({sum(demand[k] for k in K if k.startswith('CARRY_'))} containers)")

# ----------------------------
# Build model
# ----------------------------

m = gp.Model("container_repositioning")

# ----------------------------
# Variables
# ----------------------------

# Cargo flow
x = m.addVars(A, K, lb=0, vtype=GRB.INTEGER, name="x")

# Empty container flow — Carrier and Leased only (Shipper has no empty repositioning)
e = m.addVars(A, container_types, container_sizes, empty_owners, lb=0, vtype=GRB.INTEGER, name="e")

# Fulfilled demand
f = m.addVars(K, lb=0, vtype=GRB.INTEGER, name="f")

I = m.addVars(N, container_types, container_sizes, empty_owners, lb=0, vtype=GRB.INTEGER, name="inventory")

# ----------------------------
# Objective
# ----------------------------

# m.setObjective(
#     gp.quicksum(f[k] for k in K),
#     GRB.MAXIMIZE
# )

# Weight factor for cost term
lambda_cost = 0.01  # tune this to scale cost relative to fulfilled demand

m.setObjective(
    gp.quicksum(f[k] for k in K)
    - lambda_cost * gp.quicksum(
        (gp.quicksum(x[a,k] for k in K)
         + gp.quicksum(e[a, ct, cs, eo] for ct, cs in container_specs for eo in empty_owners)
        ) * cost[a]
        for a in A
    ),
    GRB.MAXIMIZE
)

# ----------------------------
# Capacity constraints
# ----------------------------

for a in A:
    m.addConstr(
        gp.quicksum(teu[k] * x[a, k] for k in K)
        + gp.quicksum(
            teu_spec[ct, cs] * e[a, ct, cs, eo]
            for ct, cs in container_specs for eo in empty_owners
        )
        <= capacity[a],
        name=f"cap_{a}"
    )

# ----------------------------
# Cargo flow conservation
# ----------------------------

for k in K:
    for i in N:

        lhs = gp.quicksum(x[a, k] for a in out_arcs[i]) - gp.quicksum(x[a, k] for a in in_arcs[i])

        if i == origin[k]:
            m.addConstr(lhs == f[k], name=f"flow_origin_{k}_{i}")

        elif i == dest[k]:
            m.addConstr(lhs == -f[k], name=f"flow_dest_{k}_{i}")

        else:
            m.addConstr(lhs == 0, name=f"flow_trans_{k}_{i}")

# ----------------------------
# Demand limits
# ----------------------------

for k in K:
    m.addConstr(f[k] <= demand[k], name=f"demand_{k}")

# ----------------------------
# Empty container balance
# ----------------------------

for i in N:
    for ct, cs in container_specs:
        for eo in empty_owners:

            inbound_empty  = gp.quicksum(e[a, ct, cs, eo] for a in in_arcs[i])
            outbound_empty = gp.quicksum(e[a, ct, cs, eo] for a in out_arcs[i])

            loaded = gp.quicksum(
                f[k] for k in K
                if origin[k] == i and commodity_spec[k] == (ct, cs) and owner_of[k] == eo
            )

            if eo == 'Carrier':
                delivered = gp.quicksum(
                    f[k] for k in K
                    if dest[k] == i and commodity_spec[k] == (ct, cs) and owner_of[k] == eo
                )
            else:  # Leased — no return to pool
                delivered = 0

            m.addConstr(
                supply.get((i, ct, cs, eo), 0)
                + inbound_empty
                + delivered
                ==
                outbound_empty
                + loaded
                + I[i[0], i[1], ct, cs, eo],
                name=f"empty_balance_{i}_{ct}_{cs}_{eo}"
            )
    # m.addConstr(
    #     outbound_empty + loaded <=
    #     supply.get(i, 0) + inbound_empty + delivered,
    # )
#     m.addConstr(
#     supply.get(i, 0)
#     + gp.quicksum(e[a] for a in in_arcs[i])
#     + gp.quicksum(f[k] for k in K if dest[k] == i)
#     >=
#     gp.quicksum(e[a] for a in out_arcs[i])
#     + gp.quicksum(f[k] for k in K if origin[k] == i),
#     name=f"empty_balance_relaxed_{i}"
# )
    # m.addConstr(
    #     supply.get(i, 0) + inbound_empty >= outbound_empty,
    #     name=f"empty_upper_{i}"
    # )
    

# ----------------------------
# Solve
# ----------------------------

m.optimize()

# ----------------------------
# Results
# ----------------------------
                
if m.Status == GRB.INFEASIBLE:
    print("Model is infeasible. Computing IIS...")
    m.computeIIS()
    m.write("results/model.ilp")
    print("IIS written to model.ilp")
elif m.Status == GRB.OPTIMAL:
    print("Optimal solution found.")
    # print("\n=== Model Flows with Cost and Time ===")

    rows = []

    for a in A:
        arc = arcs_df.loc[arcs_df["arc_id"] == a].iloc[0]

        dep = arc["DepPort"]
        arr = arc["ArrPort"]
        dep_hour = arc["DepHour"]
        arr_hour = arc["ArrHour"]
        line = arc["Line"]
        vessel = arc["Vessel"]
        cost = arc["Cost"]

        # Cargo flows
        for k in K:
            if x[a, k].X > 1e-6:
                rows.append([
                    dep,
                    dep_hour,
                    arr,
                    arr_hour,
                    line,
                    vessel,
                    k,
                    x[a, k].X,
                    cost
                ])

        # Empty flows (per type, size, and owner)
        for ct, cs in container_specs:
            for eo in empty_owners:
                if e[a, ct, cs, eo].X > 1e-6:
                    rows.append([
                        dep,
                        dep_hour,
                        arr,
                        arr_hour,
                        line,
                        vessel,
                        f"EMPTY_{ct}_{cs}_{eo}",
                        e[a, ct, cs, eo].X,
                        cost
                    ])

    flows_df = pd.DataFrame(
        rows,
        columns=[
            "DepPort",
            "DepHour",
            "ArrPort",
            "ArrHour",
            "Line",
            "Vessel",
            "Commodity",
            "Flow",
            "Cost"
        ]
    )

    # ----------------------------
    # Sort by departure hour
    # ----------------------------
    flows_df = flows_df.sort_values(by="DepHour").reset_index(drop=True)

    flows_df.to_csv("results/model_flows.csv", index=False)
    print("\nFlows saved to model_flows.csv")
    
    # ----------------------------
    # Save node container distribution
    # ----------------------------

    inventory_rows = []

    for i in N:
        port, time = i
        for ct, cs in container_specs:
            for eo in empty_owners:
                inventory_rows.append([
                    port,
                    time,
                    ct,
                    cs,
                    eo,
                    supply.get((i, ct, cs, eo), 0),
                    I[i[0], i[1], ct, cs, eo].X
                ])

    inventory_df = pd.DataFrame(
        inventory_rows,
        columns=["Port", "Time", "ContainerType", "ContainerSize", "Owner", "InitialContainers", "FinalContainers"]
    )

    inventory_df.to_csv("results/node_inventory.csv", index=False)

    print("Node inventory saved to node_inventory.csv")

    # ----------------------------
    # Save end-of-horizon stock 
    # ----------------------------
 
    from collections import defaultdict as _defaultdict
    stock_totals: dict[tuple, float] = _defaultdict(float)  # (port, ct, cs, eo, full_empty)
    for port, h in nodes:
        if h > HORIZON_END:
            continue
        for ct, cs in container_specs:
            for eo in empty_owners:
                val = I[port, h, ct, cs, eo].X
                if val > 1e-6:
                    stock_totals[(port, ct, cs, eo, "Empty")] += val


    vessel_totals: dict[tuple, float] = _defaultdict(float)  # (dep, arr, ct, cs, eo, full_empty)

    crossing_arcs = arcs_df[
        (arcs_df["DepPort"] != arcs_df["ArrPort"]) &
        (arcs_df["DepHour"].astype(int) < HORIZON_END) &
        (arcs_df["ArrHour"].astype(int) > HORIZON_END)
    ]
    for _, arc in crossing_arcs.iterrows():
        a = arc["arc_id"]
        dep, arr = arc["DepPort"], arc["ArrPort"]
        for ct, cs in container_specs:
            for eo in empty_owners:
                val = e[a, ct, cs, eo].X
                if val > 1e-6:
                    vessel_totals[(dep, arr, ct, cs, eo, "Empty")] += val


    cargo_on_vessel = flows_df[
        ~flows_df["Commodity"].str.startswith("EMPTY_") &
        (flows_df["DepPort"] != flows_df["ArrPort"]) &
        (flows_df["DepHour"].astype(int) < HORIZON_END) &
        (flows_df["ArrHour"].astype(int) > HORIZON_END)
    ]
    vessel_k: set = set(cargo_on_vessel["Commodity"])
    for _, row in cargo_on_vessel.iterrows():
        k = row["Commodity"]
        ct, cs = commodity_spec[k]
        eo = owner_of[k]
        vessel_totals[(row["DepPort"], row["ArrPort"], ct, cs, eo, "Full")] += row["Flow"]


    cargo_within_horizon = flows_df[
        ~flows_df["Commodity"].str.startswith("EMPTY_") &
        (flows_df["DepPort"] != flows_df["ArrPort"]) &
        (flows_df["ArrHour"].astype(int) <= HORIZON_END)
    ].copy()

    port_full_totals: dict[tuple, float] = _defaultdict(float)  # (port, ct, cs)
    for k in K:
        if owner_of[k] != "Carrier":
            continue
        fk = f[k].X
        if fk < 1e-6:
            continue
        if dest[k][1] <= HORIZON_END:
            continue  # delivered within horizon; empty already returned to I
        if k in vessel_k:
            continue  

        ct, cs = commodity_spec[k]
        k_rows = cargo_within_horizon[cargo_within_horizon["Commodity"] == k]
        if k_rows.empty:
            current_port = origin[k][0]  
        else:
            current_port = k_rows.loc[k_rows["ArrHour"].idxmax(), "ArrPort"]

        port_full_totals[(current_port, ct, cs)] += round(fk)

    stock_rows = []
    for (port, ct, cs, eo, fe), total in sorted(stock_totals.items()):
        stock_rows.append({
            "Location": port,
            "Last location": "",
            "Next location": "",
            "FullEmpty": fe,
            "With customer": "False",
            "ContainerType": ct,
            "ContainerSize": float(cs),
            "Owner": eo.upper(),
            "count": int(round(total)),
        })
    for (port, ct, cs), total in sorted(port_full_totals.items()):
        stock_rows.append({
            "Location": port,
            "Last location": "",
            "Next location": "",
            "FullEmpty": "Full",
            "With customer": "False",
            "ContainerType": ct,
            "ContainerSize": float(cs),
            "Owner": "CARRIER",
            "count": int(round(total)),
        })
    for (dep, arr, ct, cs, eo, fe), total in sorted(vessel_totals.items()):
        stock_rows.append({
            "Location": "VESSEL",
            "Last location": dep,
            "Next location": arr,
            "FullEmpty": fe,
            "With customer": "False",
            "ContainerType": ct,
            "ContainerSize": float(cs),
            "Owner": eo.upper(),
            "count": int(round(total)),
        })

    stock_dir = Path("results/stock")
    stock_dir.mkdir(exist_ok=True)
    stock_fname = f"{START_YEAR}w{START_WEEK}.csv"
    stock_df = pd.DataFrame(stock_rows)
    stock_df.to_csv(stock_dir / stock_fname, index=False)
    print(f"End stock saved to results/stock/{stock_fname}")
    
    # ----------------------------
    # Save fulfilled demand with vessel info
    # ----------------------------
    fulfillment_rows = []

    for k in K:
        origin_port, origin_time = origin[k]
        dest_port, dest_time = dest[k]
        
        # Find all arcs where this commodity was transported
        vessels_used = [arcs_df.loc[arcs_df['arc_id']==a, 'Vessel'].values[0] 
                        for a in A if x[a,k].X > 1e-6]
        
  
        vessel = vessels_used[0] if vessels_used else ""
        
        fulfilled = f[k].X
        if fulfilled < 1e-6:
            status = "Unfulfilled"
        elif truncated_of[k]:
            # Completed first leg; waiting at transshipment port for next voyage
            status = "WaitingTransship"
        elif dest_time > HORIZON_END:
            status = "InTransit"
        else:
            status = "Delivered"

        fulfillment_rows.append([
            k,
            origin_port,
            dest_port,
            owner_of[k],
            demand[k],
            fulfilled,
            fulfilled / demand[k],
            vessel,
            status
        ])

    fulfillment_df = pd.DataFrame(
        fulfillment_rows,
        columns=["Commodity","Origin","Destination","Owner","Demand","Fulfilled","FillRate","Vessel","Status"]
    )

    fulfillment_df.to_csv("results/demand_fulfillment.csv", index=False)
    print("\nDemand fulfillment saved to demand_fulfillment.csv")

    # ----------------------------
    # Save unfulfilled demand CSV
    # ----------------------------

    comm_df = pd.read_csv(COMMODITIES_PATH)
    comm_df["DepDate"] = comm_df["DepartureTime"].apply(to_dt)
    comm_df["ArrDate"] = comm_df["ArrivalTime"].apply(to_dt)

    unfulfilled_df = fulfillment_df[fulfillment_df["Fulfilled"] < fulfillment_df["Demand"]].copy()
    unfulfilled_df["Unfulfilled"] = unfulfilled_df["Demand"] - unfulfilled_df["Fulfilled"]

    idx_extracted = unfulfilled_df["Commodity"].str.extract(r"^k(\d+)_")[0]
    unfulfilled_df["idx"] = pd.to_numeric(idx_extracted, errors="coerce")
    comm_sub = comm_df[["DepartureTime", "ArrivalTime", "DepDate", "ArrDate"]].reset_index().rename(columns={"index": "idx"})
    unfulfilled_df = unfulfilled_df.merge(comm_sub, on="idx", how="left").drop(columns="idx")
    unfulfilled_df = unfulfilled_df.sort_values("Unfulfilled", ascending=False).reset_index(drop=True)
    unfulfilled_df.to_csv("results/unfulfilled_demand.csv", index=False)
    print(f"Unfulfilled demand saved to unfulfilled_demand.csv ({len(unfulfilled_df)} commodities, {unfulfilled_df['Unfulfilled'].sum():.0f} containers)")

    # ----------------------------
    # Diagnose unfulfilled demand
    # ----------------------------
    
    DIAGNOSE = True
    
    if DIAGNOSE and not unfulfilled_df.empty:
        TEU_SIZE = {20: 1, 40: 2, 45: 2}

        # Arc utilisation: total TEUs flowing on each arc in the solution
        arc_teu = {}
        for a in A:
            arc = arcs_df.loc[arcs_df["arc_id"] == a].iloc[0]
            if arc["DepPort"] == arc["ArrPort"]:
                continue  # skip wait arcs
            teus = sum(
                x[a, k].X * teu[k]
                for k in K if x[a, k].X > 1e-6
            ) + sum(
                e[a, ct, cs, eo].X * TEU_SIZE.get(int(cs), 1)
                for ct in container_types
                for cs in container_sizes
                for eo in empty_owners
                if e[a, ct, cs, eo].X > 1e-6
            )
            arc_key = (arc["DepPort"], arc["DepHour"], arc["ArrPort"], arc["ArrHour"])
            arc_teu[arc_key] = arc_teu.get(arc_key, 0) + teus

        arc_cap = {
            (row["DepPort"], row["DepHour"], row["ArrPort"], row["ArrHour"]): row["Capacity"]
            for _, row in arcs_df[arcs_df["DepPort"] != arcs_df["ArrPort"]].iterrows()
        }

        print("\n=== Unfulfilled Demand Diagnosis ===")
        for k in K:
            if f[k].X >= demand[k] - 1e-6:
                continue

            shortfall = demand[k] - f[k].X
            ct, cs = commodity_spec[k]
            origin_port, origin_time = origin[k]
            dest_port, dest_time     = dest[k]
            teu_needed = TEU_SIZE.get(int(cs), 1)

            print(f"\n  {k}  ({origin_port} -> {dest_port})  "
                  f"Unfulfilled: {shortfall:.0f} x {ct} {cs}ft  "
                #   f"[{teu_needed * shortfall:.0f} TEUs needed]"
                )

            # 1. Find sailing arcs from origin node
            origin_arcs = arcs_df[
                (arcs_df["DepPort"] == origin_port) &
                (arcs_df["DepHour"] == origin_time) &
                (arcs_df["DepPort"] != arcs_df["ArrPort"])
            ]
            if origin_arcs.empty:
                print(f"    -> NO sailing arc from origin node ({origin_port}, h{origin_time})")
                continue

            for _, arc_row in origin_arcs.iterrows():
                akey = (arc_row["DepPort"], arc_row["DepHour"],
                        arc_row["ArrPort"], arc_row["ArrHour"])
                used = arc_teu.get(akey, 0)
                cap  = arc_cap.get(akey, arc_row["Capacity"])
                pct  = used / cap * 100 if cap > 0 else 0
                flag = " ** AT CAPACITY **" if used >= cap - 1 else f" ({pct:.0f}% full)"
                print(f"    Arc {arc_row['DepPort']} -> {arc_row['ArrPort']} "
                      f"via {arc_row['Vessel']}: {used:.0f}/{cap:.0f} TEUs{flag}")

                # If first leg has capacity, the bottleneck is on a continuation leg
                if used < cap - 1:
                    transit_port = arc_row["ArrPort"]
                    arr_hour     = arc_row["ArrHour"]
                    cont_arcs = arcs_df[
                        (arcs_df["DepPort"] == transit_port) &
                        (arcs_df["ArrPort"] != transit_port) &
                        (arcs_df["DepHour"] >= arr_hour)
                    ].sort_values("DepHour")
                    blocked = [
                        (carc, arc_teu.get(
                            (carc["DepPort"], carc["DepHour"], carc["ArrPort"], carc["ArrHour"]), 0),
                         arc_cap.get(
                            (carc["DepPort"], carc["DepHour"], carc["ArrPort"], carc["ArrHour"]),
                            carc["Capacity"])
                        )
                        for _, carc in cont_arcs.iterrows()
                        if arc_teu.get(
                            (carc["DepPort"], carc["DepHour"], carc["ArrPort"], carc["ArrHour"]), 0
                        ) >= arc_cap.get(
                            (carc["DepPort"], carc["DepHour"], carc["ArrPort"], carc["ArrHour"]),
                            carc["Capacity"]
                        ) - 1
                    ]
                    if blocked:
                        print(f"    -> Bottleneck on continuation from {transit_port}:")
                        for carc, cused, ccap in blocked:
                            print(f"       Arc {transit_port} -> {carc['ArrPort']} "
                                  f"via {carc['Vessel']}: {cused:.0f}/{ccap:.0f} TEUs "
                                  f"** AT CAPACITY **")
                    elif cont_arcs.empty:
                        print(f"    -> No continuation arc from {transit_port} after h{arr_hour}")

            # 2. Check empty supply at origin
            # Carrier supply is injected at the port's earliest node and flows
            # via wait arcs — sum across all nodes at this port.
            # Leased supply is injected directly at the origin node.
            if owner_of[k] == 'Carrier':
                sup = sum(
                    v for (node, nct, ncs, neo), v in supply.items()
                    if node[0] == origin_port and nct == ct and ncs == cs and neo == 'Carrier'
                )
                sup_label = "at port"
            else:
                sup = supply.get((origin[k], ct, cs, owner_of[k]), 0)
                sup_label = "at origin node"
            total_demand_at_origin = sum(
                demand[kk] for kk in K
                if origin[kk] == origin[k]
                and commodity_spec[kk] == (ct, cs)
                and owner_of[kk] == owner_of[k]
            )
            if owner_of[k] != "Shipper":
                emp_flag = " ** INSUFFICIENT **" if sup < total_demand_at_origin else ""
                print(f"    Empty supply {sup_label}: {sup:.0f}  "
                      f"(total {ct} {cs}ft {owner_of[k]} demand at node: "
                      f"{total_demand_at_origin:.0f}){emp_flag}")
elif m.Status == GRB.UNBOUNDED:
    print("Model is unbounded.")
else:
    print("Optimization ended with status:", m.Status)
    