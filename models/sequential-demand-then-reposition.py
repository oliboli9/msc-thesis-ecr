from collections import defaultdict
from dataclasses import dataclass
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from datetime import timedelta


@dataclass
class ModelResults:
    flows_df:       pd.DataFrame
    inventory_df:   pd.DataFrame
    fulfillment_df: pd.DataFrame
    stock_df:       pd.DataFrame
    status:         str


def solve(arcs_df: pd.DataFrame,
          commodities_df: pd.DataFrame,
          initial_supply: dict,
          cfg: dict) -> ModelResults:

    EPOCH_DT     = cfg["epoch_dt"]
    STOCK_CUTOFF = cfg["stock_cutoff_hour"]
    HORIZON_END  = cfg["horizon_end"]

    def to_dt(h):
        return (EPOCH_DT + timedelta(hours=int(h))).strftime("%d %b %Y %H:%M")

    # ── Arc data setup ────────────────────────────────────────────────────────

    arcs_df = arcs_df.copy()
    arcs_df["arc_id"] = arcs_df.index

    A = arcs_df["arc_id"].tolist()

    capacity  = arcs_df.set_index("arc_id")["Capacity"].to_dict()
    arc_costs = arcs_df.set_index("arc_id")["Cost"].to_dict()

    arcs_df["dep_node"] = list(zip(arcs_df["DepPort"], arcs_df["DepHour"]))
    arcs_df["arr_node"] = list(zip(arcs_df["ArrPort"], arcs_df["ArrHour"]))

    nodes = set(arcs_df["dep_node"]).union(set(arcs_df["arr_node"]))
    N = list(nodes)

    out_arcs = {i: [] for i in N}
    in_arcs  = {i: [] for i in N}
    for _, row in arcs_df.iterrows():
        a = row["arc_id"]
        out_arcs[row["dep_node"]].append(a)
        in_arcs[row["arr_node"]].append(a)

    # ── Commodity data setup ──────────────────────────────────────────────────

    commodities_df = commodities_df.copy()
    commodities_df["DepartureTime"] = commodities_df["DepartureTime"].astype(int)
    commodities_df["ArrivalTime"]   = commodities_df["ArrivalTime"].astype(int)
    commodities_df["Owner"]         = commodities_df["Owner"].str.title()

    if "Truncated" not in commodities_df.columns:
        commodities_df["Truncated"] = False
    commodities_df["Truncated"] = commodities_df["Truncated"].fillna(False).astype(bool)

    if "Commodity" not in commodities_df.columns:
        commodities_df["Commodity"] = [
            f"k{i}_{row.ContainerType}_{int(row.ContainerSize)}_{row.Owner}"
            for i, row in commodities_df.iterrows()
        ]

    teu            = {}
    commodity_spec = {}
    owner_of       = {}
    truncated_of   = {}

    for _, row in commodities_df.iterrows():
        k     = row["Commodity"]
        size  = row["ContainerSize"]
        ctype = row["ContainerType"]
        teu[k] = 1 if size == 20 else 2
        commodity_spec[k] = (ctype, size)
        owner_of[k]       = row["Owner"]
        truncated_of[k]   = bool(row["Truncated"])

    container_types = set(commodities_df["ContainerType"].unique())
    container_sizes = set(commodities_df["ContainerSize"].unique())
    for (node, ct, cs, eo) in initial_supply:
        container_types.add(ct)
        container_sizes.add(cs)
    container_types = sorted(container_types)
    container_sizes = sorted(container_sizes)
    container_specs = [(ct, cs) for ct in container_types for cs in container_sizes]
    teu_spec        = {(ct, cs): 1 if cs == 20 else 2 for ct, cs in container_specs}
    empty_owners    = ["Carrier"]

    K_all = commodities_df["Commodity"].tolist()

    origin = {
        row["Commodity"]: (row["Origin"], row["DepartureTime"])
        for _, row in commodities_df.iterrows()
    }
    dest = {
        row["Commodity"]: (row["Destination"], row["ArrivalTime"])
        for _, row in commodities_df.iterrows()
    }
    demand = {
        row["Commodity"]: int(row["Count"])
        for _, row in commodities_df.iterrows()
    }

    # Split commodities: decision-horizon (week 1) vs future (weeks 2-4)
    K_decision = [k for k in K_all if origin[k][1] < STOCK_CUTOFF]
    K_future   = [k for k in K_all if origin[k][1] >= STOCK_CUTOFF]

    # ── Supply setup ──────────────────────────────────────────────────────────

    supply = dict(initial_supply)

    # ── Delayed empty return (dwell time) ────────────────────────────────────
    # After delivery, carrier empties take DWELL_HOURS to return from customer.
    DWELL_HOURS = 168

    port_node_times_sorted = defaultdict(list)
    for port, h in nodes:
        port_node_times_sorted[port].append(h)
    for port in port_node_times_sorted:
        port_node_times_sorted[port].sort()

    def delayed_return_node(port, delivery_hour):
        target = delivery_hour + DWELL_HOURS
        for h in port_node_times_sorted[port]:
            if h >= target:
                return (port, h)
        return None

    # Precompute delayed delivery lookups for Phase 1 (K_decision) and Phase 2 (K_all)
    delayed_delivery_p1 = defaultdict(list)
    for k in K_decision:
        if owner_of[k] != "Carrier":
            continue
        dest_port, dest_hour = dest[k]
        ret_node = delayed_return_node(dest_port, dest_hour)
        if ret_node is not None:
            ct, cs = commodity_spec[k]
            delayed_delivery_p1[(ret_node, ct, cs, "Carrier")].append(k)

    delayed_delivery_p2 = defaultdict(list)
    for k in K_all:
        if owner_of[k] != "Carrier":
            continue
        dest_port, dest_hour = dest[k]
        ret_node = delayed_return_node(dest_port, dest_hour)
        if ret_node is not None:
            ct, cs = commodity_spec[k]
            delayed_delivery_p2[(ret_node, ct, cs, "Carrier")].append(k)

    # ======================================================================
    # PHASE 1: Maximise demand fulfillment 
    # ======================================================================

    print("  Phase 1: Solving demand fulfillment (decision horizon only)...")

    m1 = gp.Model("phase1_demand")
    m1.Params.OutputFlag = 0

    x1 = m1.addVars(A, K_decision, lb=0, vtype=GRB.INTEGER, name="x")
    f1 = m1.addVars(K_decision, lb=0, vtype=GRB.INTEGER, name="f")

    # Objective: maximise fulfilled demand minus transport cost
    lambda_cost = 0.01
    m1.setObjective(
        gp.quicksum(f1[k] for k in K_decision)
        - lambda_cost * gp.quicksum(
            gp.quicksum(x1[a, k] for k in K_decision) * arc_costs[a]
            for a in A
        ),
        GRB.MAXIMIZE
    )

    # Capacity — full containers only (no empties in phase 1)
    for a in A:
        m1.addConstr(
            gp.quicksum(teu[k] * x1[a, k] for k in K_decision) <= capacity[a],
            name=f"cap_{a}"
        )

    # Cargo flow conservation
    for k in K_decision:
        for i in N:
            lhs = (gp.quicksum(x1[a, k] for a in out_arcs[i])
                   - gp.quicksum(x1[a, k] for a in in_arcs[i]))
            if i == origin[k]:
                m1.addConstr(lhs == f1[k], name=f"flow_origin_{k}_{i}")
            elif i == dest[k]:
                m1.addConstr(lhs == -f1[k], name=f"flow_dest_{k}_{i}")
            else:
                m1.addConstr(lhs == 0, name=f"flow_trans_{k}_{i}")

    # Demand limits
    for k in K_decision:
        m1.addConstr(f1[k] <= demand[k], name=f"demand_{k}")


    is_wait = {}
    for _, row in arcs_df.iterrows():
        is_wait[row["arc_id"]] = (row["DepPort"] == row["ArrPort"])

    e1_wait = m1.addVars(
        [(a, ct, cs, eo) for a in A if is_wait[a]
         for ct in container_types for cs in container_sizes for eo in empty_owners],
        lb=0, vtype=GRB.INTEGER, name="e_wait")

    I1 = m1.addVars(N, container_types, container_sizes, empty_owners,
                    lb=0, vtype=GRB.INTEGER, name="inv")

    for i in N:
        for ct, cs in container_specs:
            for eo in empty_owners:
                inbound_wait = gp.quicksum(
                    e1_wait[a, ct, cs, eo] for a in in_arcs[i]
                    if is_wait[a])
                outbound_wait = gp.quicksum(
                    e1_wait[a, ct, cs, eo] for a in out_arcs[i]
                    if is_wait[a])

                loaded = gp.quicksum(
                    f1[k] for k in K_decision
                    if origin[k] == i and commodity_spec[k] == (ct, cs) and owner_of[k] == eo
                )
                if eo == "Carrier":
                    # Empties return after dwell time, not at delivery node
                    delayed_ks = delayed_delivery_p1.get((i, ct, cs, eo), [])
                    delivered = gp.quicksum(f1[k] for k in delayed_ks)
                else:
                    delivered = 0

                m1.addConstr(
                    supply.get((i, ct, cs, eo), 0)
                    + inbound_wait
                    + delivered
                    ==
                    outbound_wait
                    + loaded
                    + I1[i[0], i[1], ct, cs, eo],
                    name=f"empty_bal1_{i}_{ct}_{cs}_{eo}"
                )

    m1.optimize()
    print(f"  SEQ_PHASE1_RUNTIME_S={m1.Runtime:.3f}")

    if m1.Status != GRB.OPTIMAL:
        print(f"  Phase 1 failed with status: {m1.Status}")
        empty_df = pd.DataFrame()
        return ModelResults(empty_df, empty_df, empty_df, empty_df, f"PHASE1_{m1.Status}")

    # Extract phase 1 results
    fixed_flows = {}
    for k in K_decision:
        fixed_flows[k] = f1[k].X

    total_fulfilled_p1 = sum(fixed_flows.values())
    total_demand_p1 = sum(demand[k] for k in K_decision)
    print(f"  Phase 1 result: {int(total_fulfilled_p1)}/{total_demand_p1} "
          f"({total_fulfilled_p1/total_demand_p1:.1%}) demand fulfilled")

    # ======================================================================
    # PHASE 2: Optimise empty repositioning for decision horizon, given fixed cargo flows from Phase 1 and future demand info.
    # ======================================================================

    print("  Phase 2: Solving empty repositioning...")

    m2 = gp.Model("phase2_reposition")
    m2.Params.OutputFlag = 0

    # Cargo flows are fixed from Phase 1
    x2 = m2.addVars(A, K_all, lb=0, vtype=GRB.INTEGER, name="x")
    e2 = m2.addVars(A, container_types, container_sizes, empty_owners,
                    lb=0, vtype=GRB.INTEGER, name="e")
    f2 = m2.addVars(K_all, lb=0, vtype=GRB.INTEGER, name="f")
    I2 = m2.addVars(N, container_types, container_sizes, empty_owners,
                    lb=0, vtype=GRB.INTEGER, name="inventory")

    # Fix decision-horizon cargo flows to Phase 1 values
    for k in K_decision:
        m2.addConstr(f2[k] == int(round(fixed_flows[k])), name=f"fix_f_{k}")

    # Objective: maximise future demand fulfillment + minimise repositioning cost
    m2.setObjective(
        gp.quicksum(f2[k] for k in K_future)
        - lambda_cost * gp.quicksum(
            (gp.quicksum(x2[a, k] for k in K_all)
             + gp.quicksum(e2[a, ct, cs, eo]
                           for ct, cs in container_specs for eo in empty_owners)
            ) * arc_costs[a]
            for a in A
        ),
        GRB.MAXIMIZE
    )

    # Capacity
    for a in A:
        m2.addConstr(
            gp.quicksum(teu[k] * x2[a, k] for k in K_all)
            + gp.quicksum(
                teu_spec[ct, cs] * e2[a, ct, cs, eo]
                for ct, cs in container_specs for eo in empty_owners
            ) <= capacity[a],
            name=f"cap_{a}"
        )

    # Cargo flow conservation (all commodities)
    for k in K_all:
        for i in N:
            lhs = (gp.quicksum(x2[a, k] for a in out_arcs[i])
                   - gp.quicksum(x2[a, k] for a in in_arcs[i]))
            if i == origin[k]:
                m2.addConstr(lhs == f2[k], name=f"flow_origin_{k}_{i}")
            elif i == dest[k]:
                m2.addConstr(lhs == -f2[k], name=f"flow_dest_{k}_{i}")
            else:
                m2.addConstr(lhs == 0, name=f"flow_trans_{k}_{i}")

    # Demand limits
    for k in K_all:
        m2.addConstr(f2[k] <= demand[k], name=f"demand_{k}")

    # Empty container balance
    for i in N:
        for ct, cs in container_specs:
            for eo in empty_owners:
                inbound_empty  = gp.quicksum(e2[a, ct, cs, eo] for a in in_arcs[i])
                outbound_empty = gp.quicksum(e2[a, ct, cs, eo] for a in out_arcs[i])
                loaded = gp.quicksum(
                    f2[k] for k in K_all
                    if origin[k] == i and commodity_spec[k] == (ct, cs) and owner_of[k] == eo
                )
                if eo == "Carrier":
                    # Empties return after dwell time, not at delivery node
                    delayed_ks = delayed_delivery_p2.get((i, ct, cs, eo), [])
                    delivered = gp.quicksum(f2[k] for k in delayed_ks)
                else:
                    delivered = 0

                m2.addConstr(
                    supply.get((i, ct, cs, eo), 0)
                    + inbound_empty
                    + delivered
                    ==
                    outbound_empty
                    + loaded
                    + I2[i[0], i[1], ct, cs, eo],
                    name=f"empty_balance_{i}_{ct}_{cs}_{eo}"
                )

    m2.optimize()
    print(f"  SEQ_PHASE2_RUNTIME_S={m2.Runtime:.3f}")

    if m2.Status != GRB.OPTIMAL:
        print(f"  Phase 2 failed with status: {m2.Status}")
        empty_df = pd.DataFrame()
        return ModelResults(empty_df, empty_df, empty_df, empty_df, f"PHASE2_{m2.Status}")

    total_fulfilled_p2_future = sum(f2[k].X for k in K_future)
    total_demand_future = sum(demand[k] for k in K_future)
    if total_demand_future > 0:
        print(f"  Phase 2 result: {int(total_fulfilled_p2_future)}/{total_demand_future} "
              f"({total_fulfilled_p2_future/total_demand_future:.1%}) future demand fulfilled")
    print(f"  Optimal solution found.")

    # Results

    # Flows DataFrame
    rows = []
    for a in A:
        arc      = arcs_df.loc[arcs_df["arc_id"] == a].iloc[0]
        dep, arr = arc["DepPort"], arc["ArrPort"]
        dep_hour, arr_hour = arc["DepHour"], arc["ArrHour"]
        line, vessel = arc["Line"], arc["Vessel"]
        ac = arc["Cost"]

        for k in K_all:
            if x2[a, k].X > 1e-6:
                rows.append([dep, dep_hour, arr, arr_hour, line, vessel,
                             k, x2[a, k].X, ac])
        for ct, cs in container_specs:
            for eo in empty_owners:
                if e2[a, ct, cs, eo].X > 1e-6:
                    rows.append([dep, dep_hour, arr, arr_hour, line, vessel,
                                 f"EMPTY_{ct}_{cs}_{eo}", e2[a, ct, cs, eo].X, ac])

    flows_df = pd.DataFrame(rows, columns=[
        "DepPort", "DepHour", "ArrPort", "ArrHour",
        "Line", "Vessel", "Commodity", "Flow", "Cost"
    ]).sort_values("DepHour").reset_index(drop=True)

    # Inventory DataFrame
    inventory_rows = []
    for i in N:
        port, time = i
        for ct, cs in container_specs:
            for eo in empty_owners:
                inventory_rows.append([
                    port, time, ct, cs, eo,
                    supply.get((i, ct, cs, eo), 0),
                    I2[i[0], i[1], ct, cs, eo].X
                ])
    inventory_df = pd.DataFrame(inventory_rows, columns=[
        "Port", "Time", "ContainerType", "ContainerSize", "Owner",
        "InitialContainers", "FinalContainers"
    ])

    # Stock DataFrame
    stock_totals: dict[tuple, float] = defaultdict(float)
    for port, h in nodes:
        if h > STOCK_CUTOFF:
            continue
        for ct, cs in container_specs:
            for eo in empty_owners:
                val = I2[port, h, ct, cs, eo].X
                if val > 1e-6:
                    stock_totals[(port, ct, cs, eo, "Empty")] += val

    wait_crossing_arcs = arcs_df[
        (arcs_df["DepPort"] == arcs_df["ArrPort"]) &
        (arcs_df["DepHour"].astype(int) <= STOCK_CUTOFF) &
        (arcs_df["ArrHour"].astype(int) > STOCK_CUTOFF)
    ]
    for _, arc in wait_crossing_arcs.iterrows():
        a = arc["arc_id"]
        port = arc["DepPort"]
        for ct, cs in container_specs:
            for eo in empty_owners:
                val = e2[a, ct, cs, eo].X
                if val > 1e-6:
                    stock_totals[(port, ct, cs, eo, "Empty")] += val

    vessel_totals: dict[tuple, float] = defaultdict(float)
    crossing_arcs = arcs_df[
        (arcs_df["DepPort"] != arcs_df["ArrPort"]) &
        (arcs_df["DepHour"].astype(int) <= STOCK_CUTOFF) &
        (arcs_df["ArrHour"].astype(int) > STOCK_CUTOFF)
    ]
    for _, arc in crossing_arcs.iterrows():
        a = arc["arc_id"]
        dep, arr = arc["DepPort"], arc["ArrPort"]
        arr_hour = int(arc["ArrHour"])
        for ct, cs in container_specs:
            for eo in empty_owners:
                val = e2[a, ct, cs, eo].X
                if val > 1e-6:
                    vessel_totals[(dep, arr, arr_hour, ct, cs, eo, "Empty")] += val

    cargo_on_vessel = flows_df[
        ~flows_df["Commodity"].str.startswith("EMPTY_") &
        (flows_df["DepPort"] != flows_df["ArrPort"]) &
        (flows_df["DepHour"].astype(int) <= STOCK_CUTOFF) &
        (flows_df["ArrHour"].astype(int) > STOCK_CUTOFF)
    ]
    vessel_k: set = set(cargo_on_vessel["Commodity"])
    for _, row in cargo_on_vessel.iterrows():
        k = row["Commodity"]
        ct, cs = commodity_spec[k]
        eo = owner_of[k]
        arr_hour = int(row["ArrHour"])
        vessel_totals[(row["DepPort"], row["ArrPort"], arr_hour, ct, cs, eo, "Full")] += row["Flow"]

    cargo_within_horizon = flows_df[
        ~flows_df["Commodity"].str.startswith("EMPTY_") &
        (flows_df["DepPort"] != flows_df["ArrPort"]) &
        (flows_df["ArrHour"].astype(int) <= STOCK_CUTOFF)
    ].copy()

    port_full_totals: dict[tuple, float] = defaultdict(float)
    with_customer_totals: dict[tuple, float] = defaultdict(float)
    for k in K_all:
        if owner_of[k] != "Carrier":
            continue
        fk = f2[k].X
        if fk < 1e-6 or origin[k][1] > STOCK_CUTOFF:
            continue
        if dest[k][1] <= STOCK_CUTOFF:
            dest_port, dest_time = dest[k]
            ret_node = delayed_return_node(dest_port, dest_time)
            if ret_node is None or ret_node[1] > STOCK_CUTOFF:
                ct, cs = commodity_spec[k]
                return_hour = dest_time + DWELL_HOURS
                with_customer_totals[(dest_port, ct, cs, return_hour)] += round(fk)
            continue
        if k in vessel_k:
            continue
        ct, cs = commodity_spec[k]
        k_rows = cargo_within_horizon[cargo_within_horizon["Commodity"] == k]
        current_port = k_rows.loc[k_rows["ArrHour"].idxmax(), "ArrPort"] if not k_rows.empty else origin[k][0]
        port_full_totals[(current_port, ct, cs)] += round(fk)

    stock_rows = []
    for (port, ct, cs, eo, fe), total in sorted(stock_totals.items()):
        stock_rows.append({
            "Location": port, "Last location": "", "Next location": "",
            "FullEmpty": fe, "With customer": "False",
            "ContainerType": ct, "ContainerSize": float(cs),
            "Owner": eo.upper(), "count": int(round(total)),
        })
    for (port, ct, cs), total in sorted(port_full_totals.items()):
        stock_rows.append({
            "Location": port, "Last location": "", "Next location": "",
            "FullEmpty": "Full", "With customer": "False",
            "ContainerType": ct, "ContainerSize": float(cs),
            "Owner": "CARRIER", "count": int(round(total)),
        })
    for (dep, arr, arr_hour, ct, cs, eo, fe), total in sorted(vessel_totals.items()):
        stock_rows.append({
            "Location": "VESSEL", "Last location": dep, "Next location": arr,
            "ArrivalTime": arr_hour,
            "FullEmpty": fe, "With customer": "False",
            "ContainerType": ct, "ContainerSize": float(cs),
            "Owner": eo.upper(), "count": int(round(total)),
        })
    for (port, ct, cs, return_hour), total in sorted(with_customer_totals.items()):
        stock_rows.append({
            "Location": port, "Last location": "", "Next location": "",
            "ArrivalTime": return_hour,
            "FullEmpty": "Empty", "With customer": "True",
            "ContainerType": ct, "ContainerSize": float(cs),
            "Owner": "CARRIER", "count": int(round(total)),
        })
    stock_df = pd.DataFrame(stock_rows)

    # Fulfillment DataFrame
    fulfillment_rows = []
    for k in K_all:
        origin_port, origin_time = origin[k]
        dest_port, dest_time     = dest[k]
        vessels_used = [arcs_df.loc[arcs_df["arc_id"] == a, "Vessel"].values[0]
                        for a in A if x2[a, k].X > 1e-6]
        vessel = vessels_used[0] if vessels_used else ""
        fulfilled = f2[k].X

        if fulfilled < 1e-6:
            status = "Unfulfilled"
        elif truncated_of[k]:
            status = "WaitingTransship"
        elif dest_time > STOCK_CUTOFF:
            status = "InTransit"
        else:
            status = "Delivered"

        fulfillment_rows.append([
            k, origin_port, dest_port, owner_of[k],
            demand[k], fulfilled, fulfilled / demand[k],
            vessel, status
        ])

    fulfillment_df = pd.DataFrame(fulfillment_rows, columns=[
        "Commodity", "Origin", "Destination", "Owner",
        "Demand", "Fulfilled", "FillRate", "Vessel", "Status"
    ])

    n_unfulfilled = (fulfillment_df["Fulfilled"] < fulfillment_df["Demand"]).sum()
    n_containers  = int((fulfillment_df["Demand"] - fulfillment_df["Fulfilled"]).sum())
    print(f"  Unfulfilled demand: {n_unfulfilled} commodities, {n_containers} containers")

    return ModelResults(flows_df, inventory_df, fulfillment_df, stock_df, "OPTIMAL")
