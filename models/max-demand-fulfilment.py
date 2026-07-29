from collections import defaultdict
from dataclasses import dataclass
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from datetime import timedelta


@dataclass
class ModelResults:
    flows_df:       pd.DataFrame   # DepPort, DepHour, ArrPort, ArrHour, Line, Vessel, Commodity, Flow, Cost
    inventory_df:   pd.DataFrame   # Port, Time, ContainerType, ContainerSize, Owner, InitialContainers, FinalContainers
    fulfillment_df: pd.DataFrame   # Commodity, Origin, Destination, Owner, Demand, Fulfilled, FillRate, Vessel, Status
    stock_df:       pd.DataFrame   # Stock_01012023.csv format
    status:         str            # "OPTIMAL" | "INFEASIBLE" | "OTHER_{code}"


def solve(arcs_df: pd.DataFrame,
          commodities_df: pd.DataFrame,
          initial_supply: dict,
          cfg: dict) -> ModelResults:

    EPOCH_DT        = cfg["epoch_dt"]
    STOCK_CUTOFF    = cfg["stock_cutoff_hour"]   # decision horizon end 
    HORIZON_END     = cfg["horizon_end"]          # planning horizon end 

    def to_dt(h):
        return (EPOCH_DT + timedelta(hours=int(h))).strftime("%d %b %Y %H:%M")

    # ── Arc data setup ────────────────────────────────────────────────────────

    arcs_df = arcs_df.copy()
    arcs_df["arc_id"] = arcs_df.index

    A = arcs_df["arc_id"].tolist()

    capacity = arcs_df.set_index("arc_id")["Capacity"].to_dict()
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

    teu          = {}
    commodity_spec = {}
    owner_of       = {}
    truncated_of   = {}

    for _, row in commodities_df.iterrows():
        k = row["Commodity"]
        size  = row["ContainerSize"]
        ctype = row["ContainerType"]

        if size == 20:
            teu[k] = 1
        elif size in [40, 45]:
            teu[k] = 2
        else:
            raise ValueError(f"Unknown container size: {size}")

        commodity_spec[k] = (ctype, size)
        owner_of[k]        = row["Owner"]
        truncated_of[k]    = bool(row["Truncated"])

    container_types = set(commodities_df["ContainerType"].unique())
    container_sizes = set(commodities_df["ContainerSize"].unique())

    for (_, ct, cs, eo) in initial_supply:
        container_types.add(ct)
        container_sizes.add(cs)
    container_types = sorted(container_types)
    container_sizes = sorted(container_sizes)
    container_specs = [(ct, cs) for ct in container_types for cs in container_sizes]
    teu_spec        = {(ct, cs): 1 if cs == 20 else 2 for ct, cs in container_specs}
    empty_owners    = ["Carrier"]

    K = commodities_df["Commodity"].tolist()

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


    supply = dict(initial_supply)

    # -- Gurobi model ------------------------------------

    m = gp.Model("container_repositioning")
    m.Params.MIPGap = cfg.get("mip_gap", 1e-4)
    if "seed" in cfg:
        m.Params.Seed = int(cfg["seed"])

    x = m.addVars(A, K, lb=0, vtype=GRB.INTEGER, name="x")
    e = m.addVars(A, container_types, container_sizes, empty_owners,
                  lb=0, vtype=GRB.INTEGER, name="e")
    f = m.addVars(K, lb=0, vtype=GRB.INTEGER, name="f")
    I = m.addVars(N, container_types, container_sizes, empty_owners,
                  lb=0, vtype=GRB.INTEGER, name="inventory")

    lambda_cost = float(cfg.get("lambda_cost", 0.01))

    term_reward  = float(cfg.get("terminal_empty_reward", 0.0))
    term_ports   = set(cfg.get("terminal_empty_ports", []))
    idle_penalty = float(cfg.get("idle_empty_penalty", 0.0))
    idle_ports   = set(cfg.get("idle_empty_ports", []))

    obj = (
        gp.quicksum(f[k] for k in K)
        - lambda_cost * gp.quicksum(
            (gp.quicksum(x[a, k] for k in K)
             + gp.quicksum(e[a, ct, cs, eo]
                           for ct, cs in container_specs for eo in empty_owners)
            ) * arc_costs[a]
            for a in A
        )
    )

    if term_reward > 0 and term_ports:
        last_hour = {}
        for (pp, hh) in N:
            if pp in term_ports and (pp not in last_hour or hh > last_hour[pp]):
                last_hour[pp] = hh
        obj += term_reward * gp.quicksum(
            I[p, last_hour[p], ct, cs, "Carrier"]
            for p in term_ports if p in last_hour
            for ct, cs in container_specs
        )

    if idle_penalty > 0 and idle_ports:
        obj -= idle_penalty * gp.quicksum(
            I[p, h, ct, cs, "Carrier"]
            for (p, h) in N if p in idle_ports
            for ct, cs in container_specs
        )

    m.setObjective(obj, GRB.MAXIMIZE)

    # Capacity constraints
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

    # Cargo flow conservation

    nodes_set = set(N)

    for k in K:
        over_horizon_dest = dest[k][1] > HORIZON_END or dest[k] not in nodes_set

        for i in N:
            lhs = (gp.quicksum(x[a, k] for a in out_arcs[i])
                   - gp.quicksum(x[a, k] for a in in_arcs[i]))
            if i == origin[k]:
                m.addConstr(lhs == f[k],  name=f"flow_origin_{k}_{i}")
            elif over_horizon_dest:
                m.addConstr(lhs <= 0,     name=f"flow_inflight_{k}_{i}")
            elif i == dest[k]:
                m.addConstr(lhs == -f[k], name=f"flow_dest_{k}_{i}")
            else:
                m.addConstr(lhs == 0,     name=f"flow_trans_{k}_{i}")

    # Demand limits
    for k in K:
        m.addConstr(f[k] <= demand[k], name=f"demand_{k}")

    # Delayed empty return (dwell time) 

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

    delayed_delivery = defaultdict(list)
    for k in K:
        if owner_of[k] != "Carrier":
            continue
        dest_port, dest_hour = dest[k]
        ret_node = delayed_return_node(dest_port, dest_hour)
        if ret_node is not None:
            ct, cs = commodity_spec[k]
            delayed_delivery[(ret_node, ct, cs, "Carrier")].append(k)

    # Empty container balance
    for i in N:
        for ct, cs in container_specs:
            for eo in empty_owners:
                inbound_empty  = gp.quicksum(e[a, ct, cs, eo] for a in in_arcs[i])
                outbound_empty = gp.quicksum(e[a, ct, cs, eo] for a in out_arcs[i])

                loaded = gp.quicksum(
                    f[k] for k in K
                    if origin[k] == i and commodity_spec[k] == (ct, cs) and owner_of[k] == eo
                )

                if eo == "Carrier":
                    delayed_ks = delayed_delivery.get((i, ct, cs, eo), [])
                    delivered = gp.quicksum(f[k] for k in delayed_ks)
                else:
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

    # Solve 

    m.optimize()
    print(f"  GUROBI_RUNTIME_S={m.Runtime:.3f}")

    # Results 

    if m.Status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        m.computeIIS()
        m.write("results/model.ilp")
        print("IIS written to results/model.ilp")
        empty_df = pd.DataFrame()
        return ModelResults(empty_df, empty_df, empty_df, empty_df, "INFEASIBLE")

    if m.Status != GRB.OPTIMAL:
        print(f"Optimization ended with status: {m.Status}")
        empty_df = pd.DataFrame()
        return ModelResults(empty_df, empty_df, empty_df, empty_df, f"OTHER_{m.Status}")

    print("Optimal solution found.")


    rows = []
    for a in A:
        arc      = arcs_df.loc[arcs_df["arc_id"] == a].iloc[0]
        dep      = arc["DepPort"]
        arr      = arc["ArrPort"]
        dep_hour = arc["DepHour"]
        arr_hour = arc["ArrHour"]
        line     = arc["Line"]
        vessel   = arc["Vessel"]
        arc_cost = arc["Cost"]

        for k in K:
            if x[a, k].X > 1e-6:
                rows.append([dep, dep_hour, arr, arr_hour, line, vessel,
                             k, x[a, k].X, arc_cost])

        for ct, cs in container_specs:
            for eo in empty_owners:
                if e[a, ct, cs, eo].X > 1e-6:
                    rows.append([dep, dep_hour, arr, arr_hour, line, vessel,
                                 f"EMPTY_{ct}_{cs}_{eo}", e[a, ct, cs, eo].X, arc_cost])

    flows_df = pd.DataFrame(rows, columns=[
        "DepPort", "DepHour", "ArrPort", "ArrHour",
        "Line", "Vessel", "Commodity", "Flow", "Cost"
    ]).sort_values("DepHour").reset_index(drop=True)


    inventory_rows = []
    for i in N:
        port, time = i
        for ct, cs in container_specs:
            for eo in empty_owners:
                inventory_rows.append([
                    port, time, ct, cs, eo,
                    supply.get((i, ct, cs, eo), 0),
                    I[i[0], i[1], ct, cs, eo].X
                ])
    inventory_df = pd.DataFrame(inventory_rows, columns=[
        "Port", "Time", "ContainerType", "ContainerSize", "Owner",
        "InitialContainers", "FinalContainers"
    ])


    stock_totals: dict[tuple, float] = defaultdict(float)
    for port, h in nodes:
        if h > STOCK_CUTOFF:
            continue
        for ct, cs in container_specs:
            for eo in empty_owners:
                val = I[port, h, ct, cs, eo].X
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
                val = e[a, ct, cs, eo].X
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
                val = e[a, ct, cs, eo].X
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
    for k in K:
        if owner_of[k] != "Carrier":
            continue
        fk = f[k].X
        if fk < 1e-6:
            continue
        if origin[k][1] > STOCK_CUTOFF:
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
        if k_rows.empty:
            current_port = origin[k][0]
        else:
            current_port = k_rows.loc[k_rows["ArrHour"].idxmax(), "ArrPort"]
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


    fulfillment_rows = []
    for k in K:
        origin_port, origin_time = origin[k]
        dest_port, dest_time     = dest[k]

        vessels_used = [arcs_df.loc[arcs_df["arc_id"] == a, "Vessel"].values[0]
                        for a in A if x[a, k].X > 1e-6]
        vessel = vessels_used[0] if vessels_used else ""

        fulfilled = f[k].X
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
    print(f"Unfulfilled demand: {n_unfulfilled} commodities, {n_containers} containers")


    DIAGNOSE = cfg.get("verbose_unfulfilled", True)

    if DIAGNOSE and n_unfulfilled > 0:
        TEU_SIZE = {20: 1, 40: 2, 45: 2}

        arc_used = {}
        arc_cap_by_id = {}
        for a in A:
            arc = arcs_df.loc[arcs_df["arc_id"] == a].iloc[0]
            arc_cap_by_id[a] = arc["Capacity"]
            if arc["DepPort"] == arc["ArrPort"]:
                continue  
            arc_used[a] = (
                sum(x[a, kk].X * teu[kk] for kk in K if x[a, kk].X > 1e-6)
                + sum(e[a, ct, cs, eo].X * TEU_SIZE.get(int(cs), 1)
                      for ct in container_types for cs in container_sizes
                      for eo in empty_owners if e[a, ct, cs, eo].X > 1e-6)
            )

        def arc_label(a):
            arc = arcs_df.loc[arcs_df["arc_id"] == a].iloc[0]
            used = arc_used.get(a, 0)
            cap  = arc_cap_by_id[a]
            pct  = (used / cap * 100) if cap > 0 else 0
            tag  = "AT CAPACITY" if used >= cap - 1 else f"{pct:.0f}% full"
            return (f"{arc['DepPort']}@h{int(arc['DepHour'])} -> "
                    f"{arc['ArrPort']}@h{int(arc['ArrHour'])} "
                    f"via {arc['Vessel']}: {used:.0f}/{cap:.0f} TEUs ({tag})")

        def reachable_nodes(start_node):
            """BFS from start_node along all arcs (ignoring capacity)."""
            seen = {start_node}
            stack = [start_node]
            while stack:
                n = stack.pop()
                for a in out_arcs.get(n, []):
                    arc = arcs_df.loc[arcs_df["arc_id"] == a].iloc[0]
                    nxt = (arc["ArrPort"], arc["ArrHour"])
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            return seen

        def reachable_with_capacity(start_node):
            """BFS using only arcs with residual capacity (>=1 TEU free)."""
            seen = {start_node}
            stack = [start_node]
            while stack:
                n = stack.pop()
                for a in out_arcs.get(n, []):
                    arc = arcs_df.loc[arcs_df["arc_id"] == a].iloc[0]
                    used = arc_used.get(a, 0)
                    cap  = arc_cap_by_id[a]
                    # wait arcs (no entry in arc_used) treated as uncapacitated
                    if a in arc_used and used >= cap - 1:
                        continue
                    nxt = (arc["ArrPort"], arc["ArrHour"])
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            return seen

        print("\n=== Unfulfilled Demand Diagnosis ===")
        for k in K:
            if f[k].X >= demand[k] - 1e-6:
                continue

            shortfall = demand[k] - f[k].X
            ct, cs = commodity_spec[k]
            origin_node = origin[k]
            dest_node   = dest[k]
            origin_port, origin_time = origin_node
            dest_port,   dest_time   = dest_node
            over_horizon = dest_time > HORIZON_END or dest_node not in nodes_set

            print(f"\n  {k}  ({origin_port}@h{origin_time} -> "
                  f"{dest_port}@h{dest_time})  "
                  f"Unfulfilled: {shortfall:.0f} x {ct} {cs}ft  "
                  f"[{owner_of[k]}]")

            # 0. Origin node existence
            if origin_node not in nodes_set:
                print(f"    ROOT CAUSE: origin node {origin_node} not in time-space graph")
                continue

            # 1. Destination presence / horizon relaxation
            if over_horizon:
                print(f"    Destination is past horizon (HORIZON_END=h{HORIZON_END}) "
                      f"-> intermediate conservation relaxed to <= 0")
            elif dest_node not in nodes_set:
                print(f"    ROOT CAUSE: dest node {dest_node} not in time-space graph "
                      f"(no arc arrives there)")
                continue

            # 2. Reachability (capacity-ignoring) from origin to dest
            if not over_horizon:
                reach_all = reachable_nodes(origin_node)
                if dest_node not in reach_all:
                    print(f"    ROOT CAUSE: no path in time-space graph from "
                          f"{origin_node} to {dest_node} (regardless of capacity)")
                    # show what's reachable at the dest port
                    same_port_reachable = sorted(
                        h for (p, h) in reach_all if p == dest_port and h <= dest_time
                    )
                    if same_port_reachable:
                        print(f"    -> reachable times at {dest_port}: "
                              f"{same_port_reachable[:5]}{' ...' if len(same_port_reachable) > 5 else ''}"
                              f" (need h{dest_time})")
                    else:
                        print(f"    -> {dest_port} not reachable at all from origin")
                    continue

                # 2b. Inflow into dest node in current solution
                inflow = sum(x[a, k].X for a in in_arcs.get(dest_node, []))
                print(f"    Inflow into dest node in solution: {inflow:.0f} "
                      f"(demand: {demand[k]})")

            # 3. Capacity-respecting reachability
            reach_cap = reachable_with_capacity(origin_node)
            target_reached = (dest_node in reach_cap) if not over_horizon else any(
                p != origin_port for (p, _) in reach_cap
            )
            if not target_reached:
                print("    ROOT CAUSE: every path to destination has at least one "
                      "saturated arc (capacity bottleneck)")
                # print the saturated arcs out of origin to start the trail
                sat_from_origin = [a for a in out_arcs.get(origin_node, [])
                                   if a in arc_used
                                   and arc_used[a] >= arc_cap_by_id[a] - 1]
                if sat_from_origin:
                    print("    Saturated arcs out of origin:")
                    for a in sat_from_origin:
                        print(f"       {arc_label(a)}")

            # 4. Empty supply at the origin node (Carrier only)
            if owner_of[k] == "Carrier":
                sup_node = supply.get((origin_node, ct, cs, "Carrier"), 0)
                sup_port_total = sum(
                    v for (n, nct, ncs, neo), v in supply.items()
                    if n[0] == origin_port and nct == ct and ncs == cs
                    and neo == "Carrier"
                )
                demand_at_node = sum(
                    demand[kk] for kk in K
                    if origin[kk] == origin_node
                    and commodity_spec[kk] == (ct, cs)
                    and owner_of[kk] == "Carrier"
                )
                # delayed returns landing at this exact node
                returns_here = sum(
                    f[kk].X for kk in
                    delayed_delivery.get((origin_node, ct, cs, "Carrier"), [])
                )
                avail_node = sup_node + returns_here
                flag = " ** INSUFFICIENT AT NODE **" if avail_node < demand_at_node else ""
                print(f"    Empty supply at origin node {origin_node}: "
                      f"{sup_node:.0f} stock + {returns_here:.0f} returning "
                      f"= {avail_node:.0f}  (demand at node: {demand_at_node:.0f}){flag}")
                print(f"    Empty supply across all times at {origin_port}: "
                      f"{sup_port_total:.0f} (reachable via wait/inbound arcs)")

            # 5. Origin departure arcs (capacity at the first leg)
            origin_sail_arcs = [a for a in out_arcs.get(origin_node, [])
                                if a in arc_used]
            if origin_sail_arcs:
                print("    Sailings from origin node:")
                for a in origin_sail_arcs:
                    print(f"       {arc_label(a)}")
            else:
                print(f"    No direct sailing from origin node h{origin_time}; "
                      f"must use wait arcs to a later departure")

    return ModelResults(flows_df, inventory_df, fulfillment_df, stock_df, "OPTIMAL")


# -------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from datetime import datetime

    sys.path.insert(0, "pipeline")
    from config import EPOCH_HOUR_OFFSET, H, CALENDAR_WEEKS, HORIZON_MONDAY, START_YEAR, START_WEEK, ARCS_PATH, COMMODITIES_PATH

    arcs_df = pd.read_csv(ARCS_PATH)
    commodities_df = pd.read_csv(COMMODITIES_PATH)

    # Compute epoch_dt
    _v = pd.read_csv("data/raw/Eimskip_voyages.csv", parse_dates=["etdDateTime"])
    _mask = False
    for _yr, _wk in CALENDAR_WEEKS:
        _mask = _mask | ((_v["year"] == _yr) & (_v["week"] == _wk))
    _etds = _v[_mask]["etdDateTime"]
    _epoch_dt = (_etds.min().normalize().to_pydatetime() if not _etds.empty
                 else datetime(HORIZON_MONDAY.year, HORIZON_MONDAY.month, HORIZON_MONDAY.day))

    cfg = {
        "epoch_hour_offset": EPOCH_HOUR_OFFSET,
        "H": H,
        "horizon_end": EPOCH_HOUR_OFFSET + H,
        "stock_cutoff_hour": EPOCH_HOUR_OFFSET + H,  
        "epoch_dt": _epoch_dt,
        "calendar_weeks": CALENDAR_WEEKS,
    }

    results = solve(arcs_df, commodities_df, initial_supply={}, cfg=cfg)

    if results.status == "OPTIMAL":
        Path("results/stock").mkdir(parents=True, exist_ok=True)
        results.flows_df.to_csv("results/model_flows.csv", index=False)
        results.inventory_df.to_csv("results/node_inventory.csv", index=False)
        results.fulfillment_df.to_csv("results/demand_fulfillment.csv", index=False)
        results.stock_df.to_csv(f"results/stock/{START_YEAR}w{START_WEEK}.csv", index=False)
        print("Results written to results/")
