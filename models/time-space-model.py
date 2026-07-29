import sys
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from itertools import product

sys.path.insert(0, "pipeline")
from config import ARCS_PATH

# -----------------------------
# 1. READ ARC CSV
# -----------------------------
arcs_df = pd.read_csv(ARCS_PATH)

# Make arcs unique keys: (dep_node, arr_node, line, vessel)
A = []
c = {}
u = {}
for idx, row in arcs_df.iterrows():
    dep = (row['DepPort'], int(row['DepHour']))
    arr = (row['ArrPort'], int(row['ArrHour']))
    line = row['Line']
    vessel = row['Vessel']
    arc = (dep, arr, line, vessel)
    A.append(arc)
    c[arc] = float(row['Cost'])
    u[arc] = int(row['Capacity'])
    

# -----------------------------
# 2. DEFINE COMMODITIES
# -----------------------------
commodities = {
    "C1": {"origin": ("IS REY", 198), "destination": ("IS GRT", 200), "demand": 100, "full": True},
    "E1": {"origin": ("IS GRT", 200), "destination": ("IS REY", 198), "demand": 100, "full": False}  # empty
}
K_set = list(commodities.keys())

# -----------------------------
# 3. NODES
# -----------------------------
N = set()
for arc in A:
    dep_node, arr_node = arc[0], arc[1]
    N.add(dep_node)
    N.add(arr_node)

# -----------------------------
# 4. BUILD MODEL
# -----------------------------
m = gp.Model("TimeSpace_MC_Integer")

# Decision variables: integer
# x = m.addVars(A, K_set, lb=0, vtype=GRB.INTEGER, name="x")
x = m.addVars(list(product(A, K_set)), lb=0, vtype=GRB.INTEGER, name="x")


# Objective: minimize cost
m.setObjective(gp.quicksum(c[arc]*x[arc,k] for arc in A for k in K_set), GRB.MINIMIZE)
# -----------------------------
# 5. FLOW BALANCE FOR FULL CONTAINERS
# -----------------------------
for k in K_set:
    commodity = commodities[k]
    if commodity["full"]:
        for node in N:
            b = 0
            if node == commodity["origin"]:
                b = commodity["demand"]
            elif node == commodity["destination"]:
                b = -commodity["demand"]
            m.addConstr(
                gp.quicksum(x[arc,k] for arc in A if arc[0]==node) -
                gp.quicksum(x[arc,k] for arc in A if arc[1]==node) == b,
                name=f"flow_full_{k}_{node}"
            )

# -----------------------------
# 6. NODE BALANCE (all containers)
# -----------------------------
for node in N:
    m.addConstr(
        gp.quicksum(x[arc,k] for arc in A for k in K_set if arc[0]==node) -
        gp.quicksum(x[arc,k] for arc in A for k in K_set if arc[1]==node) <= 0,
        name=f"send_le_receive_{node}"
    )

# -----------------------------
# 7. ARC CAPACITY
# -----------------------------
for arc in A:
    m.addConstr(
        gp.quicksum(x[arc,k] for k in K_set) <= u[arc],
        name=f"cap_{arc}"
    )

# -----------------------------
# 8. SOLVE MODEL
# -----------------------------
m.optimize()

# -----------------------------
# 9. HANDLE INFEASIBILITY
# -----------------------------
if m.Status == GRB.INFEASIBLE:
    print("Model is infeasible. Computing IIS...")
    m.computeIIS()
    m.write("results/model.ilp")
    print("IIS written to model.ilp")
elif m.Status == GRB.OPTIMAL:
    print("Optimal solution found.")
elif m.Status == GRB.UNBOUNDED:
    print("Model is unbounded.")
else:
    print("Optimization ended with status:", m.Status)

# -----------------------------
# 10. EXTRACT FLOWS
# -----------------------------
solution = []
if m.Status == GRB.OPTIMAL:
    for arc in A:
        for k in K_set:
            val = x[arc, k].X
            if val > 0:
                solution.append({
                    "DepPort": arc[0],
                    "ArrPort": arc[1],
                    "Line": arc[2],
                    "Vessel": arc[3],
                    "Commodity": k,
                    "Flow": val,
                    "Cost": val * c[arc]  # total cost for this flow
                })

    df_solution = pd.DataFrame(solution)
    print("\n=== Model Flows with Cost ===")
    print(df_solution)
    print("\nTotal Cost:", df_solution["Cost"].sum())