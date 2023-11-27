import numpy as np
import torch
from torch.nn import Linear, ReLU, Sigmoid
from torch_geometric.nn import Sequential, SAGEConv
from torch_geometric.nn import global_mean_pool
from omlt.io.torch_geometric import gnn_with_non_fixed_graph
import pyomo.environ as pyo
from omlt import OmltBlock
from omlt.neuralnet import ReluBigMFormulation
from gurobipy import GRB
import sys
import os


num_features = 17
hidden_channels = 16
num_classes = 2

model = Sequential(
    "x, edge_index, batch",
    [
        (SAGEConv(num_features, hidden_channels, aggr="sum"), "x, edge_index -> x"),
        ReLU(inplace=True),
        (SAGEConv(hidden_channels, hidden_channels, aggr="sum"), "x, edge_index -> x"),
        ReLU(inplace=True),
        (global_mean_pool, "x, batch -> x"),
        Linear(hidden_channels, num_classes),
    ],
)

odor = "garlic"
N = int(sys.argv[1])
seed_gnn = int(sys.argv[2])
seed_grb = int(sys.argv[3])

model.load_state_dict(torch.load(f"model/{odor}/GNN_{seed_gnn}.pt"))
# print(model)

# for parameter in model.parameters():
#     print(parameter)


def build_molecule_formulation(m, N):
    m.N = N
    m.Nt = 7
    m.Nn = 4
    m.Nh = 5
    m.F = m.Nt + m.Nn + m.Nh + 1
    m.It = range(m.Nt)
    m.In = range(m.Nt, m.Nt + m.Nn)
    m.Ih = range(m.Nt + m.Nn, m.Nt + m.Nn + m.Nh)
    m.Idb = m.F - 1
    m.fragments = ["C", "N", "S", "O", "*c1ccccc1*", "*c1ccc(*)o1", "*c1ccsc1"]
    m.covalences = [4, 3, 2, 2, 2, 2, 1]

    ub = [None, N // 4, N // 4, N // 4, None, None, None]
    lb = [None, None, None, None, None, None, None]
    ub_ring = 0
    ub_db = N // 4

    m.X = pyo.Var(
        pyo.Set(initialize=range(m.N)),
        pyo.Set(initialize=range(m.F)),
        within=pyo.Binary,
    )
    # m.A = pyo.Var(pyo.Set(initialize=range(m.N)), pyo.Set(initialize=range(m.N)), within=pyo.Binary)
    m.DB = pyo.Var(
        pyo.Set(initialize=range(m.N)),
        pyo.Set(initialize=range(m.N)),
        within=pyo.Binary,
    )

    # constraints
    m.Con = pyo.ConstraintList()

    # (C0): assuming all fragments exist
    # covered by OMLT
    # for v in range(m.N):
    #    m.Con.add((m.A[v,v] == 1))

    # (C1)
    # covered by (C0)
    # m.Con.add((m.A[0,0] == 1))
    # m.Con.add((m.A[1,1] == 1))
    m.Con.add((m.A[0, 1] == 1))

    # (C2):
    # covered by (C0)
    # for v in range(m.N-1):
    #     m.Con.add((m.A[v,v]>=m.A[v+1,v+1]))

    # (C3)
    # covered by OMLT
    # for u in range(m.N):
    #     for v in range(u+1,m.N):
    #         m.Con.add((m.A[u,v] == m.A[v,u]))

    # (C4)
    for v in range(m.N):
        expr = (m.N - 1) * m.A[v, v]
        for u in range(m.N):
            if u != v:
                expr -= m.A[u, v]
        m.Con.add(expr >= 0)

    # (C5)
    for v in range(1, m.N):
        expr = m.A[v, v]
        for u in range(v):
            expr -= m.A[u, v]
        m.Con.add(expr <= 0)

    # (C6)
    for v in range(m.N):
        m.Con.add((m.DB[v, v] == 0))

    # (C7)
    for u in range(m.N):
        for v in range(u + 1, m.N):
            m.Con.add((m.DB[u, v] == m.DB[v, u]))

    # (C8) and (C9) are skipped since there is no triple bond

    # (C10)
    # remove the triple bond term since there is no triple bond
    for u in range(m.N):
        for v in range(u + 1, m.N):
            m.Con.add((m.DB[u, v] <= m.A[u, v]))

    # (C11)
    for v in range(m.N):
        expr = m.A[v, v]
        for f in m.It:
            expr -= m.X[v, f]
        m.Con.add(expr == 0)

    # (C12)
    for v in range(m.N):
        expr = m.A[v, v]
        for f in m.In:
            expr -= m.X[v, f]
        m.Con.add(expr == 0)

    # (C13)
    for v in range(m.N):
        expr = m.A[v, v]
        for f in m.Ih:
            expr -= m.X[v, f]
        m.Con.add(expr == 0)

    # (C14)
    for v in range(m.N):
        expr = 0.0
        for u in range(m.N):
            if u != v:
                expr += m.A[u, v]
        for i in range(m.Nn):
            expr -= (i + 1) * m.X[v, m.In[i]]
        m.Con.add(expr == 0)

    # (C15)
    for u in range(m.N):
        for v in range(u + 1, m.N):
            m.Con.add(
                (3.0 * m.DB[u, v] - m.X[u, m.Idb] - m.X[v, m.Idb] - m.A[u, v] <= 0)
            )

    # (C16) is skipped since there is no triple bond

    # (C17)
    for v in range(m.N):
        expr = 0.0
        for u in range(m.N):
            if u != v:
                expr += m.DB[u, v]
        for i in range(m.Nt):
            expr -= (m.covalences[i] // 2) * m.X[v, m.It[i]]
        m.Con.add(expr <= 0)

    # (C18) is skipped since there is no triple bond

    # (C19)
    for v in range(m.N):
        expr = m.X[v, m.Idb]
        for u in range(m.N):
            if u != v:
                expr -= m.DB[u, v]
        # m.Con.add(expr <= 0)
        # restrict to equations to avoid allenes
        m.Con.add(expr == 0)

    # (C20) is skipped since there is no triple bond

    # (C21)
    # remove the triple bond term since there is no triple bond
    for v in range(m.N):
        expr = 0.0
        for i in range(m.Nt):
            expr += m.covalences[i] * m.X[v, m.It[i]]
        for i in range(m.Nn):
            expr -= (i + 1) * m.X[v, m.In[i]]
        for i in range(m.Nh):
            expr -= i * m.X[v, m.Ih[i]]
        for u in range(m.N):
            if u != v:
                expr -= m.DB[u, v]
        m.Con.add(expr == 0)

    # (C22)
    for i in range(m.Nt):
        expr = 0.0
        for v in range(m.N):
            expr += m.X[v, m.It[i]]
        if lb[i] is not None:
            m.Con.add(expr >= lb[i])
        if ub[i] is not None:
            m.Con.add(expr <= ub[i])

    # (C23)
    if ub_db is not None:
        expr = 0.0
        for u in range(m.N):
            for v in range(u + 1, m.N):
                expr += m.DB[u, v]
        m.Con.add(expr <= ub_db)

    # (C24) is skipped since there is no triple bond

    # (C25)
    if ub_ring is not None:
        expr = -(m.N - 1)
        for u in range(m.N):
            for v in range(u + 1, m.N):
                expr += m.A[u, v]
        m.Con.add(expr <= ub_ring)

    # (C26)
    coef_1 = [2**i for i in range(m.F - 1, -1, -1)]
    for v in range(1, m.N):
        expr = 0.0
        for f in range(m.F):
            expr += coef_1[f] * m.X[0, f]
        for f in range(m.F):
            expr -= coef_1[f] * m.X[v, f]
        expr -= (2**m.F) * (1.0 - m.A[v, v])
        m.Con.add(expr <= 0)

    # (C27)
    coef_2 = [2**i for i in range(m.N - 1, -1, -1)]
    for v in range(1, m.N - 1):
        expr = 0.0
        for u in range(m.N):
            if u != v and u != v + 1:
                expr += coef_2[u] * m.A[u, v]
        for u in range(m.N):
            if u != v and u != v + 1:
                expr -= coef_2[u] * m.A[u, v + 1]
        m.Con.add(expr >= 0)

    # (G1)
    for u in range(m.N):
        for v in range(m.N):
            for w in range(m.N):
                if u == v or u == w or v == w:
                    continue
                expr = (
                    (m.A[u, v] - m.DB[u, v])
                    + (m.A[v, w] - m.DB[v, w])
                    + (m.X[v, 0] + m.X[v, 1] + m.X[v, 2] + m.X[v, 3])
                    + (m.X[u, 1] + m.X[u, 2] + m.X[u, 3])
                    + (m.X[w, 1] + m.X[w, 2] + m.X[w, 3])
                )
                m.Con.add(expr <= 4)

    # (G2)
    for u in range(m.N):
        for v in range(m.N):
            if u == v:
                continue
            expr = m.X[v, 2] + (m.X[u, 1] + m.X[u, 2] + m.X[u, 3]) + m.A[u, v]
            m.Con.add(expr <= 2)

    # (G3)
    for v in range(m.N):
        expr = m.X[v, 4] + m.X[v, 5] + m.X[v, 16]
        m.Con.add(expr <= 1)

    # (G4)
    for u in range(m.N):
        for v in range(u + 1, m.N):
            expr = m.X[u, 3] + m.X[v, 3] + m.A[u, v]
            m.Con.add(expr <= 2)

    # (G5)
    expr = 0.0
    for v in range(m.N):
        expr += m.X[v, 4] + m.X[v, 5] + m.X[v, 6]
    m.Con.add(expr <= 2)


lb = np.zeros(N * num_features)
ub = np.ones(N * num_features)
input_bounds = [(l, u) for l, u in zip(lb, ub)]

m = pyo.ConcreteModel()
m.nn = OmltBlock()

gnn_with_non_fixed_graph(m.nn, model, N, scaled_input_bounds=input_bounds)

build_molecule_formulation(m.nn, N)

# connect features to inputs of OmltBlock
m.Con_Input = pyo.ConstraintList()
for v in range(m.nn.N):
    for f in range(m.nn.F):
        m.Con_Input.add((m.nn.X[v, f] == m.nn.inputs[v * m.nn.F + f]))
# m.Con_Input.pprint()


m.obj = pyo.Objective(expr=(m.nn.outputs[1] - m.nn.outputs[0]), sense=pyo.maximize)


opt = pyo.SolverFactory("gurobi_persistent")
opt.set_instance(m)
opt.set_gurobi_param("Seed", seed_grb)
opt.set_gurobi_param("TimeLimit", 36000)
# opt.set_gurobi_param("MIPFocus", 1)

cb_times = []
cb_sols = []


def my_callback(cb_m, cb_opt, cb_where):
    if cb_where == GRB.Callback.MIPSOL:
        cb_times.append(cb_opt.cbGet(GRB.Callback.RUNTIME))
        cb_sols.append(cb_opt.cbGet(GRB.Callback.MIPSOL_OBJ))


opt.set_callback(my_callback)

result = opt.solve(tee=False)

# print(cb_times)
# print(cb_sols)
for i in range(len(cb_times)):
    if np.abs(cb_sols[i] - cb_sols[-1]) < 1e-9:
        opt_time = cb_times[i]
        break

# print(opt_time)
# print(result.Solver.Wallclock_time)
# print(result.Problem.Upper_bound)
# print(result.Problem.Lower_bound)

result_saved = np.array(
    [
        opt_time,
        result.Solver.Wallclock_time,
        result.Problem.Upper_bound,
        result.Problem.Lower_bound,
    ]
)

X = np.zeros((m.nn.N, m.nn.F))
A = np.zeros((m.nn.N, m.nn.N))
DB = np.zeros((m.nn.N, m.nn.N))
for u in range(m.nn.N):
    for v in range(m.nn.N):
        A[u, v] = int(m.nn.A[u, v].value)
        DB[u, v] = int(m.nn.DB[u, v].value)
for v in range(m.nn.N):
    for f in range(m.nn.F):
        X[v, f] = int(m.nn.X[v, f].value)

folder = f"results/optimality/{odor}/N={N}/run_{seed_gnn}_{seed_grb}/"
os.makedirs(folder, exist_ok=True)

np.save(folder + "ans", result_saved)
np.save(folder + "X", X)
np.save(folder + "A", A)
np.save(folder + "DB", DB)

print("done!")
