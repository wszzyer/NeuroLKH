import numpy as np
from itertools import combinations, product
import vrplib
from typing import List, Tuple

# import gurobipy as gp
# from gurobipy import GRB
import coptpy as cp
from coptpy import COPT

# Yes this is adopted from Gurobi too. Poor shanshu.
class TSPCallback:

    def __init__(self, x, depot):
        self.x = x
        self.depot = depot

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                self.eliminate_subtours(model)
            except Exception:
                model.terminate()
    
    def remove_depot_cycle(self, edges:List[Tuple[float, float]]):
        current = self.depot
        while True:
            cursor = None
            for edge in edges:
                if edge[0] == current:
                    cursor = edge
                    break
            if cursor is None:
                breakpoint()
                raise RuntimeError
            current = cursor[1]
            edges.remove(cursor)
            if current == self.depot:
                break

    def eliminate_subtours(self, model):
        values = model.cbGetSolution(self.x)
        for index, conn_mat in enumerate(values):
            edges = []
            for i, conn_line in enumerate(conn_mat):
                for j, conn in enumerate(conn_line):
                    if conn > 0.5:
                        edges.append((i, j))
            self.remove_depot_cycle(edges)
            if edges:
                if len(edges) < 12:
                    model.cbLazy(
                        gp.quicksum(self.x[index, i, j] for i, j in combinations(map(lambda x: x[0], edges), 2))
                        <= len(edges) - 1
                    )
                else:
                    model.cbLazy(
                        gp.quicksum(self.x[index, i, j] for i, j in edges)
                        <= len(edges) - 1
                    )


def solve_cvrp(instance_file):
    instance = vrplib.read_instance(instance_file)
    dimension = int(instance['dimension'])
    capacity  = int(instance['capacity'])
    depot     = int(instance['depot'])
    demand    = instance['demand'].astype(np.int32)
    edge_weights = instance['edge_weight'].astype(np.int64) # Truncate
    vehicle_count = int(np.ceil(demand.sum() / capacity))

    normal_nodes = np.concatenate((np.arange(depot), np.arange(depot + 1, dimension)))

    env = cp.Envr()
    m = env.createModel('copt_test')
    
    # Vars
    x = m.addMVar(shape=(vehicle_count, dimension, dimension), vtype=COPT.BINARY)
    # If we use MTK formulations we need to set up a new variable u for each node except for depot
    u = m.addMVar(shape=(dimension - 1, ), vtype=COPT.CONTINUOUS)

    # Constriants
    # Vehicles leaves node that it enters
    m.addConstr(x.sum(axis=1) == x.sum(axis=2))
    # Ensure that every node is entered once
    in_degrees = x.sum(axis=0).sum(axis=0)
    m.addConstrs(in_degrees[i] == 1 for i in normal_nodes)
    # Every vehicle leaves the depot
    m.addConstrs(x[vehicle][depot].sum() == 1 for vehicle in range(vehicle_count))
    # Capacity Constraint, which is included in MTZ I guess
    # m.addConstrs((x[vehicle].sum(axis=0) * demand).sum() <= capacity for vehicle in range(vehicle_count))
    # simple diag constraint
    m.addConstrs(x[vehicle][i][i] == 0 for (vehicle, i) in product(range(vehicle_count), range(dimension)))
    # If we use lazy constraints on large CVRP problems the program can run for centuries
    # So let's try Miller-Tucker-Zemlin Formulation instead.
    # m.Params.LazyConstraints = 1
    # cb = TSPCallback(x, depot)
    # m.optimize(cb)
    # AND HERE goes MTZ Formulations.
    # For every node which is not depot, it accumlate u according to the route passing it, 
    for vehicle_id in range(vehicle_count):
        m.addConstrs(u.reshape((1, -1)) - u.reshape((-1, 1)) >= demand[normal_nodes].reshape((-1, 1)) - capacity * (1 - x[vehicle_id][normal_nodes][:, normal_nodes]))
    # Every city is satisfied
    m.addConstr(u >= demand[normal_nodes])
    # And no vehicle is overloaded.
    m.addConstr(u <= capacity)

    # Objective.
    m.setObjective((x * edge_weights).sum(), sense=COPT.MINIMIZE)
    m.setParam(COPT.Param.TimeLimit, 1200)
    m.solve()
    breakpoint()
    return m.ObjVal
