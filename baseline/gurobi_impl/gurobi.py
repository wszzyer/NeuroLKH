import numpy as np
from itertools import combinations, product
import vrplib
from typing import List, Tuple
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

# Yes this is adopted from Gurobi
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

def solve_cvrp(instance: Path | dict, first_solution=None):
    if not type(instance) is dict:
        instance = vrplib.read_instance(instance)
    dimension = int(instance['dimension'])
    capacity  = int(instance['capacity'])
    depot     = int(instance['depot'])
    demand    = instance['demand'].astype(np.int32)
    edge_weights = instance['edge_weight'].astype(np.int64) # Truncate
    vehicle_count = int(np.ceil(demand.sum() / capacity))

    normal_nodes = np.concatenate((np.arange(depot), np.arange(depot + 1, dimension)))
    
    with gp.Env() as env, gp.Model(env=env) as m:
        # Vars
        x = m.addMVar(shape=(vehicle_count, dimension, dimension), vtype=GRB.BINARY, name='x')
        # If we use MTK formulations we need to set up a new variable u for each node except for depot
        u = m.addMVar(shape=(dimension - 1, ), vtype=GRB.CONTINUOUS, name='u')
        if not first_solution is None:
            x_init = np.zeros(shape=(vehicle_count, dimension, dimension), dtype=np.bool_)
            u_init = np.zeros(shape=(dimension - 1, ), dtype=np.float64)
            last_node = first_solution[0]
            assert last_node == depot
            vehicle_number = 0
            goods = 0
            for node in np.roll(first_solution, -1):
                if node >= dimension:
                    node = depot
                x_init[vehicle_number, last_node, node] = True
                if node == depot:
                    vehicle_number += 1
                    goods = 0
                else:
                    goods += demand[node]
                    u_init[int(node - 1)] = goods
                last_node = node
            # u.Start = u_init
            # x.Start = x_init
        # Constriants
        # Vehicles leaves node that it enters
        m.addConstr(x.sum(axis=1) == x.sum(axis=2))
        assert np.all(x_init.sum(axis=1) == x_init.sum(axis=2))
        # Ensure that every node is entered once
        in_degrees = x.sum(axis=0).sum(axis=0)
        m.addConstrs(in_degrees[i] == 1 for i in normal_nodes)
        assert np.all(x_init.sum(axis=0).sum(axis=0)[normal_nodes] == 1)
        # Every vehicle leaves the depot
        m.addConstrs(x[vehicle][depot].sum() == 1 for vehicle in range(vehicle_count))
        assert np.all(x_init[:, depot, :].sum(axis=1) == 1)
        # Capacity Constraint, which is included in MTZ I guess
        # m.addConstrs((x[vehicle].sum(axis=0) * demand).sum() <= capacity for vehicle in range(vehicle_count))
        # simple diag constraint
        m.addConstrs(x[vehicle][i][i] == 0 for (vehicle, i) in product(range(vehicle_count), range(dimension)))
        assert np.all(x_init[:, np.arange(dimension), np.arange(dimension)] == 0)
        # If we use lazy constraints on large CVRP problems the program can run for centuries
        # So let's try Miller-Tucker-Zemlin Formulation instead.
        # m.Params.LazyConstraints = 1
        # cb = TSPCallback(x, depot)
        # m.optimize(cb)
        # AND HERE goes MTZ Formulations.
        # For every node which is not depot, it accumlate u according to the route passing it, 
        m.addConstrs(u.reshape(-1, 1) - u.reshape(1, -1) >= demand[normal_nodes].reshape(-1, 1) - capacity * (1 - x[vehicle_id][normal_nodes][:, normal_nodes]) for vehicle_id in range(vehicle_count))
        for vehicle_id in range(vehicle_count):
            # sugei = u_init.reshape(1, -1) - u_init.reshape(-1, 1) >= demand[normal_nodes].reshape(1, -1) - capacity * (1 - x_init[vehicle_id][normal_nodes][:, normal_nodes])
            assert np.all(u_init.reshape(1, -1) - u_init.reshape(-1, 1) >= demand[normal_nodes].reshape(1, -1) - capacity * (1 - x_init[vehicle_id][normal_nodes][:, normal_nodes]))
        # Every city is satisfied
        m.addConstr(u >= demand[normal_nodes])
        assert np.all(u_init >= demand[normal_nodes])
        # And no vehicle is overloaded.
        m.addConstr(u <= capacity)
        assert np.all(u_init <= capacity)

        # Objective.
        m.setObjective((x * edge_weights).sum(), GRB.MINIMIZE)
        m.optimize()
        # breakpoint()
        return m.ObjVal
