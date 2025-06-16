import numpy as np
from pathlib import Path
import vrplib

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def print_solution(manager, routing, solution, vehicle_count):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    max_route_distance = 0
    total_distance = 0
    for vehicle_id in range(vehicle_count):
        if not routing.IsVehicleUsed(solution, vehicle_id):
            continue
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += f" {manager.IndexToNode(index)} -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f"{manager.IndexToNode(index)}\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
        total_distance += route_distance
    print(f"Maximum of the route distances: {max_route_distance}m")
    print(f"Total route distances: {total_distance}m")

def try_solve_cvrp(instance, search_parameters, padding):
    dimension = int(instance['dimension'])
    capacity  = int(instance['capacity'])
    depot     = int(instance['depot'])
    demand    = instance['demand'].astype(np.int32)
    edge_weights = instance['edge_weight'].round().astype(np.int64) # Truncate
    vehicle_count = int(np.ceil(demand.sum() / capacity))

    manager = pywrapcp.RoutingIndexManager(dimension, vehicle_count, depot)
    routing = pywrapcp.RoutingModel(manager)

    # Register distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return edge_weights[from_node][to_node]
    transition_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transition_callback_index)

    # Register dimension
    dimension_name = "Distance"
    routing.AddDimension(
        transition_callback_index,
        0,  # no slack
        int(edge_weights.mean() * dimension / vehicle_count * 4),  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Register demands
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demand[from_node]
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        np.full(vehicle_count, capacity),  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )


    # Register collector
    solver = routing.solver()
    collector = solver.AllSolutionCollector()
    routing.CloseModelWithParameters(search_parameters)
    collector.AddObjective(routing.CostVar())
    routing.AddSearchMonitor(collector)

    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        solution_count = collector.SolutionCount()
        time = [collector.WallTime(i) for i in range(solution_count)]
        obj = [collector.ObjectiveValue(i) for i in range(solution_count)]
        result = np.pad(np.array([obj, time]), ((0, 0), (0, padding - solution_count)), 'edge')
    else:
        result = None
    return result


def solve_cvrp(instance: Path | dict):
    if not type(instance) is dict:
        instance = vrplib.read_instance(instance)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(6000)
    search_parameters.solution_limit = 5000

    result = try_solve_cvrp(instance, search_parameters, padding=search_parameters.solution_limit)
    if not result is None:
        return result
    # Use another first solution startegy and try again...?
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES
    result = try_solve_cvrp(instance, search_parameters, padding=search_parameters.solution_limit)
    if not result is None:
        return result
    else:
        return np.full((2, search_parameters.solution_limit), np.inf)
