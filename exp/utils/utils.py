import numpy as np
import networkx as nx
import functools
import torch
from typing import Optional

def map_wrapper(func):
    @functools.wraps(func)
    def expand_args_for_func(args):
        return func(*args)
    return expand_args_for_func

def smooth_matrix(adjs, mat: np.ndarray):
    # mat[u][v] will be aggregated by u's neighbours and v's neighbours.    
    mat_smoothed = mat.copy()
    inverse_adjs = [[] for _ in range(len(adjs))]
    for i in range(len(adjs)):
        for j in adjs[i]:
            inverse_adjs[j].append(i)
    
    for i in range(mat.shape[0]):
        mat_smoothed[i] += mat[adjs[i]].sum()
    for j in range(mat.shape[1]):
        mat_smoothed[:, j] += mat[:, inverse_adjs[j]].sum()
        
    return mat_smoothed

def get_problem_default_node_feat_dim(problem: str) -> int:
    if problem == "tsp":
        return 3 # x, y, nodeid
    elif problem == "cvrp":
        return 5 # x, y, nodeid, demand, capacity
    elif problem == "pdp":
        return 6 # x, y, nodeid, depot, pickup, delivery
    elif problem == "cvrptw":
        return 7 # x, y, nodeid, demand, start_time, end_time, capacity
    else:
        raise RuntimeError(f"Fail to recognize problem type {problem}")
