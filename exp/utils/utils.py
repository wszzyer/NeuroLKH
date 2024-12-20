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
        return 2 # x, y
    elif problem == "cvrp":
        return 4 # x, y, demand, capacity
    elif problem == "pdp":
        return 5 # x, y, depot, pickup, delivery
    elif problem == "cvrptw":
        return 6 # x, y, demand, start_time, end_time, capacity
    else:
        raise RuntimeError(f"Fail to recognize problem type {problem}")

def make_mask(method, 
              nn:Optional[torch.Tensor]=None, # Sparse one-hot k-nearest neighbors
              dist_mat:Optional[torch.Tensor]=None, # Dense Shortest Path Matrix
              alpha_values:Optional[torch.Tensor]=None, # Sparse Alpha values
              fix_n:Optional[int]=None,
              dist_threshold:Optional[float]=None):

    nn_flag = (method == 'nn' or method == 'mixed')
    alpha_flag = (method == 'alpha' or method == 'mixed')

    # I have to admit that this implementation is a little bit tricky
    # We use topk to implement priority, alpha > dist > nn
    accumulation_mat = None

    if alpha_flag:
        if alpha_values is None:
            raise RuntimeError('Using alpha method to make mask while alpha is None')
        accumulation_mat = alpha_values.max() - alpha_values
    
    if nn_flag:
        if dist_mat is not None:
            if accumulation_mat is None:
                accumulation_mat = dist_mat
            else:
                if dist_threshold is not None:
                    dist_mat = torch.where(dist_mat > dist_threshold, dist_mat - dist_threshold, 0)
                max_round_dist = np.ceil(dist_mat.max())
                accumulation_mat = torch.where(accumulation_mat > 0, accumulation_mat + max_round_dist, 0) + dist_mat
        elif nn is not None:
            if accumulation_mat is None:
                accumulation_mat = nn
            else:
                accumulation_mat = torch.where(accumulation_mat > 0, accumulation_mat + 10, 0) + nn
        else:
            raise RuntimeError('Using nn method to make mask while nn and dist is both None')

    if fix_n:
        return torch.topk(accumulation_mat, fix_n, sorted=False) # The indices can help further calculation
    else:
        return accumulation_mat > 0
