from feats import get_all_feats, SSSPFeat
from .lkh_utils import *
import tqdm
from .alpha_utils import get_alpha

import numpy as np
from functools import partial
from itertools import chain

# Helpers
pad_zero = partial(np.pad, mode="constant", constant_values=0)
def pad_nth(a, n, times):
    padee = a[n:n+1]
    return np.concatenate((a, np.repeat(padee, times, axis=0)), axis=0)
def pad_nth_both(a, n, times):
    itm = pad_nth(a, n, times)
    padee = itm[:, n:n+1]
    return np.concatenate((itm, np.repeat(padee, times, axis=1)), axis=1)

def make_node_feat(dataset, additional_feats, max_nodes):
    graph_id = np.stack([pad_zero(d["GRAPH_INDEX"] + 1, (0, max_nodes - d["SIZE"])) for d in dataset])
    demand = np.stack([pad_zero(d["DEMAND"], (0, max_nodes - d["SIZE"])) for d in dataset])
    capacity = np.stack([pad_zero(np.full(d["SIZE"], d["CAPACITY"]), (0, max_nodes - d["SIZE"])) for d in dataset])
    x = np.stack([pad_nth(d["COORD"], d["DEPOT"] - 1, max_nodes - d["SIZE"]) for d in dataset])
    node_feat_list = [x, graph_id[..., np.newaxis], demand[..., np.newaxis], capacity[..., np.newaxis]]
    if dataset[0]["TYPE"] == "CVRPTW":
        start_end_time = np.stack([d["TIME_WINDOW_SECTION"] for d in dataset])
        node_feat_list += [start_end_time[..., 0:1], start_end_time[..., 1:2]]
    for feat_class in get_all_feats():
        if feat_class.feat_type != "node":
            continue
        feat = np.stack([pad_nth(additional_feats[feat_class][d["GRAPH_INDEX"]], d["DEPOT"] - 1, max_nodes - d["SIZE"]) for d in dataset])
        node_feat_list.append(feat)
    return np.concatenate(node_feat_list, -1)

def make_edge_feat(dataset, additional_feats, max_nodes, n_edges, extend=False, node_num=None, chunksize=64, pool=None):
    # 1. Make weights.
    dist_mat_list = []
    if extend:
        for instance, extended_size in tqdm.tqdm(zip(dataset, node_num), total=len(dataset), desc="Selecting Edges"):
            # LKH Exodus Note: Pad extended nodes as Depot and add "Special" nodes. 
            # See "An Improved Transformation of the Symmetric Multiple Traveling Salesman Problem" for details.
            # See MTSP2TSP.c:76 (especially Forbidden.c:31) for LKH implementation.
            depot = instance["DEPOT"] - 1
            dist_mat = instance["WEIGHT"]
            # Set diagonal line to inf. If not, we have to set every pair of depots manually.
            dist_mat[np.arange(instance["SIZE"]), np.arange(instance["SIZE"])] = np.inf
            all_depot_count = extended_size - instance["SIZE"] + 1
            special_nodes = instance["SPECIAL"] - 1
            special_dists = dist_mat[depot][special_nodes].copy()
            dist_mat[depot][special_nodes] = np.inf
            dist_mat[special_nodes, depot] = np.inf
            dist_mat = pad_nth_both(dist_mat, depot, extended_size - instance["SIZE"])
            for id, depot_index in enumerate(chain([depot], range(instance["SIZE"], extended_size))):
                for special_index in (id % all_depot_count, (id + 1) % all_depot_count):
                    dist_mat[depot_index][special_nodes[special_index]] = special_dists[special_index]
                    dist_mat[special_nodes[special_index]][depot_index] = special_dists[special_index]
            dist_mat_list.append(np.pad(dist_mat, (0, max_nodes - extended_size), mode="constant", constant_values=np.inf))
        shift = 0
        sizes = node_num
    else:
        dist_mat_list.extend([np.pad(instance["WEIGHT"],
                                (0, max_nodes - instance["SIZE"]),
                                mode="constant",
                                constant_values=np.inf
                                ) for instance in dataset])
        shift = 1
        sizes = [instance["SIZE"] for instance in dataset]
    # 2. Make edge index. This step may be costy and can be further optimized.
    @map_wrapper
    def make_edge_indice(dist_mat, size):
        nn_indice = np.argsort(dist_mat, -1)
        alpha_values, alpha_indice = get_alpha(dist_mat[:size][:, :size], n_edges, not extend)
        for nn_index, alpha_value, alpha_index in zip(nn_indice, alpha_values, alpha_indice):
            legal_len = n_edges - np.isinf(alpha_value).sum()
            legal_part = alpha_index[:legal_len]
            current_nn = shift
            for fill_index in range(legal_len, n_edges):
                while np.any(legal_part == nn_index[current_nn]):
                    current_nn += 1
                alpha_index[fill_index] = nn_index[current_nn]
                current_nn += 1
        return pad_zero(alpha_indice, ((0, dist_mat.shape[0] - size), (0, 0))), alpha_values
    _map = partial(pool.imap, chunksize=chunksize) if pool else map
    edge_index, alpha_values = zip(*tqdm.tqdm(_map(make_edge_indice, zip(dist_mat_list, sizes)), total=len(dataset), desc="Making Edge Index"))
    edge_index = np.stack(edge_index)

    # 3. Construct feats.
    chunksize = min(chunksize, len(dataset))
    sample_index, node_index = np.ogrid[:chunksize, :max_nodes, :1][:-1]
    edge_feat_list = []
    for feat_class in get_all_feats():
        if feat_class.feat_type != "edge":
            continue
        # This feat mat can be tremendously large so we make chunks here.
        chunk_feat_list = []
        for index in tqdm.trange(0, len(dataset), chunksize, desc="Making Edge Feature"):
            current_chunksize = min(chunksize, len(dataset) - index)
            slice_index = slice(index, index + chunksize)
            # We have to do something special for CVRP problem on weight, and we hope that the model can make use of this too.
            if feat_class is SSSPFeat:
                feat = np.stack(dist_mat_list[slice_index])
                feat = np.where(np.isinf(feat), 0, feat)
            else:
                feat = np.stack([pad_nth_both(additional_feats[feat_class][d["GRAPH_INDEX"]], d["DEPOT"] - 1, max_nodes - d["SIZE"]) for d in dataset[slice_index]])
            chunk_feat_list.append(feat[sample_index[:current_chunksize], node_index, edge_index[slice_index]])
        edge_feat_list.append(np.concatenate(chunk_feat_list, axis=0))
    return np.stack(edge_feat_list, -1), edge_index
