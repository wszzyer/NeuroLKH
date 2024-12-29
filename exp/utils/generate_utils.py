from feats import get_all_feats, SSSPFeat
from .lkh_utils import *
import tqdm

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

def make_edge_index(dataset, additional_feats, n_edges, extend=False, max_nodes=None, node_num=None):
    graph_dist = additional_feats[SSSPFeat]
    if extend:
        dist_mat_list = []
        for instance, extended_size in tqdm.tqdm(zip(dataset, node_num), total=len(dataset), desc="Selecting Edges"):
            # LKH Exodus Note: Pad extended nodes as Depot and add "Special" nodes. 
            # See "An Improved Transformation of the Symmetric Multiple Traveling Salesman Problem" for details.
            # See MTSP2TSP.c:76 (especially Forbidden.c:31) for LKH implementation.
            depot = instance["DEPOT"] - 1
            dist_mat = graph_dist[instance["GRAPH_INDEX"]]
            # Set diagonal line to inf. If not, we have to set every pair of depots manually.
            dist_mat[np.arange(instance["SIZE"]), np.arange(instance["SIZE"])] = np.inf
            all_depot_count = extended_size - instance["SIZE"] + 1
            special_nodes = np.argsort(dist_mat[depot])[1:all_depot_count + 1]
            special_dists = dist_mat[depot][special_nodes].copy()
            dist_mat[depot][special_nodes] = np.inf
            dist_mat[special_nodes, depot] = np.inf
            dist_mat = pad_nth_both(dist_mat, depot, extended_size - instance["SIZE"])
            for id, depot_index in enumerate(chain([depot], range(instance["SIZE"], extended_size))):
                for special_index in (id % all_depot_count, (id + 1) % all_depot_count):
                    dist_mat[depot_index][special_nodes[special_index]] = special_dists[special_index]
                    dist_mat[special_nodes[special_index]][depot_index] = special_dists[special_index]
            dist_mat_list.append(np.pad(dist_mat, (0, max_nodes - extended_size), mode="constant", constant_values=np.inf))
        dist = np.stack(dist_mat_list)
        edge_index = np.argsort(dist, -1)[..., :n_edges]
    else:
        dist = np.stack([np.pad(graph_dist[instance["GRAPH_INDEX"]],
                                (0, max_nodes - instance["SIZE"]),
                                mode="constant",
                                constant_values=np.inf
                                ) for instance in dataset])
        # The index start from 1 since the diagonal line of `dist` is 0 here.
        # TODO: Utilize alpha values to make edge_index
        edge_index = np.argsort(dist, -1)[..., 1:1 + n_edges]
    # INFO: These can be moved to sgcn codes if we really want to fix them some day.
    # inverse_edge_index = -np.ones(shape=[n_samples, max_nodes, max_nodes], dtype=np.int32)
    # inverse_edge_index[sample_index, edge_index, node_index] = np.arange(N_EDGES).reshape(1, 1, -1) + node_index * N_EDGES
    # inverse_edge_index = inverse_edge_index[sample_index, node_index, edge_index]
    return edge_index

def make_node_feat(dataset, additional_feats, max_nodes):
    demand = np.stack([pad_zero(d["DEMAND"], (0, max_nodes - d["SIZE"])) for d in dataset])
    capacity = np.stack([pad_zero(np.full(d["SIZE"], d["CAPACITY"]), (0, max_nodes - d["SIZE"])) for d in dataset])
    x = np.stack([pad_nth(d["COORD"], d["DEPOT"] - 1, max_nodes - d["SIZE"]) for d in dataset])
    node_feat_list = [x, demand[..., np.newaxis], capacity[..., np.newaxis]]
    if dataset[0]["TYPE"] == "CVRPTW":
        start_end_time = np.stack([d["TIME_WINDOW_SECTION"] for d in dataset])
        node_feat_list += [start_end_time[..., 0:1], start_end_time[..., 1:2]]
    for feat_class in get_all_feats():
        if feat_class.feat_type != "node":
            continue
        feat = np.stack([pad_nth(additional_feats[feat_class][d["GRAPH_INDEX"]], d["DEPOT"] - 1, max_nodes - d["SIZE"]) for d in dataset])
        node_feat_list.append(feat)
    return np.concatenate(node_feat_list, -1)

def make_edge_feat(dataset, additional_feats, max_nodes, edge_index, chunksize=64):
    chunksize = min(chunksize, len(dataset))
    sample_index, node_index = np.ogrid[:chunksize, :max_nodes, :1][:-1]

    edge_feat_list = []
    for feat_class in get_all_feats():
        if feat_class.feat_type != "edge":
            continue
        # If no feat is generated for those extended nodes, set them the same as node 0.
        # This feat mat can be tremendously large so we make chunks here.
        chunk_feat_list = []
        for index in tqdm.trange(0, len(dataset), chunksize, desc="Making edge feature"):
            current_chunksize = min(chunksize, len(dataset) - index)
            chunk = dataset[index:index + current_chunksize]
            feat = np.stack([pad_nth_both(additional_feats[feat_class][d["GRAPH_INDEX"]], d["DEPOT"] - 1, max_nodes - d["SIZE"]) for d in chunk])
            chunk_feat_list.append(feat[sample_index[:current_chunksize], node_index, edge_index[index:index + current_chunksize]])
        edge_feat_list.append(np.concatenate(chunk_feat_list, axis=0))
    return np.stack(edge_feat_list, -1)
