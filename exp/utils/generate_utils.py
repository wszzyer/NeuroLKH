import os

from feats import get_all_feats, SSSPFeat
from .utils import map_wrapper
from .lkh_utils import *
import subprocess
import tqdm

import numpy as np
from functools import partial

# Note: This function is duplicated intentionally here and is meant to be removed with LKH assimilation.
@map_wrapper
def _solve_LKH_deprecated(task, result_hook, instance_dir, param_dir, log_dir, instance, instance_name, max_candidates=20,
              rerun=False, max_trials=1000, max_nodes=None, candidate_dir=None, candidate=None, candidate2=None, n_nodes_extend=None):
    """
    solve LKH or NeuroLKH or generate candidate set. (if candidate is given.)
    """
    assert task == "FeatGenerate"
    N_NODES = instance["COORD"].__len__() # this will be refactored.
    para_filename = os.path.join(param_dir, instance_name + ".para")
    log_filename = os.path.join(log_dir, instance_name + ".log") if log_dir else None
    instance_filename = os.path.join(instance_dir, instance_name + ".cvrp")
    candidate_type = "nn"
    candidate_filename = os.path.join(candidate_dir, f"{instance_name}_{candidate_type}.txt") if candidate_dir else None
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename, N_NODES)
        write_para(candidate_filename, instance_filename, task, para_filename, max_trials=max_trials, max_candidates=max_candidates, candidate_set_type=candidate_type)
        if candidate is not None:
            write_candidate_dispather[instance["TYPE"]](feat_filename = candidate_filename, candidate = candidate, candidate2 = candidate2, n_nodes_extend = n_nodes_extend)
        f = open(log_filename, "w") if log_filename else subprocess.DEVNULL
        subprocess.check_call(["./LKH", para_filename], stdout=f) 
    return result_hook(candidate_filename, max_nodes, max_candidates)

MAX_EXTRA_NODES_RATIO = 1.15

def make_edge_index(dataset, additional_feats, n_edges, extend=False, max_nodes=None,
                     temp_dir=None, pool=None):
    if extend:
        #This if clause is meant to be refracted without calling LKH.
        feat_param_dir = temp_dir /  "featgen_para"
        feat_dir = temp_dir /  "feat"
        instance_dir = temp_dir / "instance"
        feat_param_dir.mkdir(exist_ok=True)
        feat_dir.mkdir(exist_ok=True)
        instance_dir.mkdir(exist_ok=True)
        feats = tqdm.tqdm(pool.imap(_solve_LKH_deprecated, [("FeatGenerate", read_feat, instance_dir, feat_param_dir, None, dataset[i], str(i), n_edges,
                                                             True, 1, max_nodes, feat_dir) for i in range(len(dataset))]), total=len(dataset), desc='Generating Feat')
        edge_index, n_nodes_extend, _ = list(zip(*feats))
        edge_index = np.concatenate(edge_index, 0)
        node_num = np.array(n_nodes_extend, dtype=np.int32)
    else:
        # TODO: Utilize alpha values to make edge_index
        dist = np.stack([additional_feats[SSSPFeat][instance["GRAPH_INDEX"]] for instance in dataset])
        edge_index = np.argsort(dist, -1)[..., 1:1 + n_edges]
        node_num = np.array([d["SIZE"] for d in dataset])
    # INFO: These can be moved to sgcn codes if we really want to fix them some day.
    # inverse_edge_index = -np.ones(shape=[n_samples, max_nodes, max_nodes], dtype=np.int32)
    # inverse_edge_index[sample_index, edge_index, node_index] = np.arange(N_EDGES).reshape(1, 1, -1) + node_index * N_EDGES
    # inverse_edge_index = inverse_edge_index[sample_index, node_index, edge_index]
    return edge_index, node_num

pad_zero = partial(np.pad, mode="constant", constant_values=0)
def pad_nth(a, n, times):
    padee = a[n:n+1]
    return np.concatenate((a, np.repeat(padee, times, axis=0)), axis=0)
def pad_nth_both(a, n, times):
    itm = pad_nth(a, n, times)
    padee = itm[:, n:n+1]
    return np.concatenate((itm, np.repeat(padee, times, axis=1)), axis=1)

def make_node_feat(dataset, additional_feats, max_nodes):
    demand = np.stack([pad_zero(d["DEMAND"], (0, max_nodes - d["SIZE"])) for d in dataset])
    capacity = np.stack([pad_zero(np.full(d["SIZE"], d["CAPACITY"]), (0, max_nodes - d["SIZE"])) for d in dataset])
    assert dataset[0]["DEPOT"] == 1 #FIXME: Why?
    x = np.stack([pad_nth(d["COORD"], d["DEPOT"], max_nodes - d["SIZE"]) for d in dataset])
    node_feat_list = [x, demand[..., np.newaxis], capacity[..., np.newaxis]]
    if dataset[0]["TYPE"] == "CVRPTW":
        start_end_time = np.stack([d["TIME_WINDOW_SECTION"] for d in dataset])
        node_feat_list += [start_end_time[..., 0:1], start_end_time[..., 1:2]]
    for feat_class in get_all_feats():
        if feat_class.feat_type != "node":
            continue
        feat = np.stack([pad_nth(additional_feats[feat_class][d["GRAPH_INDEX"]], d["DEPOT"], max_nodes - d["SIZE"]) for d in dataset])
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
            chunk = dataset[index:index + chunksize]
            feat = np.stack([pad_nth_both(additional_feats[feat_class][d["GRAPH_INDEX"]], d["DEPOT"], max_nodes - d["SIZE"]) for d in chunk])
            chunk_feat_list.append(feat[sample_index, node_index, edge_index[index:index + chunksize]])
        edge_feat_list.append(np.concatenate(chunk_feat_list, axis=0))
    return np.stack(edge_feat_list, -1)
