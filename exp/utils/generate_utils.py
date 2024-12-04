import os
from .utils import map_wrapper
from .lkh_utils import *
import subprocess
import tqdm

import numpy as np
from feats import get_all_feats

# Note: This function is duplicated intentionally here and is meant to be removed with LKH assimilation.
@map_wrapper
def solve_LKH(task, result_hook, instance_dir, param_dir, log_dir, instance, instance_name, max_candidate=20,
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
        write_para(candidate_filename, instance_filename, task, para_filename, max_trials=max_trials, max_candidate=max_candidate, candidate_set_type=candidate_type)
        if candidate is not None:
            write_candidate_dispather[instance["TYPE"]](feat_filename = candidate_filename, candidate = candidate, candidate2 = candidate2, n_nodes_extend = n_nodes_extend)
        f = open(log_filename, "w") if log_filename else subprocess.DEVNULL
        subprocess.check_call(["./LKH", para_filename], stdout=f) 
    return result_hook(candidate_filename, max_nodes)

def make_edge_index(dataset, n_nodes, n_edges, extend=False, max_nodes=None,
                     temp_dir=None, pool=None):
    if extend:
        #This if clause is meant to be refracted without calling LKH.
        feat_param_dir = temp_dir /  "featgen_para"
        feat_dir = temp_dir /  "feat"
        instance_dir = temp_dir / "instance"
        feat_param_dir.mkdir(exist_ok=True)
        feat_dir.mkdir(exist_ok=True)
        feats = tqdm.tqdm(pool.imap(solve_LKH, [("FeatGenerate", read_feat, instance_dir, feat_param_dir, None, dataset[i], str(i), True, 1, max_nodes, feat_dir) for i in range(len(dataset))]), total=len(dataset), desc='Generating Feat')
        edge_index, n_nodes_extend, _ = list(zip(*feats))
        edge_index = np.concatenate(edge_index, 0)
        node_num = np.array(n_nodes_extend, dtype=np.int32)
    else:
        # TODO: Utilize alpha values to make edge_index
        from feats import SSSPFeat
        dist = np.stack([instance["ATTACHMENT"][SSSPFeat] for instance in dataset])
        edge_index = np.argsort(dist, -1)[..., 1:1 + n_edges]
        node_num = np.full((edge_index.shape[0], ), n_nodes, dtype=np.int32)
    # INFO: These can be moved to sgcn codes if we really want to fix then some day.
    # inverse_edge_index = -np.ones(shape=[n_samples, max_nodes, max_nodes], dtype=np.int32)
    # inverse_edge_index[sample_index, edge_index, node_index] = np.arange(N_EDGES).reshape(1, 1, -1) + node_index * N_EDGES
    # inverse_edge_index = inverse_edge_index[sample_index, node_index, edge_index]
    return edge_index, node_num

def make_node_feat(dataset, n_nodes, max_nodes):
    demand = np.stack([d["DEMAND"] for d in dataset])
    demand = np.pad(demand, [(0, 0), (0, max_nodes - n_nodes)], 'constant', constant_values=0)
    capacity = np.zeros([len(dataset), max_nodes], dtype=int)
    capacity[:, 0] = np.array([d["CAPACITY"] for d in dataset])
    capacity[:, n_nodes:] = capacity[:, :1]
    assert dataset[0]["DEPOT"] == 1
    x = np.stack([d["COORD"] for d in dataset])
    x = np.concatenate([x, x[:, :1, :].repeat(max_nodes - n_nodes, axis=1)], 1)
    node_feat_list = [x, demand[:, :, None], capacity[:, :, None]]
    if dataset[0]["TYPE"] == "CVRPTW":
        start_end_time = np.stack([d["TIME_WINDOW_SECTION"] for d in dataset])
        node_feat_list += [start_end_time[..., 0:1], start_end_time[..., 1:2]]
    for feat_class in get_all_feats():
        if feat_class.feat_type != "node":
            continue
        feat = np.stack([instance["ATTACHMENT"][feat_class] for instance in dataset])
        # Pad with node_0 by default.
        node_feat_list.append(np.concatenate([feat, feat[:, :1, :].repeat(max_nodes - n_nodes, axis=1)], axis=1))
    return np.concatenate(node_feat_list, -1)

def make_edge_feat(dataset, max_nodes, edge_index):
    sample_index, node_index = np.ogrid[:len(dataset), :max_nodes, :1][:-1]

    edge_feat_list = []
    for feat_class in get_all_feats():
        if feat_class.feat_type != "edge":
            continue
        feat = np.stack([instance["ATTACHMENT"][feat_class] for instance in dataset])
        dummy_mat = np.zeros((feat.shape[0], max_nodes, max_nodes), dtype=int)
        # If no feat is generated for those extended nodes, set them the same as node 0.
        dummy_mat[:, :feat.shape[1], :feat.shape[1]] = feat
        dummy_mat[:, feat.shape[1]:, :] = dummy_mat[:, :1, :]
        dummy_mat[:, :, feat.shape[1]:] = dummy_mat[:, :, :1]
        edge_feat_list.append(dummy_mat[sample_index, node_index, edge_index])
    return np.stack(edge_feat_list, -1)
