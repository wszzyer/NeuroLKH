import os
import multiprocessing as mp
import tqdm
import numpy as np
import pickle
from net.sgcn_model import SparseGCNModel
import torch
from tqdm import tqdm
import argparse
import time
from utils.dataset import LaDeTestDataset
from torch.utils.data import DataLoader
from functools import partial

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--file_path', type=str, default='data/generated/CVRP_val_scatter_yt_111_100.pkl', help='')
    parser.add_argument('--model_path', type=str, default='saved/exp1/best.pth', help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--lkh_trials', type=int, default=1000, help='')
    parser.add_argument('--neurolkh_trials', type=int, default=1000, help='')
    parser.add_argument("--num-cpus", type=int, default=32, help="num cpus pool")
    parser.add_argument('--use_feats', type=str, action='extend', default=["sssp"], nargs='+', help='')
    parser.add_argument('--device', type=str, default="cuda:0", help='')
    parser.add_argument('--output', type=str, default='', help='')
    return parser.parse_args()

from feats import get_all_feats, parse_feat_strs
from utils.lkh_utils import write_instance, write_para, read_feat, write_candidate
from lade_generate import solve_LKH, max_extra_nodes_ratio
import lade_generate

def infer_SGN(net, test_loader):
    candidate = []
    for batch in tqdm(test_loader, desc="infer SGN"):
        node_feat, edge_feat, edge_index, inverse_edge_index = map(lambda t: t.to(args.device), batch)
        batch_size = node_feat.shape[0]
        y_edges, _, _, y_nodes = net.forward(node_feat, edge_feat, edge_index, inverse_edge_index, None, None, 20)
        y_edges = y_edges.detach().cpu().numpy()
        y_edges = y_edges[:, :, 1].reshape(batch_size, node_feat.shape[1], 20)
        y_edges = np.argsort(-y_edges, -1)
        edge_index = edge_index.cpu().numpy().reshape(-1, y_edges.shape[1], 20)
        candidate_index = edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(y_edges.shape[1]).reshape(1, -1, 1), y_edges]
        candidate.append(candidate_index[:, :, :5])
    candidate = np.concatenate(candidate, 0)
    return candidate

def read_results(log_filename, _, max_trials):
    objs = []
    penalties = []
    runtimes = []
    with open(log_filename, "r") as f:
        lines = f.readlines()
        for line in lines: # read the obj and runtime for each trial
            if line[:6] == "-Trial":
                line = line.strip().split(" ")
                assert len(objs) + 1 == int(line[-4])
                objs.append(int(line[-2]))
                penalties.append(int(line[-3]))
                runtimes.append(float(line[-1]))
        final_obj = int(lines[-11].split(",")[0].split(" ")[-1])
        assert objs[-1] == final_obj
        return objs, penalties, runtimes

def eval_dataset(dataset_filename, method, args, rerun=True, max_trials=1000):
    dataset_name = os.path.splitext(os.path.basename(dataset_filename))[0]
    
    LKH_param_dir = "result/" + dataset_name + "/" + method + "_para"
    LKH_log_dir = "result/" + dataset_name + "/" + method + "_log"
    instance_dir = "result/" + dataset_name + "/cvrp"
    os.makedirs(instance_dir, exist_ok=True)
    os.makedirs(LKH_param_dir, exist_ok=True) 
    os.makedirs(LKH_log_dir, exist_ok=True)
    
    with open(dataset_filename, "rb") as f:
        dataset = pickle.load(f)
    print(dataset[0].keys())
    n_nodes = len(dataset[0]["COORD"]) # n_nodes 包含仓库节点, which is differ from original NeuroLKH.
        
    if method == "NeuroLKH":
        feat_param_dir = "result/" + dataset_name + "/featgen_para"
        feat_dir = "result/" + dataset_name + "/feat"
        candidate_dir = "result/" + dataset_name + "/candidate"
        os.makedirs(feat_param_dir, exist_ok=True) 
        os.makedirs(feat_dir, exist_ok=True)
        os.makedirs(candidate_dir, exist_ok=True)
        
        FEATS = get_all_feats()

        max_nodes = int(n_nodes * max_extra_nodes_ratio)
        n_samples = len(dataset)
        n_neighbours = 20
        feats = tqdm(pool.imap(solve_LKH, [("FeatGenerate", read_feat, instance_dir, feat_param_dir, None, dataset[i], str(i), True, 1, max_nodes, feat_dir) for i in range(len(dataset))]), total=len(dataset))
        edge_index, n_nodes_extend, feat_runtime = list(zip(*feats))
        feat_runtime = np.sum(feat_runtime)
        feat_start_time = time.time()
        edge_index = np.concatenate(edge_index, 0)
        
        # 这一部分代码写重复了，周六再封装的好看一点。
        # construct node features.
        demand = np.stack([d["DEMAND"] for d in dataset])
        demand = np.pad(demand, [(0, 0), (0, max_nodes - n_nodes)], 'constant', constant_values=0)
        capacity = np.zeros([n_samples, max_nodes], dtype=int)
        capacity[:, 0] = np.array([d["CAPACITY"] for d in dataset])
        capacity[:, n_nodes:] = capacity[:, :1]
        assert dataset[0]["DEPOT"] == 1
        x = np.stack([d["COORD"] for d in dataset])
        x = np.concatenate([x, x[:, :1, :].repeat(max_nodes - n_nodes, axis=1)], 1)
        # coords needs scale for faster training.
        node_feat_list = [x, demand[:, :, None], capacity[:, :, None]]
        for feat_class in FEATS:
            if feat_class.feat_type != "node":
                continue
            feat = np.stack([instance["ATTACHMENT"][feat_class] for instance in dataset])
            # Pad with zeros by default. This may cause problems on other features so take care.
            node_feat_list.append(np.concatenate([feat, np.zeros((feat.shape[0], max_nodes - n_nodes, feat.shape[2]))], axis=1))
        node_feat = np.concatenate(node_feat_list, -1)
        
        edge_feat_list = []
        for feat_class in FEATS:
            if feat_class.feat_type != "edge":
                continue
            feat = np.stack([instance["ATTACHMENT"][feat_class] for instance in dataset])
            dummy_mat = np.zeros((feat.shape[0], max_nodes, max_nodes), dtype=int)
            dummy_mat[:, :feat.shape[1], :feat.shape[1]] = feat
            dummy_mat[:, feat.shape[1]:, :] = dummy_mat[:, :1, :]
            dummy_mat[:, :, feat.shape[1]:] = dummy_mat[:, :, :1]
            edge_feat_list.append(dummy_mat[np.arange(n_samples).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), edge_index][..., None])
        edge_feat = np.concatenate(edge_feat_list, -1)
        
        inverse_edge_index = -np.ones(shape=[n_samples, max_nodes, max_nodes], dtype="int")
        inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), edge_index, np.arange(max_nodes).reshape(1, -1, 1)] = np.arange(n_neighbours).reshape(1, 1, -1) + np.arange(max_nodes).reshape(1, -1, 1) * n_neighbours
        inverse_edge_index = inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), edge_index]
        
        feat_runtime += time.time() - feat_start_time

        node_feats_cls, edge_feats_cls = parse_feat_strs(args.use_feats,  print_result=True)
        net = SparseGCNModel(problem="cvrp",
                         n_mlp_layers=4,
                         node_extra_dim=sum(map(lambda cls:cls.size, node_feats_cls)),
                         edge_dim=sum(map(lambda cls:cls.size, edge_feats_cls)))
        net.to(args.device)
        saved = torch.load(args.model_path, weights_only=True)
        net.load_state_dict(saved)
        sgn_start_time = time.time()
        
        test_dataset = LaDeTestDataset("cvrp", node_feat, edge_feat, edge_index, inverse_edge_index, node_feats_cls, edge_feats_cls)
        assert test_dataset.size % args.batch_size == 0
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
        with torch.no_grad():
            candidate = infer_SGN(net, test_loader)
        sgn_runtime = time.time() - sgn_start_time
        results = list(tqdm(pool.imap(solve_LKH, [("NeuroLKH", read_results, instance_dir, LKH_param_dir, LKH_log_dir, dataset[i], str(i), rerun, max_trials, None, candidate_dir, candidate[i], n_nodes_extend[i]) for i in range(len(dataset))]), total=len(dataset)))
    else:
        assert method == "LKH"
        feat_runtime = 0
        sgn_runtime = 0
        results = list(tqdm(pool.imap(solve_LKH, [("LKH", read_results, instance_dir, LKH_param_dir, LKH_log_dir, dataset[i], str(i), rerun, max_trials) for i in range(len(dataset))]), total=len(dataset), desc='Acquiring LKH Result'))
    results = np.array(results)
    dataset_objs = results[:, 0, :].mean(0)
    dataset_penalties = results[:, 1, :].mean(0)
    dataset_runtimes = results[:, 2, :].sum(0)
    return dataset_objs, dataset_penalties, dataset_runtimes, feat_runtime, sgn_runtime

if __name__ == "__main__":
    args = get_args()
    
    pool = mp.Pool(args.num_cpus)
    
    neurolkh_objs, neurolkh_penalties, neurolkh_runtimes, feat_runtime, sgn_runtime = eval_dataset(args.file_path, "NeuroLKH", args=args, rerun=True, max_trials=args.neurolkh_trials) 
    lkh_objs, lkh_penalties, lkh_runtimes, _, _ = eval_dataset(args.file_path, "LKH", args=args, rerun=True, max_trials=args.lkh_trials)

    if args.output:
        file = open(args.output, mode='w') # Throw error upon illegal output parameter
    else:
        file = None
    printf = partial(print, file=file)

    printf ("generating features runtime: %.1fs SGN inferring runtime: %.1fs" % (feat_runtime, sgn_runtime))
    printf ("method obj penalties runtime")
    trials = 1
    while trials <= lkh_objs.shape[0]:
        printf ("------experiments of trials: %d ------" % (trials))
        printf ("LKH      %d %d %ds" % (lkh_objs[trials - 1], lkh_penalties[trials - 1], lkh_runtimes[trials - 1]))
        printf ("NeuroLKH %d %d %ds" % (neurolkh_objs[trials - 1], neurolkh_penalties[trials - 1], neurolkh_runtimes[trials - 1] + feat_runtime + sgn_runtime))
        trials *= 10
    printf ("------comparison with same time limit------")
    trials = 1
    while trials <= lkh_objs.shape[0]:
        printf ("------experiments of trials: %d ------" % (trials))
        printf ("LKH      %d %d %ds" % (lkh_objs[trials - 1], lkh_penalties[trials - 1], lkh_runtimes[trials - 1]))
        neurolkh_trials = 1
        while neurolkh_trials < neurolkh_runtimes.shape[0] and neurolkh_runtimes[neurolkh_trials - 1] + feat_runtime + sgn_runtime < lkh_runtimes[trials - 1]:
            neurolkh_trials += 1
        printf ("NeuroLKH %d %d %ds (%d trials)" % (neurolkh_objs[neurolkh_trials - 1], neurolkh_penalties[trials - 1], neurolkh_runtimes[neurolkh_trials - 1] + feat_runtime + sgn_runtime, neurolkh_trials))
        trials *= 10

    if file:
        file.close()
        with open(args.output) as f:
            for line in f.readlines():
                print(line, end='')
