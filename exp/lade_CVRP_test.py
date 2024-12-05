import multiprocessing as mp
import tqdm
import numpy as np
import pickle
from net import GraphTransformer, SparseGCNModel
import torch
from tqdm import tqdm
import argparse
import time
from utils.dataset import LaDeTestDataset
from torch.utils.data import DataLoader
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--problem", type=str, default="CVRP", choices=["TSP", "CVRP", "CVRPTW", "PDP"], help="which problem")
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument('--file_path', type=str, default='data/generated/CVRP_val_scatter_yt_111_100.pkl', help='')
    parser.add_argument('--model_path', type=str, default='saved/exp1/best.pth', help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--num_candidates', type=int, default=5, help='')
    parser.add_argument('--num_edges', type=int, default=20, help='')
    parser.add_argument("--num_cpus", type=int, default=32, help="num cpus POOL")
    parser.add_argument('--use_feats', type=str, action='extend', default=["sssp"], nargs='+', help='')
    parser.add_argument('--device', type=str, default="cuda:0", help='')
    parser.add_argument('--work_dir', type=str, default="./evaluation", help='')
    parser.add_argument('--output_file', type=str, default='a.out', help='')
    parser.add_argument('--num_trials', type=int, default=30000, help='')
    return parser.parse_args()

from feats import parse_feat_strs
from utils.lkh_utils import read_performance, solve_LKH
from utils.generate_utils import make_edge_feat, make_edge_index, make_node_feat, MAX_EXTRA_NODES_RATIO

def make_candidates(net, test_loader, candidate_count=5, is_cvrptw=False):
    candidate = []
    candidate2 = []
    for batch in tqdm(test_loader, desc="inferring model"):
        node_feat, edge_feat, edge_index, pad_mask = map(lambda t: t.to(args.device), batch)
        batch_size = node_feat.size(0)
        n_nodes = node_feat.size(1)
        n_edges = edge_feat.size(1) // n_nodes
        if not is_cvrptw:
            y_node, y_edge = net.forward(node_feat, edge_feat, edge_index, pad_mask)
        else:
            # TODO:Fix CVRPTW
            y_edge, y_edge2,  _, _, y_nodes = net.directed_forward(node_feat, edge_feat, edge_index, inverse_edge_index, None, None, None, 20)
        
        y_edge = y_edge.detach().cpu().numpy()
        y_edge = y_edge[..., 1].reshape(batch_size, n_nodes, n_edges)
        y_edge = np.argsort(-y_edge, -1)
        edge_index = edge_index.cpu().numpy().reshape(batch_size, n_nodes, n_edges)
        candidate_index = edge_index[*np.ogrid[:batch_size, :n_nodes, :1][:-1], y_edge]
        candidate.append(candidate_index[:, :, :candidate_count])
        # if is_cvrptw:
        #     y_edge2 = y_edge2.detach().cpu().numpy()
        #     y_edge2 = y_edge2[:, :, 1].reshape(batch_size, node_feat.shape[1], 20)
        #     y_edge2 = np.argsort(-y_edge2, -1)
        #     candidate2_index = edge_index[np.arange(batch_size).reshape(-1, 1, 1), np.arange(y_edge2.shape[1]).reshape(1, -1, 1), y_edge2]
        #     candidate2.append(candidate2_index[:, :, :max_candidate])

    candidate = np.concatenate(candidate, 0)
    return candidate
    # if not is_cvrptw:
    #     return candidate
    # else:
    #     candidate2 = np.concatenate(candidate2, 0)
    #     return candidate, candidate2

def eval_model(dataset, args, work_dir, max_trials):
    instance_dir = work_dir / "instance"
    LKH_param_dir = work_dir / "model_para"
    LKH_log_dir = work_dir / "model_log"
    candidate_dir = work_dir / "model_candidate"
    temp_dir = work_dir / "tmp"
    instance_dir.mkdir(parents=True, exist_ok=True)
    LKH_param_dir.mkdir(exist_ok=True)
    LKH_log_dir.mkdir(exist_ok=True)
    candidate_dir.mkdir(exist_ok=True)
    temp_dir.mkdir(exist_ok=True)
    
    n_nodes = len(dataset[0]["COORD"]) # n_nodes 包含仓库节点, which is differ from original NeuroLKH.
    max_nodes = int(n_nodes * MAX_EXTRA_NODES_RATIO) if allow_extend_nodes else n_nodes
    n_edges = args.num_edges

    feat_start_time = time.time()
    edge_index, node_num = make_edge_index(dataset, n_nodes, n_edges, generate_candidate_by_LKH, max_nodes,
                                            temp_dir, POOL)
    node_feat = make_node_feat(dataset, n_nodes, max_nodes)
    edge_feat = make_edge_feat(dataset, max_nodes, edge_index)
    feat_runtime = time.time() - feat_start_time

    node_feats_cls, edge_feats_cls = parse_feat_strs(args.use_feats,  print_result=True)
    # net = SparseGCNModel(problem=args.problem.lower(),
    #                     n_mlp_layers=4,
    #                     node_extra_dim=sum(map(lambda cls:cls.size, node_feats_cls)),
    #                     edge_dim=sum(map(lambda cls:cls.size, edge_feats_cls)))
    
    net = GraphTransformer(problem=args.problem.lower(), 
                        node_extra_dim=sum(map(lambda cls:cls.size, node_feats_cls)), 
                        edge_dim=sum(map(lambda cls:cls.size, edge_feats_cls)),
                        hidden_dim=128,
                        n_mlp_layers=3,
                        n_encoder_layers=6)
    net.to(args.device)
    net.load_state_dict(torch.load(args.model_path, weights_only=True))
    model_start_time = time.time()
    test_dataset = LaDeTestDataset(args.problem.lower(), node_feat, edge_feat, edge_index, node_num, node_feats_cls, edge_feats_cls)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
    with torch.no_grad():
        if args.problem == "CVRP":
            candidate = make_candidates(net, test_loader, is_cvrptw=False)
            candidate2 = [None] * len(candidate)
        else:
            candidate, candidate2 = make_candidates(net, test_loader, is_cvrptw=True)
    model_runtime = time.time() - model_start_time
    #FIXME!fe
    results = list(tqdm(POOL.imap(solve_LKH, [("Model", read_performance, instance_dir, LKH_param_dir, LKH_log_dir, dataset[i], str(i), args.num_candidates,
                                               True, max_trials, candidate_dir, candidate[i], candidate2[i], node_num[i]) for i in range(len(dataset))]), total=len(dataset)))
    results = np.array(results).transpose(1, 0, 2)
    return results, feat_runtime, model_runtime

if __name__ == "__main__":
    # global variables
    args = get_args()
    assert args.problem in ["CVRP", "CVRPTW"]
    if args.problem == "CVRPTW":
        allow_extend_nodes = False
        split_edge_label = True
        generate_candidate_by_LKH = False
    else:
        allow_extend_nodes = True
        split_edge_label = False
        generate_candidate_by_LKH = True
    POOL = mp.Pool(args.num_cpus)

    dataset_path = Path(args.file_path).resolve()
    with dataset_path.open("rb") as f:
        dataset = pickle.load(f)
    exp_name = args.exp_name or dataset_path.stem
    work_dir = Path(args.work_dir).resolve() / exp_name
    
    eval_result = eval_model(dataset, args, work_dir, args.num_trials)

    file = open(args.output_file, mode='wb') # Throw error upon illegal output parameter
    pickle.dump(eval_result, file)
