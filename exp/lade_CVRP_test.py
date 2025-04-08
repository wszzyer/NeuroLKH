import multiprocess as mp
import tqdm
import numpy as np
import pickle
from net import GraphTransformer
import torch
from tqdm import tqdm
import argparse
import time
from utils.dataset import LaDeTestDataset
from torch.utils.data import DataLoader
from pathlib import Path
from feats import get_all_feats, SSSPFeat

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--problem", type=str, default="CVRP", choices=["TSP", "CVRP", "CVRPTW", "PDP"], help="which problem")
    parser.add_argument('--data_dir', type=str, default='data/generated/', help='')
    parser.add_argument('--geo_path', type=str, default='data/generated/CVRP_geo_scatter_yt_111_100.pkl', help='')
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
from utils.instance_utils import read_performance, solve_LKH, solve_kopt
from utils.generate_utils import make_edge_feat, make_node_feat

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

def eval_model(dataset, geo, args, work_dir, max_trials):
    instance_dir = work_dir / "instance"
    param_dir = work_dir / "param"
    output_dir = work_dir / "output"
    candidate_dir = work_dir / "candidates"
    instance_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    candidate_dir.mkdir(exist_ok=True)
    
    feat_start_time = time.time()
    additional_feats = {}
    for feat in FEATS:
        additional_feats[feat] = feat.make_feat(*geo)

    if allow_extend_nodes:
        node_num = np.ceil([instance["SIZE"] + np.sum(instance["DEMAND"]) / instance["CAPACITY"] for instance in dataset]) - 1
    else:
        node_num = np.array([instance["SIZE"] for instance in dataset])
    node_num = node_num.astype(np.int32)
    max_nodes = np.max(node_num)
    n_edges = args.num_edges

    node_feat = make_node_feat(dataset, additional_feats, max_nodes)
    edge_feat, edge_index = make_edge_feat(dataset, additional_feats, max_nodes, n_edges, extend=allow_extend_nodes, node_num=node_num, pool=POOL, chunksize=4)
    feat_runtime = time.time() - feat_start_time

    node_feats_cls, edge_feats_cls = parse_feat_strs(args.use_feats,  print_result=True)
    net = GraphTransformer(problem=args.problem.lower(), 
                        node_extra_dim=sum(map(lambda cls:cls.size, node_feats_cls)), 
                        edge_dim=sum(map(lambda cls:cls.size, edge_feats_cls)),
                        node_hidden_dim=128,
                        n_encoder_layers=6)
    # net.to(args.device) 
    # net.load_state_dict(torch.load(args.model_path, weights_only=True))
    net.load_state_dict(torch.load(args.model_path, weights_only=True), assign=True)
    model_start_time = time.time()
    test_dataset = LaDeTestDataset(args.problem.lower(), node_feat, edge_feat, edge_index, node_num, node_feats_cls, edge_feats_cls)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
    with torch.no_grad():
        if args.problem == "CVRP":
            candidate = make_candidates(net, test_loader, candidate_count=args.num_candidates, is_cvrptw=False)
            candidate2 = [None] * len(candidate)
        else:
            candidate, candidate2 = make_candidates(net, test_loader, candidate_count=args.num_candidates, is_cvrptw=True)
    model_runtime = time.time() - model_start_time

    # results = list(tqdm(POOL.imap(solve_LKH, [("Model", read_performance, instance_dir, param_dir, output_dir, dataset[i], str(i), args.num_candidates,
    #                                            True, max_trials, candidate_dir, candidate[i], candidate2[i], node_num[i]) for i in range(len(dataset))]),
    #                                             desc="Solving Problems", total=len(dataset)))
    results = list(tqdm((solve_kopt(dataset[i], str(i), node_num[i], param_dir, instance_dir, output_dir, candidate[i], candidate_dir, False, max_trials) for i in range(len(dataset))),
                        desc='Solving with k-opt', total=len(dataset)))
    results = np.stack(results).transpose(1, 0, 2)
    return results, feat_runtime, model_runtime

def path_to_name(path: Path):
    return '_'.join(str(path.stem).split('_')[1:][:-4])

if __name__ == "__main__":
    # global variables
    args = get_args()
    assert args.problem in ["CVRP", "CVRPTW"]
    if args.problem == "CVRPTW":
        allow_extend_nodes = False
        split_edge_label = True
    else:
        allow_extend_nodes = True
        split_edge_label = False
    POOL = mp.Pool(args.num_cpus)
    FEATS = get_all_feats()

    dataset_dir = Path(args.data_dir).resolve()
    geo_path = Path(args.geo_path).resolve()
    with geo_path.open("rb") as f:
        geo = pickle.load(f)
    work_dir = Path(args.work_dir).resolve() 
    model_name = Path(args.model_path).resolve().parent.parent.stem

    eval_result = {}
    output_path = Path(args.output_file).resolve()
    if output_path.exists():
        with open(output_path, "rb") as f:
            eval_result = pickle.load(f)
    
    for dataset_path in dataset_dir.iterdir():
        exp_name = path_to_name(dataset_path)
        print(exp_name)
        if 'train' in exp_name or exp_name in eval_result:
            continue
        with dataset_path.open("rb") as f:
            dataset = pickle.load(f)
        if dataset is None:
            raise RuntimeError(f"Fail to load dataset from {dataset_path}.")
        eval_result[exp_name] = eval_model(dataset, geo, args, work_dir / model_name, args.num_trials)
        with open(output_path, "wb") as f:
            pickle.dump(eval_result, f)
    POOL.close()
