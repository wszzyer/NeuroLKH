# Note! LKH can be compiled with gcc-8, and can't be compiled with gcc-10 which will raise compile error.
import os
import multiprocessing as mp
import argparse
import tqdm
from itertools import cycle, islice
from subprocess import check_call
import tempfile
import pickle
import functools

import pandas as pd
import numpy as np

from feats import get_all_feats
from utils.lade_utils import fetch_lade, get_bbox_from_coords, load_shapefile_osm_osmnx, fetch_shapefile_osm_osmnx, has_map, transform_crs, SOURCE_CRS, TARGET_CRS
from utils.lkh_utils import read_feat, read_results, write_instance, write_para

max_extra_nodes_ratio = 1.15
fetch_lade()
np.random.seed(114514)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-cpus",
    type=int,
    default=32,
    help="num cpus pool"
)
parser.add_argument(
    "--n-nodes",
    type=int,
    default=100,
    help="num nodes"
)
parser.add_argument(
    "--n-samples",
    type=int,
    default=1024,
    help="num samples"
)
parser.add_argument(
    "--train-ratio",
    type=float,
    default=0.8,
    help="train dataset ratio"
)
parser.add_argument(
    "--problem",
    type=str,
    default="CVRP",
    choices=["TSP", "CVRP", "CVRPTW", "PDP"],
    help="which problem"
)
parser.add_argument(
    "--citys",
    action="append",
    dest="citys",
    help="citys to generate data"
)
parser.add_argument(
    "--n-regions",
    type=int,
    default=1,
    help="only generate datasets for largest `--n-regions`"
)
parser.add_argument(
    "--sample-type",
    type=str,
    default="scatter",
    choices=["scatter",  "subroute"],
    help="scatter: sample directly from all tasks. "
)
parser.add_argument(
    "--postfix",
    type=str,
    default="",
    help="dataset postfix"
)
args = parser.parse_args()


def map_wrapper(func):
    @functools.wraps(func)
    def expand_args_for_func(args):
        return func(*args)
    return expand_args_for_func

@map_wrapper
def solve_LKH(instance_dir, LKH_param_dir, LKH_log_dir, instance, instance_name, rerun=False, max_trials=1000):
    para_filename = os.path.join(LKH_param_dir, instance_name + ".para")
    log_filename = os.path.join(LKH_log_dir, instance_name + ".log")
    instance_filename = os.path.join(instance_dir, instance_name + ".cvrp")
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename, N_NODES)
        write_para(None, instance_filename, "LKH", para_filename, max_trials=max_trials)
        with open(log_filename, "w") as f:
            check_call(["./LKH", para_filename], stdout=f)
    return read_results(log_filename, max_trials)

@map_wrapper
def generate_feat(instance_dir, feat_param_dir, feat_dir, instance, instance_name, max_nodes):
    para_filename = os.path.join(feat_param_dir, instance_name + ".para")
    instance_filename = os.path.join(instance_dir, instance_name + ".cvrp")
    feat_filename = os.path.join(feat_dir, instance_name + ".txt")
    write_instance(instance, instance_name, instance_filename, N_NODES)
    write_para(feat_filename, instance_filename, "FeatGenerate", para_filename)
    with tempfile.TemporaryFile() as f:
        check_call(["./LKH", para_filename], stdout=f)
    return read_feat(feat_filename, max_nodes)

def gen_TSP_instance(graph, gdf_nodes, additional_feats, problem_meta):
    problem_routes, scatter_goods = problem_meta
    instance = {}
    
    instance["TYPE"] = "TSP"
    graph_coords = gdf_nodes[["y", "x"]].values
    
    package_coords = []
    for df in problem_routes:
        # map package point to graph node.
        # 目前先按照这种简单方式进行映射，可能出现多个快递映射到同一个graph节点。
        package_coords.append(df[["lat", "lng"]].values)
    package_coords = np.vstack(package_coords)
    package_coords = transform_crs(package_coords, SOURCE_CRS, TARGET_CRS)
    package_to_node_dist = np.linalg.norm(package_coords[:, None] - graph_coords[None], axis=-1)
    corresponding_graph_index = package_to_node_dist.argmin(axis=-1)
    print(f"average distance from package to node {package_to_node_dist.min(axis=-1).mean()}")
    
    instance["COORD"] = graph_coords[corresponding_graph_index]

    additional_instance_feats = {}
    for feat_class, problem_meta in additional_feats.items():
        additional_instance_feats[feat_class] = feat_class.generate_tsp_instance_meta(problem_meta, graph, gdf_nodes, corresponding_graph_index)
    instance["ATTACHMENT"] = additional_instance_feats
    
    return instance

def gen_CVRP_instance(graph, gdf_nodes, additional_feats, problem_meta):
    problem_routes, scatter_goods = problem_meta
    instance = {}
    
    instance["TYPE"] = "CVRP"
    instance["CAPACITY"] = min(max(N_NODES // len(problem_routes) + 10, N_NODES // (N_NODES * max_extra_nodes_ratio - N_NODES + 1)), N_NODES)
    
    graph_coords = gdf_nodes[["y", "x"]].values
    
    if SAMPLE_TYPE == "subroute":
        package_coords = []
        for df in problem_routes:
            # map package point to graph node.
            # 目前先按照这种简单方式进行映射，可能出现多个快递映射到同一个graph节点。
            package_coords.append(df[["lat", "lng"]].values)
        package_coords = np.vstack(package_coords)
    else:
        assert SAMPLE_TYPE == "scatter"
        package_coords = scatter_goods[["lat", "lng"]].values
    package_coords = transform_crs(package_coords, SOURCE_CRS, TARGET_CRS)
    corresponding_graph_index = np.linalg.norm(package_coords[:, None] - graph_coords[None], axis=-1).argmin(axis=-1)
    
    instance["COORD"] = graph_coords[corresponding_graph_index]
    # 现在我还没想清楚是建模为 VRP 还是 MTSP 问题。
    # 目前做法是随机选一个位置作为 depot。后面将根据数据推断出仓库所在地，或者转换为 MTSP。
    instance["DEPOT"] = 1
    instance["DEMAND"] = np.ones(N_NODES, dtype=int)
    instance["DEMAND"][instance["DEPOT"] - 1] = 0

    additional_instance_feats = {}
    for feat_class, problem_meta in additional_feats.items():
        additional_instance_feats[feat_class] = feat_class.generate_cvrp_instance_meta(problem_meta, graph, gdf_nodes, corresponding_graph_index)
    instance["ATTACHMENT"] = additional_instance_feats
    
    return instance

def gen_CVRPTW_instance(graph, gdf_nodes, additional_feats, problem_meta):
    problem_routes, scatter_goods = problem_meta
    instance = {}
    
    instance["TYPE"] = "CVRP"
    instance["CAPACITY"] = min(N_NODES // len(problem_routes) + 10, N_NODES)
    
    graph_coords = gdf_nodes[["y", "x"]].values
    
    package_coords = []
    for df in problem_routes:
        # map package point to graph node.
        # 目前先按照这种简单方式进行映射，可能出现多个快递映射到同一个graph节点。
        package_coords.append(df[["lat", "lng"]].values)
    package_coords = np.vstack(package_coords)
    package_coords = transform_crs(package_coords, SOURCE_CRS, TARGET_CRS)
    corresponding_graph_index = np.linalg.norm(package_coords[:, None] - graph_coords[None], axis=-1).argmin(axis=-1)
    
    instance["COORD"] = graph_coords[corresponding_graph_index]

    additional_instance_feats = {}
    for feat_class, problem_meta in additional_feats.items():
        additional_instance_feats[feat_class] = feat_class.generate_cvrptw_instance_meta(problem_meta, graph, gdf_nodes, corresponding_graph_index)
    instance["ATTACHMENT"] = additional_instance_feats

    # 现在我还没想清楚是建模为 VRP 还是 MTSP 问题。
    # 目前做法是随机选一个位置作为 depot。后面将根据数据推断出仓库所在地，或者转换为 MTSP。
    instance["DEPOT"] = 1
    instance["DEMAND"] = np.ones(N_NODES, dtype=int)
    instance["DEMAND"][instance["DEPOT"] - 1] = 0
    
    return instance

def generate_dataset(dataset, n_nodes, dataset_name):
    # configs.
    # n_nodes 包含仓库节点, which is differ from original NeuroLKH.
    n_samples = len(dataset)
    max_nodes = int(n_nodes * max_extra_nodes_ratio)
    n_neighbours = 20
    
    # temperory directories.
    instance_dir = "tmp/" + dataset_name + "/instance"
    feat_param_dir = "tmp/" + dataset_name + "/featgen_para"
    feat_dir = "tmp/" + dataset_name + "/feat"
    LKH_param_dir = "tmp/" + dataset_name + "/LKH_para"
    LKH_log_dir = "tmp/" + dataset_name + "/LKH_log"
    
    os.makedirs("data/generated/", exist_ok=True)
    os.makedirs(instance_dir, exist_ok=True)
    os.makedirs(feat_param_dir, exist_ok=True) 
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(LKH_param_dir, exist_ok=True) 
    os.makedirs(LKH_log_dir, exist_ok=True)
    
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
    node_feat = np.concatenate([x, demand[:, :, None], capacity[:, :, None]], -1)
    
    # construct edge features.
    # dummy_mat is matrix with duplicated depot nodes.
    feats = list(tqdm.tqdm(pool.imap(generate_feat, [(instance_dir, feat_param_dir, feat_dir, dataset[i], str(i), max_nodes) for i in range(len(dataset))]), total=len(dataset), desc='Generating Feat'))
    edge_index, n_nodes_extend = list(zip(*feats))
    edge_index = np.concatenate(edge_index, 0)

    edge_feat_list = []
    feat_item = dataset[0]["ATTACHMENT"]
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
    
    # construct edge label.
    results = list(tqdm.tqdm(pool.imap(solve_LKH, [(instance_dir, LKH_param_dir, LKH_log_dir, dataset[i], str(i), True, 10000) for i in range(len(dataset))], chunksize=8), total=len(dataset), desc='Acquiring LKH Result'))
    label = np.zeros([n_samples, max_nodes, max_nodes], dtype="bool")
    for i in range(n_samples):
        result = np.array(results[i]) - 1
        label[i][result, np.roll(result, 1)] = True
        label[i][np.roll(result, 1), result] = True
    label = label[np.arange(n_samples).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), edge_index]    
    
    feat = {"node_feat":node_feat,
            "edge_feat":edge_feat,
            "edge_index":edge_index,
            "inverse_edge_index":inverse_edge_index,
            "label":label}
        
    with open("data/generated/" + dataset_name + ".pkl", "wb") as f:
        pickle.dump(feat, f)
        
if __name__ == "__main__":
    # global variables
    N_NODES = args.n_nodes
    SAMPLE_TYPE = args.sample_type
    FEATS = get_all_feats()
    
    pool = mp.Pool(args.num_cpus)

    for city in args.citys:
        # pickup 数据有缺失，只使用 delivery
        df = pd.read_csv(f"./data/LaDeArchive/delivery/delivery_{city}.csv")
        for data_col in ["accept_time", "accept_gps_time", "delivery_time", "delivery_gps_time"]:
            df[data_col] = pd.to_datetime("2023-" + df[data_col], format='%Y-%m-%d %H:%M:%S')
        for region_id, rdf in islice(sorted(df.groupby("region_id"), key=lambda g: -len(g[1])), args.n_regions):
            # rdf: region DataFrame
            rdf = rdf.reset_index()
            map_name = f"{city}_{region_id}"
            coords = rdf[["lat", "lng"]].to_numpy()
            bbox = get_bbox_from_coords(coords, paddings=np.array([0.01, 0.01]))
            if not has_map(map_name):
                fetch_shapefile_osm_osmnx(place=bbox, map_name=map_name)
            graph, gdf_nodes, _ = load_shapefile_osm_osmnx(map_name, target_crs=TARGET_CRS)
            rdf = rdf.drop(rdf.index[(rdf["lat"] > bbox[0]) | (rdf["lat"] < bbox[1]) | (rdf["lng"] > bbox[2]) | (rdf["lng"] < bbox[3])])
            print(f"region {city}-{region_id}: {len(rdf)} tasks, {len(graph)} nodes.")
            
            # 目前采用按时间切分训练集和验证集，但是训练集和验证集中存在相同快递员，神经网络可以提前学到某些快递员的行为特征，实际上不应该如此。
            # 后面也可以尝试按快递员切分训练集和验证集，以解决这个问题。
            # 目前采样快递员配送任务时用的是可放回采样，某一个任务可能被采样多次。
            rdf = rdf.sort_values("delivery_time")
            split_index = int(len(rdf) * args.train_ratio)
            train_rdf = rdf.iloc[:split_index]
            val_rdf= rdf.iloc[split_index:]

            # generate additional features
            additional_meta = {}
            for feat in FEATS:
                additional_meta[feat] = feat.generate_problem_meta(rdf, graph, gdf_nodes)
            
            # generate global statistics/features
            # generate OD matrix
            od = np.zeros((len(graph), len(graph)), dtype = int)
            graph_coords = gdf_nodes[["y", "x"]].values
            for courier_id, courier_rdf in rdf.groupby("courier_id"):
                for _, courier_day_rdf in courier_rdf.groupby(courier_rdf["delivery_time"].dt.date):
                    route_coords = courier_day_rdf[["lat", "lng"]].values
                    route_coords = transform_crs(route_coords, source_crs, target_crs)
                    package_to_node_dist = np.linalg.norm(route_coords[:, None] - graph_coords[None], axis=-1)
                    corresponding_graph_index = package_to_node_dist.argmin(axis=-1)
                    od[corresponding_graph_index[:-1], corresponding_graph_index[1:]] += 1
            # smooth OD matrix by aggregate neighbours's data.
            id_to_index = {id: index for index, id in enumerate(gdf_nodes.index)}
            adjs = [[id_to_index[v_id] for v_id in graph.neighbors(u_id)] for u_id in gdf_nodes.index] # each node's neighboures.
            global whole_od
            whole_od = smooth_matrix(adjs, od)
                
            def process_rdf(rdf, n_samples):
                sub_routes = []
                for courier_id, courier_rdf in rdf.groupby("courier_id"):
                    # courier_rdf currently is sorted by delivery time.
                    # 暂时没想到如何获取快递员每个trip运送的是哪些货物，就把每天当成一个trip
                    for _, courier_day_rdf in courier_rdf.groupby(courier_rdf["delivery_time"].dt.date):
                        sub_routes.append(courier_day_rdf)
                # 目前仅支持点数固定的问题。实际问题中点数往往是不固定的。
                print(f"region {city}-{region_id}: {len(sub_routes)} sub_routes.")


                # generate common part of instances
                problems_meta = []
                for problem_index in range(n_samples):
                    # generate instance using sub routes.
                    problem_routes = []
                    problem_size_so_far = 0
                    selected_flag = np.zeros(len(sub_routes), dtype=int)
                    while True:
                        i = np.random.randint(0, len(sub_routes))
                        if selected_flag[i] == 1:
                            continue
                        selected_flag[i] = 1
                        
                        if problem_size_so_far + len(sub_routes) > args.n_nodes:
                            partial_tour_df = sub_routes[i].iloc[:args.n_nodes - problem_size_so_far]
                        else:
                            partial_tour_df = sub_routes[i]
                        problem_routes.append(partial_tour_df)
                        problem_size_so_far += len(partial_tour_df)
                        
                        if problem_size_so_far == args.n_nodes:
                            break
                        
                    # generate instance using scatter goods
                    scatter_goods = rdf.loc[np.random.choice(rdf.index, size = N_NODES, replace=False)]
                    problems_meta.append((problem_routes, scatter_goods))
                
                # generate spcific part of instances 
                generate_function = {
                    "TSP": gen_TSP_instance,
                    "CVRP": gen_CVRP_instance,
                    "CVRPTW": gen_CVRPTW_instance,
                    "PDP": None
                }[args.problem]
                return list(tqdm.tqdm(pool.imap(functools.partial(generate_function, graph, gdf_nodes, additional_meta), problems_meta,
                                                 chunksize=min(32, n_samples // 8)), desc='Generating Instance', total=n_samples))
                
            train_instance = process_rdf(train_rdf, args.n_samples)
            val_instance = process_rdf(val_rdf, 32)
            
            generate_dataset(train_instance, N_NODES, f"{args.problem}_train_{args.sample_type}_{city}_{region_id}_{N_NODES}_{args.postfix}")
            generate_dataset(val_instance, N_NODES, f"{args.problem}_val_{args.sample_type}_{city}_{region_id}_{N_NODES}_{args.postfix}")
    
