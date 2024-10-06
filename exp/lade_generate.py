# Note! LKH can be compiled with gcc-8, and can't be compiled with gcc-10 which will raise compile error.
import os
import multiprocessing as mp
import argparse
import tqdm
from itertools import islice
from subprocess import check_call, DEVNULL
import statistics
import pickle
import functools

import pandas as pd
import numpy as np

from feats import get_all_feats
from utils.lade_utils import fetch_lade, get_bbox_from_coords, load_shapefile_osm_osmnx, fetch_shapefile_osm_osmnx, has_map, transform_crs, decode_gps_traj, encode_gps_traj, SOURCE_CRS, TARGET_CRS
from utils.lkh_utils import read_feat, read_results, write_instance, write_para, write_candidate_dispather
from utils.utils import smooth_matrix, map_wrapper

allow_extend_nodes = None # 在将 VRP 转化为 TSP 时，会添加一些额外节点，这个选项表示神经网络的输入是否包含额外节点。如果不允许额外节点，经过额外节点的路径相当于经过 0 号节点。
split_edge_label = None # 是否分开考虑 edge 的 label。每个点有一个入边 label 和一个出边 label，在对称问题 CVRP 中两类 label 可以合并，在非对称问题 CVRPTW 中两类 label 需要分开。
generate_candidate_by_LKH = None # 是否通过 LKH 生成候选集，否则直接在 python 端生成候选集。
max_extra_nodes_ratio = 1.15
fetch_lade()
np.random.seed(114514)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpus", type=int, default=32, help="num cpus pool")
    parser.add_argument("--n-nodes", type=int, default=100, help="num nodes")
    parser.add_argument("--n-samples", type=int, default=1024, help="num samples")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="train dataset ratio")
    parser.add_argument("--problem", type=str, default="CVRP", choices=["TSP", "CVRP", "CVRPTW", "PDP"], help="which problem")
    parser.add_argument("--citys", action="append", dest="citys", help="citys to generate data")
    parser.add_argument("--n-regions", type=int, default=1, help="only generate datasets for largest `--n-regions`")
    parser.add_argument("--sample-type", type=str, default="scatter", choices=["scatter",  "subroute"], help="scatter: sample directly from all tasks.")
    parser.add_argument("--postfix", type=str, default="", help="dataset postfix")
    return parser.parse_args()

@map_wrapper
def solve_LKH(task, result_hook, instance_dir, param_dir, log_dir, instance, instance_name, rerun=False, max_trials=1000, max_nodes=None, candidate_dir=None, candidate=None, candidate2=None, n_nodes_extend=None):
    """
    solve LKH or NeuroLKH or generate candidate set. (if candidate is given.)
    """
    N_NODES = instance["COORD"].__len__() # this will be refactored.
    para_filename = os.path.join(param_dir, instance_name + ".para")
    log_filename = os.path.join(log_dir, instance_name + ".log") if log_dir else None
    instance_filename = os.path.join(instance_dir, instance_name + ".cvrp")
    candidate_type = "nn" if task == "FeatGenerate" else "alpha"
    candidate_filename = os.path.join(candidate_dir, f"{instance_name}_{candidate_type}.txt") if candidate_dir else None
    if rerun or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename, N_NODES)
        write_para(candidate_filename, instance_filename, task, para_filename, max_trials=max_trials, candidate_set_type=candidate_type)
        if candidate is not None:
            write_candidate_dispather[instance["TYPE"]](feat_filename = candidate_filename, candidate = candidate, candidate2 = candidate2, n_nodes_extend = n_nodes_extend)
        f = open(log_filename, "w") if log_filename else DEVNULL
        check_call(["./LKH", para_filename], stdout=f)
    
    if task == "LKH" or task == "NeuroLKH":
        return result_hook(log_filename, candidate_filename, max_trials)
    else:
        assert task == "FeatGenerate"
        return result_hook(candidate_filename, max_nodes)

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

def gen_CVRP_instance(graph, gdf_nodes, additional_feats, additional_statistic, problem_meta, withTW = False):
    problem_routes, scatter_goods = problem_meta
    instance = {}
    
    instance["TYPE"] = "CVRP" if not withTW else "CVRPTW"
    instance["CAPACITY"] = min(max(N_NODES // len(problem_routes) + 10, N_NODES // (N_NODES * max_extra_nodes_ratio - N_NODES + 1), N_NODES // 20 + 1), N_NODES)
    
    graph_coords = gdf_nodes[["y", "x"]].values
    
    if SAMPLE_TYPE == "subroute":
        whole_df = pd.concat(problem_routes)
        package_coords = whole_df[["lat", "lng"]].values
        package_time_df = whole_df[["delivery_time", "accept_time"]]
    else:
        assert SAMPLE_TYPE == "scatter"
        package_coords = scatter_goods[["lat", "lng"]].values
        package_time_df = scatter_goods[["delivery_time", "accept_time"]]

    package_coords = transform_crs(package_coords, SOURCE_CRS, TARGET_CRS)
    # map package point to graph node.
    # 目前先按照这种简单方式进行映射，可能出现多个快递映射到同一个graph节点。
    corresponding_graph_index = np.linalg.norm(package_coords[:, None] - graph_coords[None], axis=-1).argmin(axis=-1)
    
    instance["COORD"] = graph_coords[corresponding_graph_index]
    # 现在我还没想清楚是建模为 VRP 还是 MTSP 问题。
    # 目前做法是随机选一个位置作为 depot。后面将根据数据推断出仓库所在地，或者转换为 MTSP。
    instance["DEPOT"] = 1
    instance["DEMAND"] = np.ones(N_NODES, dtype=int)
    instance["DEMAND"][instance["DEPOT"] - 1] = 0

    if withTW:
        # 时间会根据平均速度换算到距离
        # below parameters can be tuned.
        instance["VEHICLES"] = 20 # 这个值设大了，应该小一点。动态点数的问题目前不好解决。
        instance["CAPACITY"] = 10000
        twpadding = pd.Timedelta(minutes=15)
        # SERVICE_TIME don't work in CVRPTW problem.
        instance["SERVICE_TIME"] = pd.Timedelta(minutes=5).total_seconds()
        tw_begin = (package_time_df["delivery_time"] - package_time_df["accept_time"] - twpadding).clip(lower=pd.Timedelta(0))
        tw_end = tw_begin + twpadding * 2
        tw_begin_dist = tw_begin.dt.total_seconds() * additional_statistic["velocity"]
        tw_end_dist = tw_end.dt.total_seconds() * additional_statistic["velocity"]
        latest = (tw_end.max().total_seconds() + 1000000) * additional_statistic["velocity"]
        instance["TIME_WINDOW_SECTION"] = np.stack([tw_begin_dist, tw_end_dist], axis=1)
        instance["TIME_WINDOW_SECTION"][instance["DEPOT"] - 1] = [0, latest]

    additional_instance_feats = {}
    for feat_class, problem_meta in additional_feats.items():
        additional_instance_feats[feat_class] = feat_class.generate_cvrp_instance_meta(problem_meta, graph, gdf_nodes, corresponding_graph_index)
    instance["ATTACHMENT"] = additional_instance_feats
    
    return instance

def generate_dataset(dataset, n_nodes, dataset_name):
    # configs.
    # n_nodes 包含仓库节点, which is differ from original NeuroLKH.
    n_samples = len(dataset)
    max_nodes = int(n_nodes * max_extra_nodes_ratio) if allow_extend_nodes else n_nodes
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
    node_feat_list = [x, demand[:, :, None], capacity[:, :, None]]
    if dataset[0]["TYPE"] == "CVRPTW":
        start_end_time = np.stack([d["TIME_WINDOW_SECTION"] for d in dataset])
        node_feat_list += [start_end_time[..., 0:1], start_end_time[..., 1:2]]
    for feat_class in FEATS:
        if feat_class.feat_type != "node":
            continue
        feat = np.stack([instance["ATTACHMENT"][feat_class] for instance in dataset])
        # Pad with node_0 by default.
        node_feat_list.append(np.concatenate([feat, feat[:, :1, :].repeat(max_nodes - n_nodes, axis=1)], axis=1))
    node_feat = np.concatenate(node_feat_list, -1)
    
    if generate_candidate_by_LKH:
        # dummy_mat is matrix with duplicated depot nodes.
        feats = tqdm.tqdm(pool.imap(solve_LKH, [("FeatGenerate", read_feat, instance_dir, feat_param_dir, None, dataset[i], str(i), True, 1, max_nodes, feat_dir) for i in range(len(dataset))]), total=len(dataset), desc='Generating Feat')
        edge_index, n_nodes_extend, _ = list(zip(*feats))
        edge_index = np.concatenate(edge_index, 0)
    else:
        from feats.weight_feat import SSSPFeat
        dist = np.stack([instance["ATTACHMENT"][SSSPFeat] for instance in dataset])
        edge_index = np.argsort(dist, -1)[:, :, 1:1 + n_neighbours]

    results, alpha_raw = zip(*tqdm.tqdm(pool.imap(solve_LKH, [("LKH", read_results, instance_dir, LKH_param_dir, LKH_log_dir, dataset[i], str(i), True, 1000, None, feat_dir) for i in range(len(dataset))], chunksize=8), total=len(dataset), desc='Acquiring LKH Result'))
    if not allow_extend_nodes:
        results = np.array(results)
        results[results > n_nodes] = 0

    # Rewrite edge_index via alpha and take record of alpha values
    alpha_values = np.zeros_like(edge_index)
    if args.problem == "CVRP":
        for problem_index, problem_alpha in enumerate(alpha_raw):
            for index, alpha_list in enumerate(problem_alpha):
                nn_list = list(edge_index[problem_index][index])
                assert len(nn_list) == n_neighbours
                cur = 0
                for target, alpha in alpha_list:
                    if target in nn_list:
                        nn_list.remove(target)
                    edge_index[problem_index][index][cur] = target
                    alpha_values[problem_index][index][cur] = alpha
                    cur += 1
                max_alpha = alpha_list[-1][1] # The alpha list is alread sort
                while cur < n_neighbours:
                    edge_index[problem_index][index][cur] = nn_list.pop(0)
                    alpha_values[problem_index][index][cur] = int(max_alpha * 1.2) # An arbritary penalty
                    cur += 1
    edge_feat_list = []
    for feat_class in FEATS:
        if feat_class.feat_type != "edge":
            continue
        feat = np.stack([instance["ATTACHMENT"][feat_class] for instance in dataset])
        dummy_mat = np.zeros((feat.shape[0], max_nodes, max_nodes), dtype=int)
        # If no feat is generated for those extended nodes, set them the same as node 0.
        dummy_mat[:, :feat.shape[1], :feat.shape[1]] = feat
        dummy_mat[:, feat.shape[1]:, :] = dummy_mat[:, :1, :]
        dummy_mat[:, :, feat.shape[1]:] = dummy_mat[:, :, :1]
        edge_feat_list.append(dummy_mat[np.arange(n_samples).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), edge_index][..., None])
    edge_feat = np.concatenate(edge_feat_list, -1)

    inverse_edge_index = -np.ones(shape=[n_samples, max_nodes, max_nodes], dtype="int")
    inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), edge_index, np.arange(max_nodes).reshape(1, -1, 1)] = np.arange(n_neighbours).reshape(1, 1, -1) + np.arange(max_nodes).reshape(1, -1, 1) * n_neighbours
    inverse_edge_index = inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), edge_index]
    
    # construct edge label.
    label = np.zeros([n_samples, max_nodes, max_nodes], dtype="bool")
    label2 = np.zeros([n_samples, max_nodes, max_nodes], dtype="bool")
    for i in range(n_samples):
        result = np.array(results[i]) - 1
        label[i][result, np.roll(result, 1, -1)] = True
        if not split_edge_label:
            label[i][np.roll(result, 1, -1), result] = True
        else:
            label2[i][np.roll(result, 1, -1), result] = True
    label = label[np.arange(n_samples).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), edge_index]    
    label2 = label2[np.arange(n_samples).reshape(-1, 1, 1), np.arange(max_nodes).reshape(1, -1, 1), edge_index]
    
    feat = {
        "node_feat":node_feat,
        "edge_feat":edge_feat,
        "edge_index":edge_index,
        "inverse_edge_index":inverse_edge_index,
        "alpha_values": alpha_values
    }
    if not split_edge_label:
        feat["label"] = label
    else:
        feat["label1"] = label
        feat["label2"] = label2

    with open("data/generated/" + dataset_name + ".pkl", "wb") as f:
        pickle.dump(feat, f)
        
if __name__ == "__main__":
    # global variables
    args = get_args()
    N_NODES = args.n_nodes
    SAMPLE_TYPE = args.sample_type
    FEATS = get_all_feats()
    if args.problem == "CVRPTW":
        allow_extend_nodes = False
        split_edge_label = True
        generate_candidate_by_LKH = False
    else:
        allow_extend_nodes = True
        split_edge_label = False
        generate_candidate_by_LKH = True
    trajectory_df = None

    pool = mp.Pool(args.num_cpus)

    for city in args.citys:
        # pickup 数据有缺失，只使用 delivery
        # accept_gps_lng/accept_gps_lat 数据完全是乱的，根本没法用。
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
            graph, gdf_nodes, gdf_edges = load_shapefile_osm_osmnx(map_name, target_crs=TARGET_CRS)
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
            
            # generate dataset statistic informations
            additional_statistic = {}
            if args.problem == "CVRPTW":
                # this process is too slow, use calculated velocity by yt-111
                additional_statistic["velocity"] = 1.3475401310142756 # meter/second

                # if trajectory_df is None:
                #     trajectory_df = pd.read_pickle("data/LaDeArchive/data_with_trajectory_20s/courier_detailed_trajectory_20s.pkl")
                # bbox_encode = pd.DataFrame([[bbox[1], bbox[3]], [bbox[0], bbox[2]]], columns=["lat", "lng"])
                # encode_gps_traj(bbox_encode)
                # region_traj_df = trajectory_df[(trajectory_df["lat"] >= bbox_encode.loc[0, "lat"]) & 
                #                                 (trajectory_df["lat"] <= bbox_encode.loc[1, "lat"]) &
                #                                 (trajectory_df["lng"] >= bbox_encode.loc[0, "lng"]) &
                #                                 (trajectory_df["lng"] <= bbox_encode.loc[1, "lng"])].copy()
                # decode_gps_traj(region_traj_df)
                # region_traj_df["gps_time"] = pd.to_datetime("2023-" + region_traj_df["gps_time"], format='%Y-%m-%d %H:%M:%S')
                # most_day = statistics.multimode(region_traj_df["gps_time"].dt.date)[0]
                # day_traj_df = region_traj_df[region_traj_df["gps_time"].dt.date == most_day]
                # adjacent_time = day_traj_df.iloc[1:]["gps_time"].reset_index(drop = True) - day_traj_df.iloc[:-1]["gps_time"].reset_index(drop = True)
                # coords_proj = transform_crs(day_traj_df[["lat", "lng"]].values, SOURCE_CRS, TARGET_CRS)
                # adjacent_length = np.linalg.norm(coords_proj[1:] - coords_proj[:-1], axis=-1)
                # valid_mask = (adjacent_time < pd.Timedelta(minutes=1)) & (adjacent_time > pd.Timedelta(minutes=0))
                # velocity = (adjacent_length / (adjacent_time.astype(int)/1e9) )[valid_mask].mean()
                # additional_statistic["velocity"] = velocity

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
                    "CVRPTW": functools.partial(gen_CVRP_instance, withTW = True),
                    "PDP": None
                }[args.problem]
                return list(tqdm.tqdm(pool.imap(functools.partial(generate_function, graph, gdf_nodes, additional_meta, additional_statistic), problems_meta,
                                                 chunksize=min(32, n_samples // 8)), desc='Generating Instance', total=n_samples))
                
            train_instance = process_rdf(train_rdf, args.n_samples)
            val_instance = process_rdf(val_rdf, 32)
            
            
            postfix = "_" + args.postfix if args.postfix else ""
            dataset_name_template = f"{args.problem}_%s_{args.sample_type}_{city}_{region_id}_{N_NODES}{postfix}"
            
            # save raw instance to file, and can run lade_CVRP_train.py to evaluate it.
            # 目前用验证集的 instance 当成测试集，周六再改。
            os.makedirs("data/raw_instance", exist_ok=True)
            with open("data/raw_instance/" + dataset_name_template % "train_raw" + ".pkl", "wb") as f:
                pickle.dump(train_instance, f)
            with open("data/raw_instance/" + dataset_name_template % "val_raw" + ".pkl", "wb") as f:
                pickle.dump(val_instance, f)
            
            generate_dataset(train_instance, N_NODES, dataset_name_template % "train")
            generate_dataset(val_instance, N_NODES, dataset_name_template % "val")
    
