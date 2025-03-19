import multiprocess as mp
import argparse
import tqdm
from itertools import islice, chain
from pathlib import Path
import pickle
import functools

import pandas as pd
import numpy as np

from feats import get_all_feats, SSSPFeat
from utils.lade_utils import fetch_lade, get_bbox_from_coords, load_shapefile_osm_osmnx, fetch_shapefile_osm_osmnx, has_map, transform_crs, SOURCE_CRS, TARGET_CRS
from utils.instance_utils import read_solution, solve_LKH
from utils.generate_utils import make_node_feat, make_edge_feat

allow_extend_nodes = None # 在将 VRP 转化为 TSP 时，会添加一些额外节点，这个选项表示神经网络的输入是否包含额外节点。如果不允许额外节点，经过额外节点的路径相当于经过 0 号节点。
split_edge_label = None # 是否分开考虑 edge 的 label。每个点有一个入边 label 和一个出边 label，在对称问题 CVRP 中两类 label 可以合并，在非对称问题 CVRPTW 中两类 label 需要分开。
N_EDGES = 20
fetch_lade()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpus", type=int, default=32, help="num cpus pool")
    parser.add_argument("--n-edges", type=int, default=20, help="num edges")
    parser.add_argument("--n-samples", type=int, default=1024, help="num samples")
    parser.add_argument("--train-ratio", type=float, default=0.833, help="train dataset time ratio")
    parser.add_argument("--problem", type=str, default="CVRP", choices=["TSP", "CVRP", "CVRPTW", "PDP"], help="which problem")
    parser.add_argument("--cities", action="extend", nargs="+", dest="cities", help="cities to generate data")
    parser.add_argument("--n-regions", type=int, default=1, help="only generate datasets for largest `--n-regions`")
    parser.add_argument("--sample-type", type=str, default="scatter", choices=["scatter",  "subroute"], help="scatter: sample directly from all tasks.")
    parser.add_argument("--output-dir", type=str, default="./data", help="data save folder")
    parser.add_argument("--seed", type=int, default=42, help="seed for data sampling")
    return parser.parse_args()

def gen_TSP_instance(graph_coords, additional_statistic, rdf, gen_count=8):
    instance = {}  
    result = []
    for _ in range(gen_count):
        instance = {
            "TYPE": "TSP"
        }
        sampled_df = rdf.sample(frac=0.8, replace=True)
        problem_df = sampled_df.graph_index.value_counts(sort=False)

        graph_index = problem_df.index.to_numpy()
        raise RuntimeError("Not Sync with CVRP!")
        instance["COORD"] = graph_coords[problem_df.index]
        instance["WEIGHT"] = additional_statistic["dist"][graph_index]
        # This is kept for further use
        instance["SIZE"] = len(problem_df)
        instance["GRAPH_INDEX"] = problem_df.index.to_numpy()
    return result

def gen_CVRP_instance(graph_coords, additional_statistic, rdf, gen_count=8, gen_frac=0.9, seed=42, withTW = False):
    if gen_frac <= 1 and len(rdf) * gen_frac < 21: # 20 node + one depot. Ugly though.
        return []
    if gen_frac > len(rdf):
        print(f"Warning: Sampling instance of size {gen_frac} from rdf of size {len(rdf)}. Skipped.")
        return []
    TYPE = "CVRP" if not withTW else "CVRPTW"
    estimated_routes_num = rdf.groupby(pd.Grouper(key="delivery_time", freq="D")).apply(lambda df: df.courier_id.nunique(), include_groups=False).sum()
    # raise RuntimeError(estimated_routes_num.sum())
    # node_count = rdf.graph_index.nunique()
    node_count = len(rdf)
    DEMANDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    MIN_CAPACITY = max(5, int(node_count * 0.05))
    MAX_CAPACITY = min(100, int(node_count * 0.2))
    CAPACITY = int(np.ceil(min(max(node_count // estimated_routes_num + 5, MIN_CAPACITY), MAX_CAPACITY) * np.mean(DEMANDS)))
    
    random_state = np.random.RandomState(seed)
    result = []
    for _ in range(gen_count):
        instance = {
            "TYPE": TYPE,
            "CAPACITY": CAPACITY
        }
        if gen_frac > 1:
            sampled_df = rdf.sample(n=gen_frac, replace=True, random_state=random_state)
        else:
            sampled_df = rdf.sample(frac=gen_frac, replace=True, random_state=random_state)

        graph_index = sampled_df.graph_index.to_numpy()
        instance["COORD"] = transform_crs(sampled_df[["lat", "lng"]].to_numpy(), SOURCE_CRS, TARGET_CRS)
        # Use merged weights to avoid zeros. Even though they act like nothing but zeros.
        weight = additional_statistic["dist"][graph_index]
        instance["SIZE"] = len(graph_index)
        diff = np.abs(instance["COORD"] - graph_coords[graph_index]).sum(axis=-1)
        instance["WEIGHT"] = weight + diff[np.newaxis, :] + diff[:, np.newaxis]
        instance["WEIGHT"][np.arange(instance["SIZE"]), np.arange(instance["SIZE"])] = 0
        instance["GRAPH_INDEX"] = graph_index
        instance["DEPOT"] = 1
        instance["DEMAND"] = random_state.choice(DEMANDS, graph_index.shape, replace=True)
        instance["DEMAND"][instance["DEPOT"] - 1] = 0
        min_depot_num = int(np.ceil(np.sum(instance["DEMAND"]) / instance["CAPACITY"]))
        instance["SPECIAL"] = np.argsort(instance["WEIGHT"][instance["DEPOT"] - 1])[1:min_depot_num+ 1] + 1

        if withTW:
            # 时间会根据平均速度换算到距离
            # below parameters can be tuned.
            instance["VEHICLES"] = 20 # 这个值设大了，应该小一点。动态点数的问题目前不好解决。
            instance["CAPACITY"] = 10000
            twpadding = pd.Timedelta(minutes=15)
            # SERVICE_TIME don't work in CVRPTW problem.
            instance["SERVICE_TIME"] = pd.Timedelta(minutes=5).total_seconds()
            tw_begin = (sampled_df["delivery_time"] - sampled_df["accept_time"] - twpadding).clip(lower=pd.Timedelta(0))
            tw_end = tw_begin + twpadding * 2
            tw_begin_dist = tw_begin.dt.total_seconds() * additional_statistic["velocity"]
            tw_end_dist = tw_end.dt.total_seconds() * additional_statistic["velocity"]
            latest = (tw_end.max().total_seconds() + 1000000) * additional_statistic["velocity"]
            instance["TIME_WINDOW_SECTION"] = np.stack([tw_begin_dist, tw_end_dist], axis=1)
            instance["TIME_WINDOW_SECTION"][instance["DEPOT"] - 1] = [0, latest]
        result.append(instance)
    return result

def generate_dataset(dataset, additional_feats, dataset_name, output_dir):
    # n_nodes 包含仓库节点, which is different from original NeuroLKH.
    n_samples = len(dataset)
    # LKH Exodus Note: See MTSP2TSP.c:33 and ReadProblem.c:524.
    if allow_extend_nodes:
        node_num = np.ceil([instance["SIZE"] + np.sum(instance["DEMAND"]) / instance["CAPACITY"] for instance in dataset]) - 1
    else:
        node_num = np.array([instance["SIZE"] for instance in dataset])
    node_num = node_num.astype(np.int32)
    max_nodes = np.max(node_num)
    
    # temperory directories.
    tmp_dir = output_dir / "tmp"
    instance_dir = tmp_dir / dataset_name / "instance"
    LKH_param_dir = tmp_dir / dataset_name /  "LKH_para"
    LKH_log_dir = tmp_dir / dataset_name /  "LKH_log"
    generated_dir = output_dir / "generated"
    
    instance_dir.mkdir(exist_ok=True, parents=True)
    LKH_param_dir.mkdir(exist_ok=True)
    LKH_log_dir.mkdir(exist_ok=True)
    generated_dir.mkdir(exist_ok=True)
    
    # construct node features.
    node_feat = make_node_feat(dataset, additional_feats, max_nodes)
    edge_feat, edge_index = make_edge_feat(dataset, additional_feats, max_nodes, N_EDGES, extend=allow_extend_nodes, node_num=node_num, pool=pool)

    # The sizes of the problems varies greatly. A large chunksize will cause great inbalance between threads.
    results = list(tqdm.tqdm(pool.imap(solve_LKH, [("LabelGen", read_solution, instance_dir, LKH_param_dir, LKH_log_dir, dataset[i], str(i), N_EDGES,
                                        True, 1000, tmp_dir / dataset_name / "feat") for i in range(len(dataset))], chunksize=8), total=len(dataset), desc='Acquiring LKH Result'))
    if not allow_extend_nodes:
        results = np.array(results)
        raise RuntimeError(result.shape, node_num.shape)
        results[results > node_num] = 0

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
        "node_feat": node_feat,
        "edge_feat": edge_feat,
        "edge_index": edge_index,
        "node_num": node_num
    }
    if not split_edge_label:
        feat["label"] = label
    else:
        feat["label"] = np.stack((label, label2)).transpose(1, 0, 2, 3)

    with (generated_dir / f"{dataset_name}.pkl").open("wb") as f:
        pickle.dump(feat, f)
        
if __name__ == "__main__":
    # global variables
    args = get_args()
    N_EDGES = args.n_edges
    SAMPLE_TYPE = args.sample_type
    FEATS = get_all_feats()
    if args.problem == "CVRPTW":
        allow_extend_nodes = False
        split_edge_label = True
    else:
        allow_extend_nodes = True
        split_edge_label = False
    trajectory_df = None
    output_dir = Path(args.output_dir).resolve()

    pool = mp.Pool(args.num_cpus)

    for city in args.cities:
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
            print(f"Generating graph from {bbox}")
            if not has_map(map_name):
                fetch_shapefile_osm_osmnx(place=bbox, map_name=map_name)
            graph, gdf_nodes, _ = load_shapefile_osm_osmnx(map_name, target_crs=TARGET_CRS)
            rdf = rdf.drop(rdf.index[(rdf["lat"] > bbox[0]) | (rdf["lat"] < bbox[1]) | (rdf["lng"] > bbox[2]) | (rdf["lng"] < bbox[3])])
            print(f"region {city}-{region_id}: {len(rdf)} tasks, {len(graph)} nodes.")
        
            # Will the behavior of deliveryman result in "Data Leak"? I reckon the answer is no, imasara.
            # LaDe dataset last for 6 months, so we just use the ratio 5:1 by default here.
            start_day = rdf.delivery_time.min().round("D")
            interval = (rdf.delivery_time.max() - start_day).round("D")
            split_day = (start_day + interval * args.train_ratio).round("D")
            
            # map package point to graph node.
            package_coords = transform_crs(rdf[["lat", "lng"]].to_numpy(), SOURCE_CRS, TARGET_CRS)
            graph_coords = gdf_nodes[["y", "x"]].to_numpy()
            corresponding_graph_index = np.linalg.norm(package_coords[:, np.newaxis, :] - graph_coords[np.newaxis, ...], axis=-1).argmin(axis=-1)
            rdf["graph_index"] = corresponding_graph_index

            train_rdf = rdf[rdf.delivery_time <= split_day]
            val_rdf= rdf[rdf.delivery_time > split_day]
            
            # generate dataset statistic informations
            additional_statistic = {}
            additional_statistic["dist"] = SSSPFeat.make_feat(rdf, graph, gdf_nodes)
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

            def sample_instance(rdf, windows=["D", "3D"], gen_kwargs={}):
                rdf = rdf.reset_index(drop=True)
                sub_routes = []
                for courier_id, courier_rdf in rdf.groupby("courier_id"):
                    # courier_rdf currently is sorted by delivery time.
                    # 暂时没想到如何获取快递员每个trip运送的是哪些货物，就把每天当成一个trip
                    for _, courier_day_rdf in courier_rdf.groupby(courier_rdf["delivery_time"].dt.date):
                        sub_routes.append(courier_day_rdf)
                # 目前仅支持点数固定的问题。实际问题中点数往往是不固定的。
                print(f"region {city}-{region_id}: {len(sub_routes)} sub_routes.")

                instance_list = []
                for window in windows:
                    grouper = pd.Grouper(key="delivery_time", freq=window)
                    instance_list += [df for (_, df) in rdf.groupby(grouper)]
                
                # generate spcific part of instances 
                generate_function = {
                    "TSP": gen_TSP_instance,
                    "CVRP": gen_CVRP_instance,
                    "CVRPTW": functools.partial(gen_CVRP_instance, withTW = True),
                    "PDP": None
                }[args.problem]
                return list(chain(*tqdm.tqdm(pool.imap(functools.partial(generate_function, graph_coords, additional_statistic, seed=args.seed, **gen_kwargs),
                                                        instance_list, chunksize=32), desc='Generating Instance', total=len(instance_list))))
                
            train_instance = sample_instance(train_rdf)
            val_instance = sample_instance(val_rdf)
            
            dataset_name_template = f"{args.problem}_%s_{args.sample_type}_{city}_{region_id}_{N_EDGES}"
            # save raw instance to file, and can run lade_CVRP_train.py to evaluate it.
            raw_dir = output_dir / "raw_instance"
            raw_dir.mkdir(exist_ok=True)
            raw_city_dir = raw_dir / f"{city}_{region_id}_{N_EDGES}"
            raw_city_dir.mkdir(exist_ok=True)
            with open(raw_city_dir / (dataset_name_template % "train_raw" + ".pkl"), "wb") as f:
                pickle.dump(train_instance, f)
            with open(raw_city_dir / (dataset_name_template % "val_raw" + ".pkl"), "wb") as f:
                pickle.dump(val_instance, f)
            with open(raw_dir / (dataset_name_template % "geo_raw" + ".pkl"), "wb") as f:
                pickle.dump((rdf, graph, gdf_nodes), f)
            # save fixed size problems to another dir
            for size in (100, 200, 500, 1000):
                instance = sample_instance(val_rdf, windows=['W'], gen_kwargs={'gen_count': 64, 'gen_frac': size})
                if instance:
                    with open(raw_city_dir / (dataset_name_template % f"test_fixed_{size}" + ".pkl"), "wb") as f:
                        pickle.dump(instance, f)
            # generate additional features
            additional_feats = {}
            for feat in FEATS:
                additional_feats[feat] = feat.make_feat(rdf, graph, gdf_nodes)
            
            # And then generate dataset for training. By the way, why?
            generate_dataset(train_instance, additional_feats, dataset_name_template % "train", output_dir)
            generate_dataset(val_instance, additional_feats, dataset_name_template % "val", output_dir)
    pool.close()
