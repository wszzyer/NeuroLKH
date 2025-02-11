from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import numpy as np
from utils.instance_utils import read_performance, solve_LKH, write_instance
from multiprocessing import Pool
import argparse
import pickle
from pathlib import Path
from functools import partial
from utils.baseline_utils import solve_HGS, write_HGS_instance

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--problem", type=str, default="CVRP", choices=["TSP", "CVRP", "CVRPTW", "PDP"], help="which problem")
    parser.add_argument('--data_path', type=str, default='data/generated/CVRP_val_scatter_yt_111_100.pkl', help='')
    parser.add_argument('--baselines', type=str, action='extend', default=['hgs'], nargs='+', help='')
    parser.add_argument('--num_candidates', type=int, default=20, help='')
    parser.add_argument("--num_cpus", type=int, default=32, help="num cpus POOL")
    parser.add_argument('--work_dir', type=str, default="./evaluation", help='')
    parser.add_argument('--output_file', type=str, default='a.out', help='')
    parser.add_argument('--num_trials', type=int, default=30000, help='')
    return parser.parse_args()

def sure_path(path: Path):
    if path.exists():
        if path.parent.parts[-1] != "evaluation":
            raise RuntimeError(f"Danger! Cant afford to remove {path}")
        for root, dirs, files in path.walk(top_down=False):
            for name in files:
                (root / name).unlink()
            for name in dirs:
                (root / name).rmdir()
    path.mkdir(exist_ok=True)
    return path

def eval_lkh(dataset_path, work_dir, max_candidates, max_trials, pool=None, ignore_cache=True):
    LKH_param_dir = work_dir / "param"
    LKH_log_dir = work_dir / "log"
    LKH_param_dir.mkdir(exist_ok=True)
    LKH_log_dir.mkdir(exist_ok=True)
    instance_dir = work_dir / "instance"

    if pool:
        pmap = partial(process_map, chunksize=4, max_workers=pool)
    else:
        pmap = map

    with dataset_path.open("rb") as f:
        dataset = pickle.load(f)

    results = list(tqdm(pmap(solve_LKH, [("LKH", read_performance, instance_dir, LKH_param_dir, LKH_log_dir, dataset[i], str(i), max_candidates,
                                                ignore_cache, max_trials) for i in range(len(dataset))]), total=len(dataset), desc='Solving problem with LKH'))
    results = np.array(results).transpose(1, 0, 2)
    return results


def eval_hgs(dataset_path, work_dir, pool=None):
    instance_dir = work_dir
    with dataset_path.open("rb") as f:
        dataset = pickle.load(f)
    for id, raw_instance in enumerate(tqdm(dataset, desc="Writing HGS Instances")):
        write_HGS_instance(raw_instance, str(id), instance_dir / f"{id}.cvrp")
    if pool:
        pmap = partial(process_map, max_workers=pool, total=len(dataset), desc='Solving problem with HGS')
    else:
        pmap = map
    results = list(pmap(solve_HGS, instance_dir.iterdir()))
    results = np.array(results).transpose(1, 0, 2)
    return results

if __name__ == "__main__":
    args = get_args()
    pool = Pool(args.num_cpus)
    dataset_path = Path(args.data_path).resolve()
    work_dir = Path(args.work_dir).resolve()
    output_file = Path(args.output_file).resolve()
    if output_file.exists():
        with output_file.open(mode="rb") as file:
            result = pickle.load(file)
            existing_keys = set(result.keys())
    else:
        existing_keys = []
        result = {}

    print(f'Evaluating {dataset_path.stem}')
    if 'lkh' in args.baselines and 'LKH' not in existing_keys:
        result['LKH'] = eval_lkh(dataset_path, sure_path(work_dir / "lkh"), args.num_candidates, args.num_trials, pool=pool, ignore_cache=True)
    if 'hgs' in args.baselines and 'HGS' not in existing_keys:
        result['HGS'] = eval_hgs(dataset_path, sure_path(work_dir / "hgs"), pool=args.num_cpus)
    with output_file.open(mode='wb') as file:
        pickle.dump(result, file)
    pool.close()
