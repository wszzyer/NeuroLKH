from tqdm import tqdm
import numpy as np
from utils.lkh_utils import read_performance, solve_LKH
from multiprocessing import Pool
import argparse
import pickle
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--problem", type=str, default="CVRP", choices=["TSP", "CVRP", "CVRPTW", "PDP"], help="which problem")
    parser.add_argument('--data_path', type=str, default='data/generated/CVRP_val_scatter_yt_111_100.pkl', help='')
    parser.add_argument('--num_candidates', type=int, default=20, help='')
    parser.add_argument("--num_cpus", type=int, default=32, help="num cpus POOL")
    parser.add_argument('--work_dir', type=str, default="./evaluation", help='')
    parser.add_argument('--output_file', type=str, default='a.out', help='')
    parser.add_argument('--num_trials', type=int, default=30000, help='')
    return parser.parse_args()

def eval_lkh(dataset, work_dir, max_candidates, max_trials, pool=None, ignore_cache=True):
    instance_dir = work_dir / "instance"
    LKH_param_dir = work_dir / "param"
    LKH_log_dir = work_dir / "log"
    instance_dir.mkdir(parents=True, exist_ok=True)
    LKH_param_dir.mkdir(exist_ok=True)
    LKH_log_dir.mkdir(exist_ok=True)

    if pool:
        pmap = pool.imap
    else:
        pmap = map

    results = list(tqdm(pmap(solve_LKH, [("LKH", read_performance, instance_dir, LKH_param_dir, LKH_log_dir, dataset[i], str(i), max_candidates,
                                                ignore_cache, max_trials) for i in range(len(dataset))]), total=len(dataset), desc='Acquiring LKH Result'))
    results = np.array(results).transpose(1, 0, 2)
    return results

if __name__ == "__main__":
    args = get_args()
    pool = Pool(args.num_cpus)
    dataset_path = Path(args.data_path).resolve()
    with dataset_path.open("rb") as f:
        dataset = pickle.load(f)
    work_dir = Path(args.work_dir).resolve()
    lkh_result = eval_lkh(dataset, work_dir, args.num_candidates, args.num_trials, pool=pool, ignore_cache=True)
    
    file = open(args.output_file, mode='wb')
    pickle.dump(lkh_result, file)
