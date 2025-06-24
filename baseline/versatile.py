import gurobi_impl
import ortools_impl
from pathlib import Path
import argparse
import numpy as np
import pickle

import multiprocessing as mp
from tqdm import tqdm
from solution import get_init_solution

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', type=str, default='data/generated/', help='')
    parser.add_argument('--output_dir', type=str, default='result/', help='')
    return parser.parse_args()

def path_to_name(path: Path):
    return '_'.join(str(path.stem).split('_')[1:][:-4])

if __name__ == '__main__':
    args =  get_args()
    pool = mp.Pool(20)

    dataset_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        raise RuntimeError('No More!')

    if (output_dir / 'baseline.pkl').exists():
        with (output_dir / 'baseline.pkl').open('rb') as f:
            record = pickle.load(f)
    else:
        record = {}

    # baselines = ['ORTools', 'Gurobi', 'COPT']
    baselines = ['Gurobi', 'COPT']
    for dataset_file in dataset_dir.iterdir():
        exp_name = path_to_name(dataset_file)
        if 'train' in exp_name or '1000' in exp_name:
            continue
        with dataset_file.open('rb') as f:
            dataset = pickle.load(f)
        for baseline in baselines:
            if not baseline in record:
                record[baseline] = {}
        if 'ORTools' in baselines and  exp_name not in record['ORTools']:
            record['ORTools'][exp_name] = np.stack(list(tqdm(pool.imap(ortools_impl.solve_cvrp, [{
                'dimension': len(instance["COORD"]),
                'capacity': instance["CAPACITY"],
                'depot': instance["DEPOT"] - 1,
                'demand': instance["DEMAND"],
                'edge_weight': np.where(np.isinf(instance["WEIGHT"]), 0, instance["WEIGHT"])[:instance["SIZE"], :instance["SIZE"]],
            } for instance in dataset]), desc=exp_name, total=len(dataset)))).transpose(1, 0, 2)
            with (output_dir / 'baseline.pkl').open('wb') as f:
                pickle.dump(record, f)
        if 'Gurobi' in baselines and  exp_name not in record['Gurobi']:
            performance_list = []
            for instance in dataset: # Gurobi is multithreaded by default
                init_tour = get_init_solution(instance)
                performance_list.append(gurobi_impl.solve_cvrp({
                    'dimension': len(instance["COORD"]),
                    'capacity': instance["CAPACITY"],
                    'depot': instance["DEPOT"] - 1,
                    'demand': instance["DEMAND"],
                    'edge_weight': np.where(np.isinf(instance["WEIGHT"]), 0, instance["WEIGHT"])[:instance["SIZE"], :instance["SIZE"]],
                }, init_tour))
            record['Gurobi'][exp_name] = np.stack(performance_list).transpose(1, 0, 2)
            with (output_dir / 'baseline.pkl').open('wb') as f:
                pickle.dump(record, f)
    pool.close()
    