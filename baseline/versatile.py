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
    pool = mp.Pool(32)

    dataset_dir = Path(args.data_dir).resolve() / 'raw_instance'
    result_dir = Path(args.output_dir).resolve()
    if not result_dir.exists():
        raise RuntimeError('No More!')

    baselines = ['ORTools', 'Gurobi', 'COPT']
    for city_dir in dataset_dir.iterdir():
        if not city_dir.is_dir():
            continue
        city_name = city_dir.name
        output_dir = result_dir / city_name
        print(city_name)
        if not output_dir.exists():
            raise RuntimeError('Please run other baselines first!')
        output_file = output_dir / 'or_baselines.pkl'
        if output_file.exists():
            with output_file.open('rb') as f:
                record = pickle.load(f)
        else:
            record = {}
        for dataset_file in city_dir.iterdir():
            exp_name = path_to_name(dataset_file)
            if 'train' in exp_name:
                continue
            with dataset_file.open('rb') as f:
                dataset = pickle.load(f)
            if 'ORTools' not in record:
                record['ORTools'] = {}
            if exp_name not in record['ORTools']:
                record['ORTools'][exp_name] = np.stack(list(tqdm(pool.imap(ortools_impl.solve_cvrp, [{
                    'dimension': len(instance["COORD"]),
                    'capacity': instance["CAPACITY"],
                    'depot': instance["DEPOT"] - 1,
                    'demand': instance["DEMAND"],
                    'edge_weight': np.where(np.isinf(instance["WEIGHT"]), 0, instance["WEIGHT"])[:instance["SIZE"], :instance["SIZE"]],
                } for instance in dataset]), desc=f"{exp_name}(ORTools)", total=len(dataset)))).transpose(1, 0, 2)
                with output_file.open('wb') as f:
                    pickle.dump(record, f)
            exp_type = exp_name.split('_')[-1]
            if exp_type == 'raw' or int(exp_type) > 500:
                continue
            # if 'Gurobi' not in record:
            #     record['Gurobi'] = {}
            # if exp_name not in record['Gurobi']:
            #     performance_list = []
            #     for instance in tqdm(dataset, desc=f"{exp_name}(ORTools)"): # Gurobi is multithreaded by default
            #         init_tour = get_init_solution(instance)
            #         performance_list.append(gurobi_impl.solve_cvrp({
            #             'dimension': len(instance["COORD"]),
            #             'capacity': instance["CAPACITY"],
            #             'depot': instance["DEPOT"] - 1,
            #             'demand': instance["DEMAND"],
            #             'edge_weight': np.where(np.isinf(instance["WEIGHT"]), 0, instance["WEIGHT"])[:instance["SIZE"], :instance["SIZE"]],
            #         }, init_tour, max_runtime=180))
            #     record['Gurobi'][exp_name] = np.stack(performance_list).transpose(1, 0, 2)
            #     with output_file.open('wb') as f:
            #         pickle.dump(record, f)
    pool.close()
    