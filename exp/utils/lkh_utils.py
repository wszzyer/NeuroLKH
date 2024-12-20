import os
import numpy as np
from feats import SSSPFeat
from utils.utils import map_wrapper
from subprocess import check_call, DEVNULL

def write_instance(instance, instance_name, instance_filename, n_nodes):
    with open(instance_filename, "w") as f:
        f.write("NAME : " + instance_name + "\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : " + instance["TYPE"] + "\n")
        f.write("DIMENSION : " + str(len(instance["COORD"])) + "\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        if "VEHICLES" in instance:
            f.write("VEHICLES : " + str(instance["VEHICLES"]) + "\n")
        if "CAPACITY" in instance:
            f.write("CAPACITY : " + str(instance["CAPACITY"]) + "\n")
        if "SERVICE_TIME" in instance:
            f.write("SERVICE_TIME : " + str(instance["SERVICE_TIME"]) + "\n" )
        f.write("EDGE_WEIGHT_SECTION\n")
        for line in instance["WEIGHT"]:
            f.write(" ".join(map(str, line)) + "\n")
        if "DEMAND" in instance:
            f.write("DEMAND_SECTION\n")
            for i, demand in enumerate(instance["DEMAND"]):
                f.write(f"{i+1} {demand}\n")
        if "DEPOT" in instance:
            f.write("DEPOT_SECTION\n " + str(instance["DEPOT"]) + "\n -1\n")
        if "TIME_WINDOW_SECTION" in instance:
            f.write("TIME_WINDOW_SECTION\n")
            for i, (tw_begin, tw_end) in enumerate(instance["TIME_WINDOW_SECTION"]):
                f.write(f"{i+1} {tw_begin} {tw_end}\n")
        f.write("EOF\n")

def write_para(feat_filename, instance_filename, method, para_filename, candidate_set_type="nn",
                max_trials=1000, max_candidates=20, seed=1234):
    candidate_type_map = {
        "nn": "NEAREST-NEIGHBOR",
        "alpha": "ALPHA"
    }
    with open(para_filename, "w") as f:
        f.write("PROBLEM_FILE = " + instance_filename + "\n")
        f.write("PRECISION = 1\n")
        f.write("MAX_TRIALS = " + str(max_trials) + "\n")
        f.write("SPECIAL\n")
        f.write("RUNS = 1\n")
        f.write("SEED = " + str(seed) + "\n")
        if method == "FeatGenerate":
            # f.write("GerenatingFeature\n")
            if os.path.exists(feat_filename):
                os.remove(feat_filename)
            f.write("CANDIDATE_FILE = " + feat_filename + "\n")
            f.write(f"CANDIDATE_SET_TYPE = {candidate_type_map[candidate_set_type.lower()]}\n")
            f.write(f"MAX_CANDIDATES = {max_candidates}\n")
        elif method == "Model":
            if os.path.exists(feat_filename):
                os.remove(feat_filename)
            f.write("SUBGRADIENT = NO\n")
            f.write("CANDIDATE_FILE = " + feat_filename + "\n")
        else:
            assert method == "LKH"
            if feat_filename:
                if os.path.exists(feat_filename):
                    os.remove(feat_filename)
                f.write("CANDIDATE_FILE = " + feat_filename + "\n")
                f.write(f"MAX_CANDIDATES = {max_candidates}\n")
            
def read_feat(feat_filename, max_nodes, n_neighbours=20):
    edge_index = np.zeros([1, max_nodes, n_neighbours], dtype="int")
    with open(feat_filename, "r") as f:
        lines = f.readlines()
        n_nodes_extend = int(lines[0].strip())
        for j in range(n_nodes_extend):
            line = lines[j + 1].strip().split(" ")
            line = [int(_) for _ in line]
            assert len(line) == n_neighbours * 2 + 3, f"See {feat_filename}"
            assert line[0] == j + 1
            for _ in range(n_neighbours):
                edge_index[0, j, _] = line[3 + _ * 2] - 1
    feat_runtime = float(lines[-2].strip())
    return edge_index, n_nodes_extend, feat_runtime

#TODO: generate alpha on ourselves.
def read_solution_and_alpha(log_filename, feat_filename, max_trials):
    with open(log_filename, "r") as f:
        line = f.readlines()[-1]
        line = line.strip().split(" ")
        result = [int(_) for _ in line]
    alpha_lists = []
    with open(feat_filename, "r") as f:
        n_nodes_extend = int(f.readline().strip())
        for _ in range(n_nodes_extend):
            parts = list(map(int.__call__, f.readline().strip().split()))
            alpha_lists.append(list(zip(parts[3::2], parts[4::2])))
    return result, alpha_lists

def read_performance(log_filename, _, max_trials):
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
    

def write_candidate_CVRP(feat_filename, candidate, n_nodes_extend, **unused):
    n_node = candidate.shape[0]
    with open(feat_filename, "w") as f:
        f.write(str(n_nodes_extend) + "\n")
        for j in range(n_nodes_extend):
            line = str(j + 1) + " 0 5"
            for _ in range(5):
                line += " " + str(int(candidate[j, _]) + 1) + " " + str(_ * 100)
            f.write(line + "\n")
        f.write("-1\nEOF\n")

def write_candidate_CVRPTW(feat_filename, candidate, candidate2, **unused):
    candidate1 = candidate
    n_node = candidate1.shape[0] - 1 # n_node without depot
    with open(feat_filename, "w") as f:
        f.write(str((n_node + 20) * 2) + "\n")
        line = "1 0 5 " + str(1 + n_node + 20) + " 0"
        for _ in range(4):
            line += " " + str(2 * n_node + 2 * 20 - _) + " 1"
        f.write(line + "\n")
        for j in range(1, n_node + 1):
            line = str(j + 1) + " 0 5 " + str(j + 1 + n_node + 20) + " 1"
            for _ in range(4):
                line += " " + str(candidate2[j, _] + 1 + n_node + 20) + " 1"
            f.write(line + "\n")
        for j in range(19):
            line = str(n_node + 1 + 1 + j) + " 0 5 " + str(n_node + 1 + 1 + j + n_node + 20) + " 0 " + str(1 + n_node + 20) + " 1"
            for _ in range(3):
                line += " " + str(n_node + 2 + _ + n_node + 20) + " 1" 
            f.write(line + "\n")
        
        line = str(1 + n_node + 20) + " 0 5 1 0"
        for _ in range(4):
            line += " " + str( n_node + 20 - _) + " 1"
        f.write(line + "\n")
        for j in range(1, n_node + 1):
            line = str(j + 1 + n_node + 20) + " 0 5 " + str(j + 1) + " 1"
            for _ in range(4):
                line += " " + str(candidate1[j, _] + 1) + " 1"
            f.write(line + "\n")
        for j in range(19):
            line = str(n_node + 2 + j + n_node + 20) + " 0 5 " + str(n_node + 2 + j) + " 0"
            for _ in range(4):
                line += " " + str(n_node + 20 - _) + " 1"
            f.write(line + "\n")
        f.write("-1\nEOF\n")


@map_wrapper
def solve_LKH(task, result_hook, instance_dir, param_dir, log_dir, instance, instance_name, max_candidates,
              overwrite=False, max_trials=1000, candidate_dir=None, candidate=None, candidate2=None, n_nodes=None):
    """
    solve LKH.
    """
    assert task == "LKH" or task == "Model"
    N_NODES = instance["COORD"].__len__() # this will be refactored.
    para_filename = os.path.join(param_dir, instance_name + ".para")
    log_filename = os.path.join(log_dir, instance_name + ".log") if log_dir else None
    instance_filename = os.path.join(instance_dir, instance_name + ".cvrp")
    candidate_type = "alpha"
    candidate_filename = os.path.join(candidate_dir, f"{instance_name}_{candidate_type}.txt") if candidate_dir else None
    if overwrite or not os.path.isfile(log_filename):
        write_instance(instance, instance_name, instance_filename, N_NODES)
        write_para(candidate_filename, instance_filename, task, para_filename, max_trials=max_trials, max_candidates=max_candidates, candidate_set_type=candidate_type)
        if candidate is not None:
            write_candidate_dispather[instance["TYPE"]](feat_filename=candidate_filename, candidate=candidate, candidate2=candidate2, n_nodes_extend=n_nodes)
        f = open(log_filename, "w") if log_filename else DEVNULL
        check_call(["./LKH", para_filename], stdout=f)

    return result_hook(log_filename, candidate_filename, max_trials)

write_candidate_dispather = {
    "CVRP": write_candidate_CVRP,
    "CVRPTW": write_candidate_CVRPTW
}