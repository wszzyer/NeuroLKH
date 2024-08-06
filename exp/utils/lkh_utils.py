import os
import numpy as np
from feats import SSSPFeat

def write_instance(instance, instance_name, instance_filename, n_nodes):
    with open(instance_filename, "w") as f:
        f.write("NAME : " + instance_name + "\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : " + instance["TYPE"] + "\n")
        f.write("DIMENSION : " + str(len(instance["COORD"])) + "\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        if "CAPACITY" in instance:
            f.write("CAPACITY : " + str(instance["CAPACITY"]) + "\n")
        if "SERVICE_TIME" in instance:
            f.write("SERVICE_TIME : " + str(instance["SERVICE_TIME"]) + "\n" )
        f.write("EDGE_WEIGHT_SECTION\n")
        for line in instance["ATTACHMENT"][SSSPFeat]:
            f.write(" ".join(map(str, line)) + "\n")
        if "DEMAND" in instance:
            f.write("DEMAND_SECTION\n")
            for i, demand in enumerate(instance["DEMAND"]):
                f.write(f"{i+1} {demand}\n")
        if "DEPOT" in instance:
            f.write("DEPOT_SECTION\n " + str(instance["DEPOT"]) + "\n -1\n")
        if "TIME_WINDOW_SECTION" in instance:
            f.write("TIME_WINDOW_SECTION\n")
            for i, tw_begin, tw_end in range(n_nodes):
                f.write(f"{i+1} {tw_begin} {tw_end}")
        f.write("EOF\n")

def write_para(feat_filename, instance_filename, method, para_filename, max_trials=1000, seed=1234):
    with open(para_filename, "w") as f:
        f.write("PROBLEM_FILE = " + instance_filename + "\n")
        f.write("PRECISION = 1\n")
        f.write("MAX_TRIALS = " + str(max_trials) + "\n")
        f.write("SPECIAL\nRUNS = 1\n")
        f.write("SEED = " + str(seed) + "\n")
        if method == "FeatGenerate":
            # f.write("GerenatingFeature\n")
            if os.path.exists(feat_filename):
                os.remove(feat_filename)
            f.write("CANDIDATE_FILE = " + feat_filename + "\n")
            f.write("CANDIDATE_SET_TYPE = NEAREST-NEIGHBOR\n")
            f.write("MAX_CANDIDATES = 20\n")
        else:
            assert method == "LKH"
            
def read_feat(feat_filename, max_nodes):
    n_neighbours = 20
    edge_index = np.zeros([1, max_nodes, n_neighbours], dtype="int")
    with open(feat_filename, "r") as f:
        lines = f.readlines()
        n_nodes_extend = int(lines[0].strip())
        for j in range(n_nodes_extend):
            line = lines[j + 1].strip().split(" ")
            line = [int(_) for _ in line]
            assert len(line) == 43
            assert line[0] == j + 1
            for _ in range(n_neighbours):
                edge_index[0, j, _] = line[3 + _ * 2] - 1
    return edge_index, n_nodes_extend

def read_results(log_filename, max_trials):
    with open(log_filename, "r") as f:
        line = f.readlines()[-1]
        line = line.strip().split(" ")
        result = [int(_) for _ in line]
    return result
