import numpy as np
from utils.utils import map_wrapper
from subprocess import check_call, DEVNULL, CalledProcessError

def try_call(argv, *args, **kwargs):
    try:
        check_call(argv, *args, **kwargs)
    except CalledProcessError as e:
        print("Failed Call: " + ' '.join(map(str, argv)))
        raise e

def write_instance(instance, instance_name, instance_file, write_special=False, zero_start=False):
    shift = 0 if zero_start else 1
    
    with open(instance_file, "w") as f:
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
        # Our weight matrix is redundant for LKH.
        # I know that LKH and HGS can only make use of integers, but I keep this intentionally.
        # But if we do not set the precision then the instance file would be very large...
        for line in instance["WEIGHT"][:instance["SIZE"]]:
            f.write(" ".join(map(lambda dec: f"{dec:.2f}", np.where(np.isinf(line), 0, line)[:instance["SIZE"]])) + "\n")
        if "DEMAND" in instance:
            f.write("DEMAND_SECTION\n")
            for i, demand in enumerate(instance["DEMAND"]):
                f.write(f"{i + shift} {demand}\n")
        if "DEPOT" in instance:
            f.write("DEPOT_SECTION\n " + str(instance["DEPOT"] + shift - 1) + "\n -1\n")
        if write_special and "SPECIAL" in instance:
            f.write(f"SPECIAL_SECTION\n{len(instance["SPECIAL"])} {" ".join(map(str, instance["SPECIAL"] + shift - 1))}\n")
        if "TIME_WINDOW_SECTION" in instance:
            f.write("TIME_WINDOW_SECTION\n")
            for i, (tw_begin, tw_end) in enumerate(instance["TIME_WINDOW_SECTION"]):
                f.write(f"{i + shift} {tw_begin} {tw_end}\n")
        f.write("EOF\n")

def write_para(feat_file, instance_file, method, para_file, candidate_set_type="nn",
                max_trials=1000, max_candidates=20, seed=1234):
    candidate_type_map = {
        "nn": "NEAREST-NEIGHBOR",
        "alpha": "ALPHA"
    }
    with open(para_file, "w") as f:
        f.write(f"PROBLEM_FILE = {instance_file}\n")
        f.write("PRECISION = 1\n")
        f.write(f"MAX_TRIALS = {max_trials}\n")
        f.write("SPECIAL\n")
        f.write("RUNS = 1\n")
        f.write(f"SEED = {seed}\n")
        if method == "LabelGen":
            # f.write("GerenatingFeature\n")
            # if os.path.exists(feat_file):
            #     os.remove(feat_file)
            # f.write(f"CANDIDATE_FILE = {feat_file}\n")
            # f.write(f"CANDIDATE_SET_TYPE = {candidate_type_map[candidate_set_type.lower()]}\n")
            f.write(f"MAX_CANDIDATES = {max_candidates}\n")
        elif method == "Model":
            if feat_file.exists():
                feat_file.unlink()
            f.write("SUBGRADIENT = NO\n")
            f.write(f"CANDIDATE_FILE = {feat_file}\n")
        elif method == "Kopt":
            pass
        else:
            assert method == "LKH"
            f.write(f"MAX_CANDIDATES = {max_candidates}\n")
            
def read_feat(feat_file, max_nodes, n_neighbours=20):
    edge_index = np.zeros([1, max_nodes, n_neighbours], dtype="int")
    with open(feat_file, "r") as f:
        lines = f.readlines()
        n_nodes_extend = int(lines[0].strip())
        for j in range(n_nodes_extend):
            line = lines[j + 1].strip().split(" ")
            line = [int(_) for _ in line]
            assert len(line) == n_neighbours * 2 + 3, f"See {feat_file}"
            assert line[0] == j + 1
            for _ in range(n_neighbours):
                edge_index[0, j, _] = line[3 + _ * 2] - 1
    feat_runtime = float(lines[-2].strip())
    return edge_index, n_nodes_extend, feat_runtime

def read_solution(log_file, feat_file, max_trials):
    with open(log_file, "r") as f:
        line = f.readlines()[-1]
        line = line.strip().split(" ")
        result = [int(_) for _ in line]
    # alpha_lists = []
    # with open(feat_file, "r") as f:
    #     n_nodes_extend = int(f.readline().strip())
    #     for _ in range(n_nodes_extend):
    #         parts = list(map(int.__call__, f.readline().strip().split()))
    #         alpha_lists.append(list(zip(parts[3::2], parts[4::2])))
    return result

def read_performance(log_file, _, max_trials):
    objs = []
    penalties = []
    runtimes = []
    with open(log_file, "r") as f:
        lines = f.readlines()
        for line in lines: # read the obj and runtime for each trial
            if line[:6] == "-Trial":
                line = line.strip().split(" ")
                if len(line) == 4: # The problem may degenerate to TSP if capacity is too large
                    assert len(objs) + 1 == int(line[-3]), str(log_file)
                    penalties.append(0)
                else:
                    assert len(objs) + 1 == int(line[-4]), str(log_file)
                    penalties.append(int(line[-3]))
                objs.append(int(line[-2]))
                runtimes.append(float(line[-1]))
        final_obj = int(lines[-11].split(",")[0].split(" ")[-1])
        assert objs[-1] == final_obj
        return objs, runtimes, penalties
    
# TODO: Remove
def write_candidate_CVRP(feat_file, candidate, n_nodes_extend, **unused):
    with open(feat_file, "w") as f:
        f.write(str(n_nodes_extend) + "\n")
        for j in range(n_nodes_extend):
            line = str(j + 1) + " 0 5"
            for _ in range(5):
                line += " " + str(int(candidate[j, _]) + 1) + " " + str(_ * 100)
            f.write(line + "\n")
        f.write("-1\nEOF\n")

def write_candidate_CVRPTW(feat_file, candidate, candidate2, **unused):
    candidate1 = candidate
    n_node = candidate1.shape[0] - 1 # n_node without depot
    with open(feat_file, "w") as f:
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
    assert task == "LabelGen" or task == "LKH" or task == "Model"
    para_file = param_dir / f"{instance_name}.para"
    log_file = log_dir /  f"{instance_name}.log" if log_dir else None
    instance_file = instance_dir / f"{instance_name}.cvrp"
    candidate_type = "alpha"
    candidate_file = candidate_dir / f"{instance_name}_{candidate_type}.txt" if candidate_dir else None
    if overwrite or not log_file.isfile():
        write_instance(instance, instance_name, instance_file, task == "LabelGen")
        write_para(candidate_file, instance_file, task, para_file, max_trials=max_trials, max_candidates=max_candidates, candidate_set_type=candidate_type)
        if candidate is not None:
            write_candidate_dispather[instance["TYPE"]](feat_file=candidate_file, candidate=candidate, candidate2=candidate2, n_nodes_extend=n_nodes)
        f = open(log_file, "w") if log_file else DEVNULL
        try_call(["./LKH", para_file], stdout=f)

    return result_hook(log_file, candidate_file, max_trials)

write_candidate_dispather = {
    "CVRP": write_candidate_CVRP,
    "CVRPTW": write_candidate_CVRPTW
}

def solve_kopt(instance, instance_name, expanded_node_num, param_dir, instance_dir, output_dir, mode="perf", node_weights=None, candidates=('alpha', 10), info_dir=None, max_trials=3000, seed=1234):
    para_file = param_dir / f"{instance_name}.para"
    instance_file = instance_dir / f"{instance_name}.cvrp"
    output_file = output_dir /  f"{instance_name}.npy"
    if mode == "perf":
        exe_path = "./zyclk"
        subsidiary_output = None
    elif mode == "perfsolve":
        exe_path = "./zyclk"
        subsidiary_output = output_dir / f"{instance_name}_tour.npy"
    elif mode == "log_perturb":
        exe_path = "./zyclk_log_perturb"
        subsidiary_output = output_dir / f"{instance_name}_edges.npy"
    else:
        raise RuntimeError(f"No such solve mode: {mode}")
    if type(candidates) is np.ndarray or type(candidates) is list:
        candidate_type = "external"
    elif type(candidates) is tuple:
        if len(candidates) != 2:
            raise RuntimeError(f"Invalid candidates type: {candidates}")
        candidate_type = candidates[0]
        candidate_count = candidates[1]
    else:
        raise RuntimeError(f"Fail to parse candidates {candidates}")
    
    write_instance(instance, instance_name, instance_file, write_special=True, zero_start=True)
    if candidate_type == "external":
        candidate_file = info_dir / f"{instance_name}.candidates"
        with candidate_file.open('w') as f:
            f.write(f"{expanded_node_num}\n")
            for node_candidates in candidates[:expanded_node_num]:
                f.write(" ".join(map(str, node_candidates)))
                f.write("\n")
    if mode != "log_perturb" and not node_weights is None:
        node_weights_file = info_dir / f"{instance_name}.weights"
        with node_weights_file.open('w') as f:
            f.write(" ".join(map(lambda weight: f"{weight:.5f}", node_weights[:expanded_node_num])))
            f.write("\n")
    with para_file.open('w') as f:
        f.write(f"problem_path = \"{instance_file}\"\n")
        f.write(f"candidate_type = \"{candidate_type}\"\n")
        if candidate_type == 'external':
            f.write(f"candidate_path = \"{candidate_file}\"\n")
        else:
            f.write(f"candidate_count = {candidate_count}\n")
        f.write(f"trial_limit = {max_trials}\n")
        if not node_weights is None:
            f.write(f"swap_weight_path = \"{node_weights_file}\"\n")
        if mode != "log_perturb":
            f.write(f"seed = {seed}\n")

    if subsidiary_output:
        try_call([exe_path, para_file, output_file, subsidiary_output])
        return np.load(output_file), np.load(subsidiary_output)
    else:
        try_call([exe_path, para_file, output_file])
        return np.load(output_file)
