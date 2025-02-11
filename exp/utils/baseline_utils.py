# Ura file
import numpy as np
from .instance_utils import write_instance
from utils import map_wrapper
from pyvrp import Model as HGSModel, read as HGS_read
from pyvrp.stop import MaxIterations, MaxRuntime, MultipleCriteria

# Helpers
def swap_1d(a, index1, index2):
    temp = a[index2]
    a[index2] = a[index1]
    a[index1] = temp

def swap_2d(a, index1, index2):
    assert a.shape[0] == a.shape[1]
    for index in range(a.shape[0]):
        swap_1d(a[index], index1, index2)
    for index in range(a.shape[0]):
        swap_1d(a[:, index], index1, index2)

# Warning: This may modify the instance.
def write_HGS_instance(instance, instance_name, instance_file):
    depot = instance["DEPOT"] - 1
    instance["DEPOT"] = 1
    swap_1d(instance["COORD"], 0, depot)
    swap_2d(instance["WEIGHT"], 0, depot)
    swap_1d(instance["GRAPH_INDEX"], 0, depot)
    swap_1d(instance["DEMAND"], 0, depot)
    instance["WEIGHT"][np.arange(instance["SIZE"]), np.arange(instance["SIZE"])] = 0
    write_instance(instance, instance_name, instance_file, False)


def solve_HGS(instance_file, max_iter=5000, max_runtime=3600):
    instance = HGS_read(instance_file)
    model = HGSModel.from_data(instance)
    stop_crit = MultipleCriteria([MaxIterations(max_iter), MaxRuntime(max_runtime)])
    result = model.solve(stop=stop_crit, seed=42, display=False)
    runtime =  np.cumsum(result.stats.runtimes)
    performance = list(map(lambda datum: datum.best_cost, result.stats.feas_stats))
    if len(runtime) < max_iter:
        # Pad up to max_iter
        pad_length = max_iter - len(runtime)
        runtime = np.pad(runtime, (0, pad_length), "constant", constant_values=runtime[-1])
        performance = performance + [performance[-1] for _ in range(pad_length)]
    return runtime, performance
