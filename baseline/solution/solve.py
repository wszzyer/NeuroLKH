from subprocess import check_call
from pathlib import Path
import numpy as np

WORK_DIR = Path(__file__).resolve().parent / 'temp'
if not WORK_DIR.exists():
    WORK_DIR.mkdir()
    
def get_init_solution(instance):
    with (WORK_DIR / 'init.cvrp').open('w') as f:
        f.write(f"NAME : init\n")
        f.write("TYPE : CVRP\n")
        f.write(f"DIMENSION : {len(instance["COORD"])}\n")
        f.write(f"CAPACITY : {instance["CAPACITY"]}\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for line in instance["WEIGHT"][:instance["SIZE"]]:
            f.write(" ".join(map(lambda dec: f"{dec:.2f}", np.where(np.isinf(line), 0, line)[:instance["SIZE"]])) + "\n")
        f.write("DEMAND_SECTION\n")
        for i, demand in enumerate(instance["DEMAND"]):
            f.write(f"{i} {demand}\n")
        if "DEPOT" in instance:
            f.write(f"DEPOT_SECTION\n{instance["DEPOT"] - 1}\n -1\n")
        f.write(f"SPECIAL_SECTION\n{len(instance["SPECIAL"])} {" ".join(map(str, instance["SPECIAL"]))}\n")
        f.write("EOF\n")
    check_call([WORK_DIR / 'zyclk_first_solution', WORK_DIR / 'init.toml', WORK_DIR / 'init.perf', WORK_DIR / 'init.tour'])
    tour = np.load(WORK_DIR / 'init.tour')
    # Roll the tour to the depot
    for index, node in enumerate(tour):
        if node == instance['DEPOT'] - 1:
            tour = np.roll(tour, -index)
            break
    return tour

