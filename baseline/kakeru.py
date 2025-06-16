from gurobi_impl import solve_cvrp as gurobi_solve
from ortools_impl import solve_cvrp as ortools_solve
from copt_impl import solve_cvrp as copt_solve
from pathlib import Path

if __name__ == '__main__':
    work_dir = Path('/home/tadshi/ot/NeuroLKH/exp')
    evaluation_dir = work_dir / 'evaluation'
    copt_solve(evaluation_dir / 'hgs' / '1.cvrp')
