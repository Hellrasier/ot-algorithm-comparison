import time
import numpy as np
from uot.algorithms.sinkhorn import sinkhorn
from memory_profiler import memory_usage, profile
from uot.core.experiment import OTProblem, Experiment

def time_precision_experiment_func(ot_problem: OTProblem, solver: callable = sinkhorn):
    a, b, C = ot_problem.to_jax_arrays()
    start_time = time.perf_counter()
    result = solver(a, b, C)
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    output_T, output_dist, converged = result
    end_time = time.perf_counter()

    exact_T, exact_dist = ot_problem.exact_map, ot_problem.exact_cost

    # precision = np.abs(output_dist - exact_dist) / exact_dist
    # coupling_precision = np.sum(np.abs(output_T - exact_T)) / np.prod(output_T.shape)

    precision = np.abs(output_dist - exact_dist)
    coupling_precision = np.linalg.norm(output_T - exact_T)

    return {'time': (end_time - start_time) * 1000, 'cost_rerr': precision, 'coupling_avg_err': coupling_precision, 'converged': converged}


def time_experiment_func(ot_problem: OTProblem, solver: callable = sinkhorn):
    a, b, C = ot_problem.to_jax_arrays()
    start_time = time.perf_counter()
    _, _ = solver(a, b, C)
    end_time = time.perf_counter()

    return {'time': (end_time - start_time) * 1000}


# Requires if __name__ == '__main__' to work properly
def memory_experiment(ot_problem: OTProblem, solver: callable = sinkhorn):
    a, b, C = ot_problem.to_jax_arrays()

    mem_usage = memory_usage((solver, (a, b, C)), interval=1e-3)

    return {'memory': max(mem_usage)}


def memory_profiler_out(ot_problem: OTProblem, solver: callable = sinkhorn):
    a, b, C = ot_problem.to_jax_arrays()

    profiled_solver = profile(solver)
    profiled_solver(a, b, C)


time_precision_experiment = Experiment("Measure time and precision", run_function=time_precision_experiment_func)

time_experiment = Experiment("Measure time", run_function=time_experiment_func)


