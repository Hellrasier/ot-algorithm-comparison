import gc
import numpy as np
import pandas as pd
from functools import partial
from tqdm import tqdm
from uot.core.problem_gen import get_problemset, OTProblem



def run_experiment(experiment: 'Experiment',
                   jit_algorithms=None,
                   solvers: dict[str, callable] = None,
                   problemsets_names: list[tuple] = None,
                   folds: int = 1) -> pd.DataFrame:
    """
    Executes a series of experiments using specified solvers on a set of problems.
    Args:
        experiment (Experiment): The experiment object that defines how to run the experiments.
        solvers (dict[str, callable]): A dictionary where keys are solver names and values are tuples 
            containing the solver function and optional sets of keyword arguments for the solver.
        problemsets_names (list[tuple]): A list of problem set names to retrieve and use in the experiments.
        jit_algorithms (list[str], optional): A list of algorithm names to exclude from the results. Defaults to None.
        folds (int, optional): The number of folds for cross-validation. Defaults to 1.
    Returns:
        pd.DataFrame: A DataFrame containing the results of the experiments, including solver names 
        and any additional parameters used.
    """

    problem_sets = [get_problemset(name, number=4) for name in problemsets_names]
    problems = [problem for problemset in problem_sets for problem in problemset]

    problems *= folds

    for problem in problems:
        source_distribution = problem.source_measure.distribution
        target_distribution = problem.target_measure.distribution
        
        if np.any(np.logical_or(np.isnan(source_distribution), np.isinf(source_distribution))):
            print(f"Detected problem in: {problem}")
            print("Source distribution contains NaN or Inf")
            print(source_distribution)
            raise ValueError("Nan or Inf in source distributions")
        
        if np.any(np.logical_or(np.isnan(target_distribution), np.isinf(target_distribution))):
            print(f"Detected problem in: {problem}")
            print("Source distribution contains NaN or Inf")
            print(target_distribution)
            raise ValueError("Nan or Inf in target distributions")
    
    dfs = []

    solvers_number = sum(len(kwargs) if kwargs else 1 for _, kwargs in solvers.values())

    with tqdm(total=solvers_number * len(problems), desc="Running experiments") as pbar:
        progress_callback = lambda: pbar.update(1)

        for solver_name, solver in solvers.items():
            
            solver_function, kwargs_sets = solver
            kwargs_sets = kwargs_sets if kwargs_sets else [{}]
            solvers = [(partial(solver_function, **kwargs), kwargs) for kwargs in kwargs_sets]

            for solver, kwargs in solvers:
                pbar.set_description(f"Solver: {solver_name}({kwargs})")
                solver_result = experiment.run_experiment(ot_problems=problems, progress_callback=progress_callback, solver=solver)

                solver_result['name'] = solver_name
                
                for kwarg_name, value in kwargs.items():
                    solver_result[kwarg_name] = value
            
            dfs.append(solver_result)

    df = pd.concat(dfs)

    for dataset in df.dataset.unique():
        for algorithm_name in jit_algorithms:
            algorithm_results = df[(df.dataset == dataset) & (df.name == algorithm_name)]
            if len(algorithm_results):
                df.drop(algorithm_results.index[0], inplace=True)

    return df


class Experiment:

    def __init__(self, name: str, run_function: callable):
        self.name = name
        self.run_function = run_function

    def run_experiment(self, ot_problems: list[OTProblem], progress_callback: callable = None, solver = None) -> dict:
        results = {}
        for i, ot_problem in enumerate(ot_problems):

            run_function = partial(self.run_function, solver=solver)

            results[ot_problem] = run_function(ot_problem)
            if progress_callback is not None:
                progress_callback()
            ot_problem.free_memory()
            if i % 100 == 0:
                gc.collect()

        df_rows = []
        for ot_problem in ot_problems:
            row_dict = ot_problem.to_dict() 
            row_dict.update(results[ot_problem])
            df_rows.append(row_dict)

        return pd.DataFrame(df_rows)
    
    def run_single(self, ot_problem: OTProblem) -> dict:
        return self.run_function(ot_problem)
