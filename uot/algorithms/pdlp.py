from jax import numpy as jnp
from .rapdhg import raPDHG, create_ot_problem
from .rapdhg.utils import RestartScheme


def solve_pdlp(a, b, C, epsilon=0.01, precision=1e-4, max_iters=10_000):
    solver = raPDHG(
        verbose=False,
        jit=True,
        reg=float(epsilon),
        eps_abs=precision,
        eps_rel=precision,
        iteration_limit=max_iters,
        termination_evaluation_frequency=100
    )
    problem = create_ot_problem(C, a, b)
    result, _ = solver.optimize(problem)

    converged = result.termination_status == 2
    cost = jnp.sum(problem.cost_matrix * result.primal_solution)

    return result.primal_solution, cost, converged