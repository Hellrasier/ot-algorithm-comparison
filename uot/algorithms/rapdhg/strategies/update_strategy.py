import jax
import jax.numpy as jnp
from functools import partial
from enum import IntEnum
from typing import Callable, Tuple

from ..utils import (
    QuadraticProgrammingProblem,
    PdhgSolverState
)

from ..loop_utils import while_loop
from ..utils import OTProblem, ot_apply_A


def compute_next_solution(
    problem: OTProblem,
    solver_state: PdhgSolverState,
    step_size: float,
    extrapolation_coefficient: float,
):
    """Compute the next primal and dual solutions.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The quadratic programming problem.
    solver_state : PdhgSolverState
        The current state of the solver.
    step_size : float
        The step size used in the PDHG algorithm.
    extrapolation_coefficient : float
        The extrapolation coefficient.

    Returns
    -------
    tuple
        The delta primal, delta primal product, and delta dual.
    """
    # Compute the next primal solution.

    primal_step =  step_size / solver_state.primal_weight

    next_primal_solution = (solver_state.current_primal_solution - primal_step * (
        problem.cost_matrix - solver_state.current_dual_product
    )) / (1 + primal_step * solver_state.reg)

    #Projection.
    next_primal_solution = jnp.clip(next_primal_solution, 0.0)

    delta_primal = next_primal_solution - solver_state.current_primal_solution
    delta_primal_product = ot_apply_A(delta_primal)

    # Compute the next dual solution.
    next_dual_solution = solver_state.current_dual_solution + (
        solver_state.primal_weight * step_size
    ) * (
        problem.marginals
        - (1 + extrapolation_coefficient) * delta_primal_product
        - solver_state.current_primal_product
    )

    delta_dual_solution = next_dual_solution - solver_state.current_dual_solution
    return delta_primal, delta_primal_product, delta_dual_solution

def line_search(
    compute_next_solution, problem, solver_state, reduction_exponent, growth_exponent, step_size_limit_coef
):
    """Perform a line search to find a good step size.

    Parameters
    ----------
    compute_next_solution : Callable
    problem : QuadraticProgrammingProblem
        The quadratic programming problem.
    solver_state : PdhgSolverState
        The current state of the solver.
    reduction_exponent : float
        The reduction exponent for adaptive step size.
    growth_exponent : float
        The growth exponent for adaptive step size.
    step_size_limit_coef: float
        The step size limit coefficient for adaptive step size.

    Returns
    -------
    tuple
        The delta_primal, delta_dual, delta_primal_product, step_size, and line_search_iter.
    """

    def cond_fun(line_search_state):
        step_size_limit = line_search_state[4]
        old_step_size = line_search_state[6]
        return jax.lax.cond(
            old_step_size <= step_size_limit,
            lambda _: False,
            lambda _: True,
            operand=None,
        )

    def body_fun(line_search_state):
        line_search_iter = line_search_state[0]
        step_size_limit = line_search_state[4]
        step_size = line_search_state[5]
        line_search_iter += 1
        delta_primal, delta_primal_product, delta_dual = compute_next_solution(
            problem, solver_state, step_size, 1.0
        )
        interaction = jnp.abs(jnp.dot(delta_primal_product.ravel(), delta_dual.ravel()))
        movement = 0.5 * solver_state.primal_weight * jnp.sum(
            jnp.square(delta_primal)
        ) + (0.5 / solver_state.primal_weight) * jnp.sum(jnp.square(delta_dual))

        step_size_limit = jax.lax.cond(
            interaction > 0,
            lambda _: movement / interaction * step_size_limit_coef,
            lambda _: jnp.inf,
            operand=None,
        )
        old_step_size = step_size
        first_term = (
            1
            - 1
            / (solver_state.num_steps_tried + line_search_iter + 1)
            ** reduction_exponent
        ) * step_size_limit
        second_term = (
            1
            + 1
            / (solver_state.num_steps_tried + line_search_iter + 1) ** growth_exponent
        ) * step_size
        step_size = jnp.minimum(first_term, second_term)

        return (
            line_search_iter,
            delta_primal,
            delta_dual,
            delta_primal_product,
            step_size_limit,
            step_size,
            old_step_size,
        )

    (
        line_search_iter,
        delta_primal,
        delta_dual,
        delta_primal_product,
        step_size_limit,
        step_size,
        old_step_size,
    ) = while_loop(
        cond_fun,
        body_fun,
        init_val=(
            0,
            jnp.zeros_like(solver_state.current_primal_solution),
            jnp.zeros_like(solver_state.current_dual_solution),
            jnp.zeros_like(solver_state.current_dual_solution),
            -jnp.inf,
            solver_state.step_size,
            solver_state.step_size,
        ),
        maxiter=10,
        unroll=False,
        jit=True,
    )

    return delta_primal, delta_dual, delta_primal_product, step_size, line_search_iter



def calculate_constant_step_size(
    primal_weight, iteration, last_step_size, reg, norm_A
) -> float:
    norm_reg = jnp.max(reg)
    next_step_size = (
        0.99
        * (2 + iteration)
        / (
            norm_reg / primal_weight
            + jnp.sqrt(
                (2 + iteration) ** 2 * norm_A**2
                + norm_reg**2 / primal_weight**2
            )
        )
    )
    # We use jnp.true_divide here since it returns inf for division by zero.
    step_size_limit = (1 + jnp.true_divide(1, iteration)) * last_step_size
    return jnp.minimum(next_step_size, step_size_limit)

# # ---- helper used by kernels ----
# def _static_one():
#     return 0

# ---- KERNEL 0 : keep step size, advance once ----
def _kernel_keep(compute_next_solution, problem, state):
    extrap = state.solutions_count / (state.solutions_count + 1.0)
    Δx, ΔAx, Δy = compute_next_solution(problem, state, state.step_size, extrap)
    return Δx, Δy, ΔAx, state.step_size, 0

# ---- KERNEL 1 : compute new constant step size, advance once ----
def _kernel_const(compute_next_solution, problem, state, reg, norm_A):
    step_size = calculate_constant_step_size(
        state.primal_weight, state.solutions_count, state.step_size, reg, norm_A
    )
    extrap = state.solutions_count / (state.solutions_count + 1.0)
    Δx, ΔAx, Δy = compute_next_solution(problem, state, step_size, extrap)
    return Δx, Δy, ΔAx, step_size, 0

# ---- KERNEL 2 : adaptive back‑tracking ----
def _kernel_linesearch(
    compute_next_solution,
    problem,
    state,
    reduction_exp,
    growth_exp,
    limit_coef,
):
    return line_search(
        compute_next_solution,
        problem,
        state,
        reduction_exp,
        growth_exp,
        limit_coef,
    )

class UpdateStrategy(IntEnum):
    CONSTANT_KEEP       = 0   # keep previous step_size
    CONSTANT_COMPUTED   = 1   # compute once from weights & reg
    ADAPTIVE_LINESEARCH = 2   # back‑tracking line‑search

@partial(jax.jit, static_argnums=0)       # strategy is compile‑time constant
def advance_iterate(
    # compute_next_solution: Callable,
    strategy: UpdateStrategy,
    problem,
    solver_state,
    *,                       # keyword‑only below
    reg,                     # used by CONSTANT_COMPUTED
    reduction_exp     = 0.5, # ↓ used by ADAPTIVE_LINESEARCH
    growth_exp        = 1.5,
    limit_coef        = 2.0,
    norm_A            = 1.0
):
    """
    Dispatch to the chosen update strategy and return:
        (delta_primal, delta_dual, delta_primal_product,
         new_step_size, line_search_iter)
    """
    kernels = (
        lambda _: _kernel_keep(compute_next_solution, problem, solver_state),
        lambda _: _kernel_const(compute_next_solution, problem, solver_state, reg=reg, norm_A=norm_A),
        lambda _: _kernel_linesearch(
            compute_next_solution,
            problem,
            solver_state,
            reduction_exp=reduction_exp,
            growth_exp=growth_exp,
            limit_coef=limit_coef,
        ),
    )
    return jax.lax.switch(int(strategy), kernels, operand=None)
