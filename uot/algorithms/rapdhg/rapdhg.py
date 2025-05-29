import abc
import logging
import timeit
from dataclasses import dataclass
from typing import Any, Tuple, Optional

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCSR, BCOO

from .loop_utils import while_loop, while_loop_iters
from .preprocess import rescale_problem
from .strategies.update_strategy import (
    UpdateStrategy,
    advance_iterate
)
from .strategies.reg_strategy import (
    RegStrategy, compute_reg
)
from .restart import (
    run_restart_scheme,
    select_initial_primal_weight,
    unscaled_saddle_point_output,
)
from .solver_log import (
    display_iteration_stats_heading,
    pdhg_final_log,
    setup_logger,
)
from .termination import (
    check_termination_criteria,
)
from .utils import (
    PdhgSolverState,
    QuadraticProgrammingProblem,
    RestartInfo,
    RestartParameters,
    RestartScheme,
    RestartToCurrentMetric,
    SaddlePointOutput,
    TerminationCriteria,
    TerminationStatus,
    ScaledQpProblem,
    ConvergenceInformation, cut_dict_at_first_zero,
)
from .iteration_stats_utils import evaluate_unscaled_iteration_stats
from .utils import OTProblem, ot_apply_A, ot_apply_AT

logger = logging.getLogger(__name__)


def estimate_maximum_singular_value(
    matrix: BCSR,
    probability_of_failure: float = 0.01,
    desired_relative_error: float = 0.1,
    seed: int = 1,
) -> tuple:
    """
    Estimate the maximum singular value of a sparse matrix using the power method.

    Parameters
    ----------
    matrix : BCSR
        The sparse matrix in BCSR format.
    probability_of_failure : float, optional
        The acceptable probability of failure.
    desired_relative_error : float, optional
        The desired relative error for the estimation.
    seed : int, optional
        The random seed for reproducibility.

    Returns
    -------
    tuple
        The estimated maximum singular value and the number of power iterations.
    """
    epsilon = 1.0 - (1.0 - desired_relative_error) ** 2
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, (matrix.shape[1],))
    if isinstance(matrix, BCSR):
        matrix_transpose = BCSR.from_bcoo(matrix.to_bcoo().T)
    elif isinstance(matrix, BCOO):
        matrix_transpose = BCSR.from_bcoo(matrix.T)
    elif isinstance(matrix, jnp.ndarray):
        matrix_transpose = matrix.T
    number_of_power_iterations = 0

    def cond_fun(state):
        # Corresponds to the power_method_failure_probability in CuPDLP.jl
        x, number_of_power_iterations = state
        power_method_failure_probability = jax.lax.cond(
            # We have to use bitwise operators | instead of or in JAX.
            # Ref: https://github.com/jax-ml/jax/issues/3761#issuecomment-658456938
            (number_of_power_iterations < 2) | (epsilon <= 0.0),
            lambda _: 1.0,
            lambda _: (
                jax.lax.min(
                    0.824, 0.354 / jnp.sqrt(epsilon * (number_of_power_iterations - 1))
                )
                * jnp.sqrt(matrix.shape[1])
                * (1.0 - epsilon) ** (number_of_power_iterations - 0.5)
            ),
            operand=None,
        )
        return power_method_failure_probability > probability_of_failure

    def body_fun(state):
        x, number_of_power_iterations = state
        x = x / jnp.linalg.norm(x, 2)
        x = matrix_transpose @ (matrix @ x)
        return x, number_of_power_iterations + 1

    # while_loop() compiles cond_fun and body_fun, so while it can be combined with jit(), it’s usually unnecessary.
    x, number_of_power_iterations = while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=(x, 0),
        maxiter=1000,
        unroll=False,
        jit=True,
    )
    return (
        jnp.sqrt(
            jnp.dot(x, matrix_transpose @ (matrix @ x)) / jnp.linalg.norm(x, 2) ** 2
        ),
        number_of_power_iterations,
    )

@dataclass(eq=False)
class raPDHG(abc.ABC):
    """
    The raPDHG solver class.
    """

    verbose: bool = False
    debug: bool = False
    display_frequency: int = 10
    jit: bool = True
    unroll: bool = False
    termination_evaluation_frequency: int = 64
    optimality_norm: float = 2
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4
    eps_primal_infeasible: float = 1e-8
    eps_dual_infeasible: float = 1e-8
    reg: jnp.ndarray = 0.0
    # time_sec_limit: float = float("inf")
    iteration_limit: int = jnp.iinfo(jnp.int32).max
    l_inf_ruiz_iterations: int = 10
    l2_norm_rescaling: bool = False
    pock_chambolle_alpha: float = 1.0
    primal_importance: float = 1.0
    step_size: Optional[float] = 0.1
    scale_invariant_initial_primal_weight: bool = False
    restart_scheme: int = RestartScheme.ADAPTIVE_KKT
    restart_to_current_metric: int = RestartToCurrentMetric.KKT_GREEDY
    restart_frequency_if_fixed: int = 1000
    artificial_restart_threshold: float = 0.36
    sufficient_reduction_for_restart: float = 0.2
    necessary_reduction_for_restart: float = 0.8
    primal_weight_update_smoothing: float = 0.5
    adaptive_step_size: bool = True
    adaptive_step_size_reduction_exponent: float = 0.3
    adaptive_step_size_growth_exponent: float = 0.6
    adaptive_step_size_limit_coef: float = 1.0
    warm_start: bool = False
    feasibility_polishing: bool = False
    eps_feas_polish: float = 1e-06
    infeasibility_detection: bool = True
    reference_solution: Optional[jnp.ndarray] = None
    save_iters: bool = False
    reg_strategy: RegStrategy = RegStrategy.CONSTANT
    cool_down_param: float = 1.0
    cool_down_threshold: float = 1e-6
    update_strategy: UpdateStrategy = UpdateStrategy.ADAPTIVE_LINESEARCH

    def check_config(self):
        self._termination_criteria = TerminationCriteria(
            eps_abs=self.eps_abs,
            eps_rel=self.eps_rel,
            eps_primal_infeasible=self.eps_primal_infeasible,
            eps_dual_infeasible=self.eps_dual_infeasible,
            iteration_limit=self.iteration_limit,
        )
        self._restart_params = RestartParameters(
            restart_scheme=self.restart_scheme,
            restart_to_current_metric=self.restart_to_current_metric,
            restart_frequency_if_fixed=self.restart_frequency_if_fixed,
            artificial_restart_threshold=self.artificial_restart_threshold,
            sufficient_reduction_for_restart=self.sufficient_reduction_for_restart,
            necessary_reduction_for_restart=self.necessary_reduction_for_restart,
            primal_weight_update_smoothing=self.primal_weight_update_smoothing,
        )
        self._polishing_termination_criteria = TerminationCriteria(
            eps_abs=self.eps_feas_polish,
            eps_rel=self.eps_feas_polish,
            eps_primal_infeasible=self.eps_primal_infeasible,
            eps_dual_infeasible=self.eps_dual_infeasible,
            iteration_limit=self.iteration_limit,
        )

    def initialize_solver_status(
        self,
        problem: OTProblem,
        initial_primal_solution: jnp.array,
        initial_dual_solution: jnp.array,
    ) -> Tuple[PdhgSolverState, RestartInfo]:
        """Initialize the solver status for PDHG.

        Parameters
        ----------
        scaled_problem : ScaledQpProblem
            Scaled quadratic programming problem instance.
        initial_primal_solution : jnp.array
            The initial primal solution.
        initial_dual_solution : jnp.array
            The initial dual solution.

        Returns
        -------
        PdhgSolverState
            The initial solver status.
        """
        primal_shape = (problem.n, problem.n)
        dual_shape = (2, problem.n)

        # Primal weight initialization
        if self.scale_invariant_initial_primal_weight:
            self._initial_primal_weight = select_initial_primal_weight(
                problem, 1.0, 1.0, self.primal_importance
            )
        else:
            self._initial_primal_weight = self.primal_importance


        if self.step_size is None:
            step_size = 1.0 / problem.norm_A
        else:
            step_size = self.step_size

        if self.warm_start:
            # scaled_initial_primal_solution = (
            #     initial_primal_solution * scaled_problem.variable_rescaling
            # )
            # scaled_initial_dual_solution = (
            #     initial_dual_solution * scaled_problem.constraint_rescaling
            # )
            initial_primal_product = ot_apply_A(initial_primal_solution)

            initial_dual_product = ot_apply_AT(initial_dual_solution)

            primal_obj_product = (
                self.reg * initial_primal_solution
            )
        else:
            initial_primal_solution = jnp.zeros(primal_shape)
            initial_dual_solution = jnp.zeros(dual_shape)
            initial_primal_product = jnp.zeros(dual_shape)
            initial_dual_product = jnp.zeros(primal_shape)
            primal_obj_product = jnp.zeros(primal_shape)
        solver_state = PdhgSolverState(
            current_primal_solution=initial_primal_solution,
            current_dual_solution=initial_dual_solution,
            current_primal_product=initial_primal_product,
            current_dual_product=initial_dual_product,
            current_primal_obj_product=primal_obj_product,
            reg=self.reg,
            solutions_count=0,
            weights_sum=0.0,
            step_size=step_size,
            primal_weight=self._initial_primal_weight,
            numerical_error=False,
            # total_number_iterations=0,
            avg_primal_solution=initial_primal_solution,
            avg_dual_solution=initial_dual_solution,
            avg_primal_product=initial_primal_product,
            avg_dual_product=initial_dual_product,
            avg_primal_obj_product=primal_obj_product,
            initial_primal_solution=initial_primal_solution,
            initial_dual_solution=initial_dual_solution,
            initial_primal_product=initial_primal_product,
            initial_dual_product=initial_dual_product,
            num_steps_tried=0,
            num_iterations=0,
            termination_status=TerminationStatus.UNSPECIFIED,
            delta_primal=jnp.zeros(primal_shape),
            delta_dual=jnp.zeros(dual_shape),
            delta_primal_product=jnp.zeros(dual_shape),
        )

        last_restart_info = RestartInfo(
            primal_solution=initial_primal_solution,
            dual_solution=initial_dual_solution,
            primal_diff=jnp.zeros(primal_shape),
            dual_diff=jnp.zeros(dual_shape),
            primal_diff_product=jnp.zeros(dual_shape),
            primal_product=initial_primal_product,
            dual_product=initial_dual_product,
            primal_obj_product=primal_obj_product,
        )
        return solver_state, last_restart_info

    def take_step(
        self, solver_state: PdhgSolverState, problem: OTProblem
    ) -> PdhgSolverState:
        """
        Take a PDHG step with adaptive step size.

        Parameters
        ----------
        solver_state : PdhgSolverState
            The current state of the solver.
        problem : QuadraticProgrammingProblem
            The problem being solved.
        """

        (
            delta_primal,
            delta_dual,
            delta_primal_product,
            step_size,
            line_search_iter,
        ) = advance_iterate(
            self.update_strategy,  # pick once, at Python level
            problem,
            solver_state,
            reg=solver_state.reg,
            reduction_exp=self.adaptive_step_size_reduction_exponent,
            growth_exp=self.adaptive_step_size_growth_exponent,
            limit_coef=self.adaptive_step_size_limit_coef,
            norm_A=problem.norm_A,
        )
        next_primal_solution = solver_state.current_primal_solution + delta_primal
        next_primal_product = solver_state.current_primal_product + delta_primal_product
        next_primal_obj_product = solver_state.reg * next_primal_solution
        next_dual_solution = solver_state.current_dual_solution + delta_dual
        next_dual_product = ot_apply_AT(next_dual_solution)

        ratio = step_size / (solver_state.weights_sum + step_size)
        next_avg_primal_solution = solver_state.avg_primal_solution + ratio * (
            next_primal_solution - solver_state.avg_primal_solution
        )
        next_avg_dual_solution = solver_state.avg_dual_solution + ratio * (
            next_dual_solution - solver_state.avg_dual_solution
        )
        next_avg_primal_product = solver_state.avg_primal_product + ratio * (
            next_primal_product - solver_state.avg_primal_product
        )
        next_avg_dual_product = solver_state.avg_dual_product + ratio * (
            next_dual_product - solver_state.avg_dual_product
        )
        next_avg_primal_obj_product = solver_state.avg_primal_obj_product + ratio * (
            next_primal_obj_product - solver_state.avg_primal_obj_product
        )
        new_solutions_count = solver_state.solutions_count + 1
        new_weights_sum = solver_state.weights_sum + step_size

        return PdhgSolverState(
            current_primal_solution=next_primal_solution,
            current_dual_solution=next_dual_solution,
            current_primal_product=next_primal_product,
            current_dual_product=next_dual_product,
            current_primal_obj_product=next_primal_obj_product,
            reg=solver_state.reg,
            avg_primal_solution=next_avg_primal_solution,
            avg_dual_solution=next_avg_dual_solution,
            avg_primal_product=next_avg_primal_product,
            avg_dual_product=next_avg_dual_product,
            avg_primal_obj_product=next_avg_primal_obj_product,
            initial_primal_solution=solver_state.initial_primal_solution,
            initial_dual_solution=solver_state.initial_dual_solution,
            initial_primal_product=solver_state.initial_primal_product,
            initial_dual_product=solver_state.initial_dual_product,
            delta_primal=delta_primal,
            delta_dual=delta_dual,
            delta_primal_product=delta_primal_product,
            solutions_count=new_solutions_count,
            weights_sum=new_weights_sum,
            step_size=step_size,
            primal_weight=solver_state.primal_weight,
            numerical_error=False,
            num_steps_tried=solver_state.num_steps_tried + line_search_iter,
            num_iterations=solver_state.num_iterations + 1,
            termination_status=TerminationStatus.UNSPECIFIED,
        )

    def take_multiple_steps(
        self, solver_state: PdhgSolverState, problem: OTProblem
    ) -> PdhgSolverState:
        """
        Take multiple PDHG step with adaptive step size.

        Parameters
        ----------
        solver_state : PdhgSolverState
            The current state of the solver.
        problem : QuadraticProgrammingProblem
            The problem being solved.
        """
        new_solver_state = jax.lax.fori_loop(
            lower=0,
            upper=self.termination_evaluation_frequency,
            body_fun=lambda i, x: self.take_step(x, problem),
            init_val=solver_state,
        )
        return new_solver_state

    def initial_iteration_update(
        self,
        solver_state,
        last_restart_info,
        should_terminate,
        problem: OTProblem,
    ):
        """The inner loop of PDLP algorithm.

        Parameters
        ----------
        solver_state : PdhgSolverState
            The current state of the solver.
        last_restart_info : RestartInfo
            The information of the last restart.
        should_terminate : bool
            Whether the algorithm should terminate.
        scaled_problem : ScaledQpProblem
            The scaled quadratic programming problem.
        qp_cache : CachedQuadraticProgramInfo
            The cached quadratic programming information.

        Returns
        -------
        tuple
            The updated solver state, the updated last restart info, whether to terminate, the scaled problem, and the cached quadratic programming information.
        """
        # Skip termination check for initial iterations
        restarted_solver_state, new_last_restart_info = run_restart_scheme(
            problem,
            solver_state,
            last_restart_info,
            self._restart_params,
            self.optimality_norm,
        )

        new_solver_state = self.take_step(
            restarted_solver_state, problem
        )
        new_solver_state.termination_status = TerminationStatus.UNSPECIFIED
        return (
            new_solver_state,
            new_last_restart_info,
            False,
            problem,
        )

    def main_iteration_update(
        self,
        solver_state,
        last_restart_info,
        should_terminate,
        problem: OTProblem,
        ci,
    ):
        """The inner loop of PDLP algorithm.

        Parameters
        ----------
        solver_state : PdhgSolverState
            The current state of the solver.
        last_restart_info : RestartInfo
            The information of the last restart.
        should_terminate : bool
            Whether the algorithm should terminate.
        problem : OTProblem
            The scaled quadratic programming problem.
        qp_cache : CachedQuadraticProgramInfo
            The cached quadratic programming information.

        Returns
        -------
        tuple
            The updated solver state, the updated last restart info, whether to terminate, the scaled problem, and the cached quadratic programming information.
        """
        # Check for termination
        new_should_terminate, new_termination_status, new_convergence_information = (
            check_termination_criteria(
                problem,
                solver_state,
                self._termination_criteria,
                solver_state.numerical_error,
                1.0,
                self.optimality_norm,
                average=False,
                infeasibility_detection=self.infeasibility_detection,
                reference_solution=self.reference_solution
            )
        )

        primal_error = new_convergence_information.primal_residual_norm
        new_reg = compute_reg(
            strategy=self.reg_strategy,
            error=primal_error,
            value=solver_state.reg,
            cool_down_param=self.cool_down_param,
            thr_err=self.cool_down_threshold,
            init_err=self._init_error,
            init_reg=self.reg,
        )

        solver_state.reg = new_reg

        restarted_solver_state, new_last_restart_info = run_restart_scheme(
            problem,
            solver_state,
            last_restart_info,
            self._restart_params,
            self.optimality_norm,
        )

        new_solver_state = self.take_multiple_steps(
            restarted_solver_state, problem
        )
        new_solver_state.termination_status = new_termination_status

        return (
            new_solver_state,
            new_last_restart_info,
            new_should_terminate,
            problem,
            new_convergence_information,
        )


    def optimize(
        self,
        original_problem: OTProblem,
        initial_primal_solution=None,
        initial_dual_solution=None,
    ) -> Tuple[SaddlePointOutput, Any]:
        setup_logger(self.verbose, self.debug)

        self.check_config()

        # precondition_start_time = timeit.default_timer()
        # scaled_problem = rescale_problem(
        #     self.l_inf_ruiz_iterations,
        #     self.l2_norm_rescaling,
        #     self.pock_chambolle_alpha,
        #     original_problem,
        # )
        # precondition_time = timeit.default_timer() - precondition_start_time
        # logger.info("Preconditioning Time (seconds): %.2e", precondition_time)

        solver_state, last_restart_info = self.initialize_solver_status(
            original_problem,
            initial_primal_solution,
            initial_dual_solution,
        )

        # Iteration loop
        display_iteration_stats_heading()

        iteration_start_time = timeit.default_timer()
        # (solver_state, last_restart_info, should_terminate, _) = while_loop(
        #     cond_fun=lambda state: state[2] == False,
        #     body_fun=lambda state: self.initial_iteration_update(*state),
        #     init_val=(solver_state, last_restart_info, False, original_problem),
        #     maxiter=10,
        #     unroll=self.unroll,
        #     jit=self.jit,
        # )

        iter_stats = evaluate_unscaled_iteration_stats(
            problem=original_problem,
            solver_state=solver_state,
            eps_ratio=self.eps_abs/self.eps_rel,
            cumulative_time=0.0,
            norm_ord=self.optimality_norm
        )

        self._init_error = iter_stats.convergence_information.primal_residual_norm

        (solver_state, last_restart_info, should_terminate, _, ci), iters = while_loop_iters(
            cond_fun=lambda state: state[2] == False,
            body_fun=lambda state: self.main_iteration_update(*state),
            init_val=(
                solver_state,
                last_restart_info,
                False,
                original_problem,
                ConvergenceInformation(),
            ),
            maxiter=self.iteration_limit,
            save_iters=self.save_iters,
            jit=self.jit,
        )
        iteration_time = timeit.default_timer() - iteration_start_time

        timing = {
            # "Preconditioning": precondition_time,
            "Iteration loop": iteration_time,
        }

        # Log the stats of the final iteration.
        pdhg_final_log(
            original_problem,
            solver_state.avg_primal_solution,
            solver_state.avg_dual_solution,
            solver_state.num_iterations,
            solver_state.termination_status,
            timing,
            ci,
        )
        return unscaled_saddle_point_output(
            original_problem,
            solver_state.avg_primal_solution,
            solver_state.avg_dual_solution,
            solver_state.termination_status,
            solver_state.num_iterations - 1,
            ci,
            timing,
        ), cut_dict_at_first_zero(iters)
