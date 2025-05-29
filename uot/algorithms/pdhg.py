import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from matplotlib import pyplot as plt

class ConvInfo(NamedTuple):
    primal_residual: jax.Array = 0.0
    dual_residual: jax.Array = 0.0
    obj_difference: jax.Array = 0.0
    l2_difference: jax.Array = 0.0


class OTState(NamedTuple):
    """
    NamedTuple to hold the state during PDHG iteration for OT.

    Attributes
    ----------
    coupling_k : jax.Array
        Current coupling (transport plan), shape (n, m).
    u_k : jax.Array
        Dual vector for the "mu" (row) constraints, shape (n,).
    v_k : jax.Array
        Dual vector for the "nu" (column) constraints, shape (m,).
    computed_marginals : tuple
        Cached row/column sums from the coupling, i.e. (row_sums, col_sums).
    k : int
        Current iteration count.
    done : bool
        Whether the stopping criterion has been met.
    """
    coupling_k: jax.Array
    u_k: jax.Array
    v_k: jax.Array
    computed_marginals: Tuple[jax.Array, jax.Array]
    k: int
    done: bool
    convergence_info: ConvInfo


# ==============================================================================
#  PDHG step functions
# ==============================================================================

@jax.jit
def pdhg_quadratic_ot_step(
        coupling_k: jax.Array,
        u: jax.Array,
        v: jax.Array,
        computed_marginals_prev: Tuple[jax.Array, jax.Array],
        C: jax.Array,
        mu: jax.Array,
        nu: jax.Array,
        tau: float,
        sigma: float,
        eps: float,
) -> Tuple[jax.Array, jax.Array, jax.Array, Tuple[jax.Array, jax.Array]]:
    """
    One iteration of PDHG for the Quadratic-regularized OT problem.

    We perform:
      1) coupling^(k+1) = prox_quadratic_coupling(coupling^k, u^k, v^k, C, tau, eps)
      2) u^(k+1), v^(k+1) = dual updates using row/column marginals.

    Parameters
    ----------
    coupling_k : jax.Array
        Current coupling (n, m).
    u : jax.Array
        Current dual vector for row constraints (n,).
    v : jax.Array
        Current dual vector for column constraints (m,).
    computed_marginals_prev : Tuple[jax.Array, jax.Array]
        (row_sum(coupling_k), col_sum(coupling_k)).
    C : jax.Array
        Cost matrix (n, m).
    mu : jax.Array
        Vector of supply constraints (n,).
    nu : jax.Array
        Vector of demand constraints (m,).
    tau : float
        Primal step size.
    sigma : float
        Dual step size.
    eps : float
        Quadratic regularization weight.

    Returns
    -------
    coupling_next : jax.Array
        Updated coupling (n, m).
    u_next : jax.Array
        Updated dual vector for row constraints (n,).
    v_next : jax.Array
        Updated dual vector for column constraints (m,).
    computed_marginals_next : Tuple[jax.Array, jax.Array]
        (row_sum, col_sum) of coupling_next.
    grad : jax.Array
        Gradient C - (u + v)
    """
    # 1) Primal update
    grad = C - (u[:, None] + v[None, :])
    coupling_unconstrained = (coupling_k - tau * grad) / (1.0 + tau * eps)
    coupling_next = jnp.clip(coupling_unconstrained, 0.0, None)

    # 2) Dual update
    row_prev, col_prev = computed_marginals_prev
    row_next = jnp.sum(coupling_next, axis=1)
    col_next = jnp.sum(coupling_next, axis=0)

    u_next = u + sigma * (mu - (2.0 * row_next - row_prev))
    v_next = v + sigma * (nu - (2.0 * col_next - col_prev))

    return coupling_next, u_next, v_next, (row_next, col_next), grad


# ==============================================================================
#  PDHG "main" functions for Quadratic vs. LP OT
# ==============================================================================

def pdhg_quadratic_ot(
        C: jax.Array,
        mu: jax.Array,
        nu: jax.Array,
        eps: float = 1e-2,
        tau: float = 0.9,
        sigma: float = 0.9,
        tol: float = 1e-8,
        max_outer_iter: int = 1000,
        initial_point=None,
        save_iters: bool = True,
        ref_coupling=None,
        verbose: bool = False,
):
    """
    Run PDHG to solve the Quadratic-regularized OT problem:

      min_{coupling >= 0}  <coupling, C> + (eps/2) ||coupling||^2
      subject to  coupling @ 1 = mu,  coupling^T @ 1 = nu.

    This code uses an iterative primal-dual hybrid gradient method with a
    "preconditioned" update. The loop is unrolled via `jax.lax.scan`.

    Parameters
    ----------
    C : jax.Array
        Cost matrix of shape (n, m).
    mu : jax.Array
        Supply vector (size n).
    nu : jax.Array
        Demand vector (size m).
    eps : float, optional
        Quadratic regularization weight, by default 1e-2.
    tau : float, optional
        Primal step size, by default 0.9.
    sigma : float, optional
        Dual step size, by default 0.9.
    tol : float, optional
        Stopping tolerance on the maximum row/col constraints mismatch, by default 1e-8.
    max_outer_iter : int, optional
        Maximum number of PDHG iterations, by default 1000.
    initial_point : optional
        A tuple (coupling_init, dual_init) to initialize primal/dual. If None, uses a default guess.
        `dual_init` should be concatenated (u, v) or similar.
    save_iters : bool, optional
        If True, returns the coupling at each iteration from the scan, by default True.

    Returns
    -------
    coupling_final : jax.Array
        The final transport plan (n, m).
    u_final : jax.Array
        The final dual vector for row constraints (n,).
    v_final : jax.Array
        The final dual vector for column constraints (m,).
    iters : jax.Array or None
        If `save_iters=True`, returns a stack of coupling iterates. Otherwise None.

    Notes
    -----
    - GPU-ready: if you have JAX installed with CUDA support and your default device
      is GPU, it will automatically run there.
    - For extremely large problems, consider multi-GPU or HPC approaches.
    """
    n, m = C.shape

    # 1) Initialize primal/dual
    if initial_point is None:
        # coupling_k = jnp.ones((n, m)) * (1.0 / (n * m))
        coupling_k = jnp.zeros((n, m))
        u = jnp.zeros(n)
        v = jnp.zeros(m)
    else:
        coupling_k = initial_point[0]
        full_dual = initial_point[1]
        u = full_dual[:n]
        v = full_dual[n:]

    computed_marginals = (
        jnp.sum(coupling_k, axis=1),  # row sums
        jnp.sum(coupling_k, axis=0)  # column sums
    )

    mu = jnp.asarray(mu)
    nu = jnp.asarray(nu)

    @jax.jit
    def stopping_criteria(marginals):
        """
        Check if max row/col mismatch is below tol.
        """
        row_sum, col_sum = marginals
        marginals_error = jnp.maximum(
            jnp.linalg.norm(row_sum - mu),
            jnp.linalg.norm(col_sum - nu)
        )
        return (marginals_error < tol)

    def one_step(carry, stats):
        """
        Single iteration that updates (coupling, u, v) if not done,
        otherwise skips updating if 'done' is True.
        """
        (coupling_k, u_k, v_k, cmarg, k, done, _) = carry

        def skip_updates(state: OTState):
            # saved_stats = stats if save_iters else None
            stats = {
                "primal_feas": state.convergence_info.primal_residual,
                "obj_diff": state.convergence_info.obj_difference,
                "l2_diff": state.convergence_info.l2_difference,
                "dual_feas": state.convergence_info.dual_residual,

            } if save_iters else None
            return state, stats

        def do_one_step(state):
            coupling_k, u_k, v_k, cmarg_prev, k, _done, _ = state
            coupling_next, u_next, v_next, cmarg_next, grad = pdhg_quadratic_ot_step(
                coupling_k, u_k, v_k, cmarg_prev, C, mu, nu, tau, sigma, eps
            )

            row_sum, col_sum = state.computed_marginals
            marginals_error = jnp.maximum(
                jnp.linalg.norm(row_sum - mu),
                jnp.linalg.norm(col_sum - nu)
            )

            new_done = (marginals_error < tol)

            # saved_iter = coupling_next if save_iters else None

            convergence_info= ConvInfo(
                primal_residual=marginals_error,
                # dual_residual=jnp.linalg.norm(coupling_k - jnp.clip((-grad) / eps, 0.0))
                #     if eps != 0 else jnp.linalg.norm(jnp.clip(-grad, 0.0)),
            )

            # stats = {
            #     "primal_feas": convergence_info.primal_residual,
            #     "obj_diff": convergence_info.obj_difference,
            #     "l2_diff": convergence_info.l2_difference,
            #     "dual_feas": convergence_info.dual_residual,
            #
            # } if save_iters else None
            return OTState(coupling_next, u_next, v_next, cmarg_next, k + 1, new_done, convergence_info), None

        return jax.lax.cond(
            done,
            skip_updates,
            do_one_step,
            carry
        )

    # 2) Use lax.scan to unroll up to max_outer_iter
    init_state = OTState(
        coupling_k,
        u, v,
        computed_marginals,
        0, False,
        ConvInfo(
            primal_residual=jnp.inf,
            dual_residual=jnp.inf,
            obj_difference=jnp.inf,
            l2_difference=jnp.inf,
        )
    )

    (cpl_fin, u_fin, v_fin, _, final_iter, done, _), iters = jax.lax.scan(
        one_step, init_state, xs=None, length=max_outer_iter
    )

    if verbose:

        # 3) Debug print info
        jax.lax.cond(
            jnp.logical_and(done),
            lambda: jax.debug.print("Quadratic OT converged at {x} iterations.", x=final_iter),
            lambda: jax.debug.print(
                "Quadratic OT didn't converge within {mi} iterations, stopped at {fi}.",
                mi=max_outer_iter, fi=final_iter
            )
        )

    return cpl_fin, jnp.sum(cpl_fin * C), done





def make_graphs(array_iters_reg = [], regs = [], filename=None, labels=None, plot_every=1):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2 , figsize=(20,14))


    # if plot_every > 1:
    #     array_iters_reg = array_iters_reg[::plot_every]

    fig.suptitle("Convergence plots for Quadratic PDHG")

    array_primal_residuals_reg = [iters["primal_feas"][::plot_every] for iters in array_iters_reg]
    array_dual_residuals = [iters["dual_feas"][::plot_every] for iters in array_iters_reg]
    array_obj_differences_reg = [iters["obj_diff"][::plot_every] for iters in array_iters_reg]
    array_l2_differences_reg = [iters["l2_diff"][::plot_every] for iters in array_iters_reg]

    min_len = min(map(len, array_primal_residuals_reg))
    x = jnp.arange(min_len)
    # x = jnp.arange(0, len(array_primal_residuals_reg[0]))

    if labels is None:
        labels = [f"Reg={reg}" for reg in regs]

    # Axis 1 Absolute tolerance Primal Feasibility
    for (primal_residuals_reg, reg, label) in zip(array_primal_residuals_reg, regs, labels):
        ax1.plot(x, primal_residuals_reg[:min_len], label=label)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_title("Marginals error")

    for (dual_residuals, reg, label) in zip(array_dual_residuals, regs, labels):
        ax2.plot(x, dual_residuals[:min_len], label=label)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.set_title("Dual feasibility")


    for (obj_differences_reg, reg, label) in zip(array_obj_differences_reg, regs, labels):
        ax3.plot(x, obj_differences_reg[:min_len], label=label)
    ax3.set_yscale('log')
    ax3.legend()
    ax3.set_title("Objective differences with LP solution")

    for (l2_differences_reg, reg, label) in zip(array_l2_differences_reg, regs, labels):
        ax4.plot(x, l2_differences_reg[:min_len], label=label)
    ax4.set_yscale('log')
    ax4.legend()
    ax4.set_title("L2 differences with LP solution")


    plt.tight_layout()
    if filename is not None:
        plt.savefig(f"plots/pdlp-gpu-{filename}.png")
    else:
        plt.plot()

def solve_pdhg(a, b, C, epsilon=0.01, precision=1e-4, max_iters=10_000, tau=0.2, sigma=0.2):
   result = pdhg_quadratic_ot(C, a, b, epsilon, tau, sigma, precision, max_iters, save_iters=False, verbose=False)
   return result

