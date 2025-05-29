"""Loop utilities."""

import jax
import jax.numpy as jnp

from .utils import save_conv_info_to_dict, blank_conv_info, cut_dict_at_first_zero


def _while_loop_scan(cond_fun, body_fun, init_val, max_iter):
    """Scan-based implementation (jit ok, reverse-mode autodiff ok)."""

    def _iter(val):
        next_val = body_fun(val)
        next_cond = cond_fun(next_val)
        return next_val, next_cond

    def _fun(tup, it):
        val, cond = tup
        # When cond is met, we start doing no-ops.
        return jax.lax.cond(cond, _iter, lambda x: (x, False), val), it

    init = (init_val, cond_fun(init_val))
    return jax.lax.scan(_fun, init, None, length=max_iter)[0][0]

def _while_loop_scan_iters(cond_fun, body_fun, init_val, max_iter):
    def one_step(carry, _):
        state, keep_going = carry

        def _do_iter(state):
            state_next = body_fun(state)
            return (state_next, cond_fun(state_next)), save_conv_info_to_dict(state_next[-1])

        def _skip_update(state):
            return (state, False), blank_conv_info()

        (state, keep_going), emitted = jax.lax.cond(
            keep_going, _do_iter, _skip_update, state
        )
        return (state, keep_going), emitted

    init_carry = (init_val, cond_fun(init_val))
    (final_val, _), iters = jax.lax.scan(one_step, init_carry, None, length=max_iter)

    return final_val, iters


def _while_loop_python(cond_fun, body_fun, init_val, maxiter):
    """Python based implementation (no jit, reverse-mode autodiff ok)."""
    val = init_val
    for _ in range(maxiter):
        cond = cond_fun(val)
        if not cond:
            # When condition is met, break (not jittable).
            break
        val = body_fun(val)
    return val

def _while_loop_lax(cond_fun, body_fun, init_val, maxiter):
    """lax.while_loop based implementation (jit by default, no reverse-mode)."""

    def _cond_fun(_val):
        it, val = _val
        return jnp.logical_and(cond_fun(val), it <= maxiter - 1)

    def _body_fun(_val):
        it, val = _val
        val = body_fun(val)
        return it + 1, val

    return jax.lax.while_loop(_cond_fun, _body_fun, (0, init_val))[1]

def _while_loop_lax_iters(cond_fun, body_fun, init_val, maxiter):
    result = _while_loop_lax(cond_fun, body_fun, init_val, maxiter)
    return result, None


def while_loop(cond_fun, body_fun, init_val, maxiter, unroll=False, jit=False):
    """A while loop with a bounded number of iterations."""

    if unroll:
        if jit:
            fun = _while_loop_scan
        else:
            fun = _while_loop_python
    else:
        if jit:
            fun = _while_loop_lax
        else:
            raise ValueError("unroll=False and jit=False cannot be used together")

    if jit and fun is not _while_loop_lax:
        # jit of a lax while_loop is redundant, and this jit would only
        # constrain maxiter to be static where it is not required.
        fun = jax.jit(fun, static_argnums=(0, 1, 3))

    return fun(cond_fun, body_fun, init_val, maxiter)

def while_loop_iters(cond_fun, body_fun, init_val, maxiter, save_iters=None, jit=True):
    if not jit:
        return _while_loop_python(cond_fun, body_fun, init_val, maxiter)
    if save_iters is None:
        fun = _while_loop_lax_iters
    else:
        fun = _while_loop_scan_iters

    return jax.jit(fun, static_argnums=(0, 1, 3))(cond_fun, body_fun, init_val, maxiter)