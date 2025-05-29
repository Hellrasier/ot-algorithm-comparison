import jax
import jax.numpy as jnp
from enum import IntEnum
from functools import partial
from typing import Callable, Tuple

# ---------------------------------------------------------------------
# 1.  Define individual strategy kernels (each JAX‑friendly, no Python if)
# ---------------------------------------------------------------------

def log_cooldown(
    error: float,
    value: float,
    cool_down_param: float,
    init_reg: float,
    thr_err: float,
    init_err: float,
):
    e = jnp.clip(error, thr_err, init_err)
    ratio = jnp.log(e / thr_err) / jnp.log(init_err / thr_err)
    new_reg = jnp.clip(init_reg * jnp.power(ratio, cool_down_param), 0.0, init_reg)
    return jnp.minimum(new_reg, value)

def pow_cooldown(
    error: float,
    value: float,
    cool_down_param: float,
    init_reg: float,
    thr_err: float,
    init_err: float,
):
    ratio = (error - thr_err) / (init_err - thr_err)
    new_reg = jnp.clip(init_reg * jnp.power(jnp.clip(ratio, 0.0, 1.0), 1/cool_down_param), 0.0, init_reg)
    return jnp.minimum(new_reg, value)

class RegStrategy(IntEnum):
    CONSTANT       = 0
    POW_COOLDOWN   = 1
    LOG_COOLDOWN   = 2


@partial(jax.jit, static_argnums=0)
def compute_reg(
    strategy: RegStrategy,
    *,
    error: float | jax.Array = 0.0,
    value: float = 1.0,             # for CONSTANT
    cool_down_param: float = 1.0,   # for LINEAR_COOLDOWN
    init_reg: float,
    thr_err: float,
    init_err: float,
) -> jax.Array:
    """Return the updated regularisation parameter according to `strategy`."""
    # Wrap each kernel so it captures only the operands it needs
    kernels = (
        lambda _: value,
        lambda _: pow_cooldown(
            error=error,
            value=value,
            cool_down_param=cool_down_param,
            init_reg=init_reg,
            thr_err=thr_err,
            init_err=init_err,
        ),
        lambda _: log_cooldown(
            error=error,
            value=value,
            cool_down_param=cool_down_param,
            init_reg=init_reg,
            thr_err=thr_err,
            init_err=init_err,
        )
    )
    return jax.lax.switch(int(strategy), kernels, operand=None)