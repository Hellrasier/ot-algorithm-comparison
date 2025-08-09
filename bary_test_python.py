import jax
import ot
from jax import numpy as jnp
from typing import Any
import numpy as np
import time
import logging
from uot.problems.generators.gaussian_mixture_generator import GaussianMixtureGenerator
from uot.experiments.experiment import Experiment
from uot.solvers.base_solver import BaseSolver
from uot.utils.costs import cost_euclid_squared
from matplotlib import pyplot as plt
from pathlib import Path

from uot.solvers.pdlp_barycenter import PDLPBarycenterSolver
from uot.solvers.pot_barycenter import POTSinkhornBarycenterSolver
from uot.problems.problem_generator import ProblemGenerator
from uot.utils.logging import logger
from matplotlib.ticker import FuncFormatter
from typing import Any, Sequence

jax.config.update("jax_enable_x64", True)

plt.style.use("seaborn-v0_8-whitegrid")          # bundled with Matplotlib
plt.rcParams.update(
    {
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "font.size": 11,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "grid.alpha": 0.4,
    }
)

_PALETTE = plt.cm.get_cmap("tab10").colors       # colour-blind friendly


def _log_format(x, _):
    """Format log ticks like 1e-3 instead of 0.001."""
    if x == 0:
        return "0"
    exp = int(np.log10(x))
    coeff = x / 10**exp
    if np.isclose(coeff, 1.0):
        return fr"$10^{{{exp}}}$"
    return fr"${coeff:.0f}\times10^{{{exp}}}$"


def _apply_axes_style(ax, x_label: str):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_log_format))
    ax.yaxis.set_major_formatter(FuncFormatter(_log_format))
    ax.set_xlabel(x_label)
    ax.grid(True, which="both", ls="--", lw=0.3)


def _wait_jax_finish(result: dict[str, Any]) -> dict[str, Any]:
    """Block until all JAX arrays in `result` are ready."""
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if isinstance(x, jax.Array) else x,
        result
    )

def gm_pdf(x, means, sigmas, weights=None):
    means  = jnp.asarray(means,  dtype=float)
    sigmas = jnp.asarray(sigmas, dtype=float)
    if weights is None:
        weights = jnp.ones_like(means) / means.size
    else:
        weights = jnp.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    x = jnp.asarray(x, dtype=float)[..., None]          # broadcast over comps
    z = (x - means) / sigmas
    comp_pdf = jnp.exp(-0.5 * z**2) / (sigmas * jnp.sqrt(2.0 * jnp.pi))
    return jnp.sum(weights * comp_pdf, axis=-1)

def gm_pdf_nd(
    x: jnp.ndarray,             # (N, d)
    means: jnp.ndarray,         # (k, d)
    sigmas: jnp.ndarray,        # (k, d)  *std-devs*, diag cov
    weights: jnp.ndarray | None = None,  # (k,)
):
    """Evaluate Σ_j π_j 𝒩(μ_j, diag(σ_j²)) at *N* points x∈ℝᵈ."""
    means, sigmas = map(lambda a: jnp.asarray(a, dtype=float), (means, sigmas))
    if weights is None:
        weights = jnp.ones(means.shape[0], dtype=float) / means.shape[0]
    else:
        weights = jnp.asarray(weights, dtype=float) / jnp.sum(weights)

    x = jnp.asarray(x, dtype=float)               # (N, d)
    d  = x.shape[-1]

    # broadcast: (N, k, d)
    diff = x[:, None, :] - means[None, :, :]
    z2   = jnp.sum((diff / sigmas) ** 2, axis=-1)           # (N, k)

    log_norm = -0.5 * (d * jnp.log(2.0 * jnp.pi) +
                       jnp.sum(jnp.log(sigmas), axis=-1))   # (k,)
    comp_pdf = jnp.exp(log_norm - 0.5 * z2)                 # (N, k)

    return jnp.sum(weights * comp_pdf, axis=-1)

def gm_barycenter_1d(
    margs,
    marg_weights: jnp.ndarray | None = None,   # (n_marginals,), optional
):
    means = [margs[0].means[0], margs[1].means[0]] if margs[0].means.shape == (1, 1) else jnp.array([
        margs[0].means, margs[1].means
    ]).reshape(2, 2)
    sigmas = [
        jnp.sqrt(margs[0].covs[0][0]), jnp.sqrt(margs[1].covs[0][0])
    ] if margs[0].covs.shape == (1, 1, 1) else [
        jnp.sqrt(margs[0].covs), jnp.sqrt(margs[1].covs)
    ]
    mix_weights = [margs[0].comp_weights, margs[1].comp_weights]

    means   = jnp.asarray(means,  dtype=float)
    sigmas  = jnp.asarray(sigmas, dtype=float)
    weights = jnp.asarray(mix_weights, dtype=float)
    marg_weights = jnp.asarray(marg_weights, dtype=float)
    marg_weights = marg_weights / marg_weights.sum()

    if means.shape != sigmas.shape or means.shape != weights.shape:
        raise ValueError("means, sigmas, mix_weights must share the same shape")

    # barycenter mixture weights  π̄_j
    bar_mix = (marg_weights[:, None] * weights).sum(axis=0)
    bar_mix = bar_mix / bar_mix.sum()        # ensure ∑π̄ = 1

    # component‑wise Gaussian barycenters
    bar_means  = (marg_weights[:, None] * means).sum(axis=0)
    bar_sigmas = (marg_weights[:, None] * sigmas).sum(axis=0)

    return bar_means, bar_sigmas, bar_mix

def gm_barycenter_2d(
    margs: Sequence,                 # your marginal objects
    marg_weights: jnp.ndarray,       # (n_marginals,)
):
    """
    Very simple component-wise barycenter:
      μ̄_j = Σ_i α_i μ_{i,j} ,  σ̄_j = Σ_i α_i σ_{i,j}
    matching the *same approximation* you used in 1-D.
    """
    # unpack means, (diag) covs, mixture weights per marginal
    means   = jnp.stack([m.means          for m in margs])        # (m, k, d)
    sigmas  = jnp.stack([jnp.sqrt(jnp.diagonal(m.covs, axis1=1, axis2=2))
                         for m in margs])                         # (m, k, d)
    weights = jnp.stack([m.comp_weights  for m in margs])         # (m, k)

    marg_weights = jnp.asarray(marg_weights, dtype=float)
    marg_weights = marg_weights / marg_weights.sum()

    # mixture weights π̄_j
    bar_mix = jnp.sum(marg_weights[:, None] * weights, axis=0)
    bar_mix = bar_mix / bar_mix.sum()

    # component-wise centre / spread
    bar_means  = jnp.sum(marg_weights[:, None, None] * means,  axis=0)   # (k, d)
    bar_sigmas = jnp.sum(marg_weights[:, None, None] * sigmas, axis=0)   # (k, d)

    return bar_means, bar_sigmas, bar_mix


def direct_bias(bary, grid, m_star, sigma_star, comp_weights):
    nu_star = gm_pdf(grid, m_star, sigma_star, comp_weights)
    nu_star /= nu_star.sum()

    mse = jnp.sum((bary - nu_star) ** 2)

    C = (grid[:, None] - grid[None, :])**2
    w2 = jnp.sqrt(ot.emd2(bary / bary.sum(), nu_star, C))
    return mse, w2

def direct_bias_2d(
    bary: jnp.ndarray,          # (N,)  – flattened barycenter from solver
    grid: jnp.ndarray,          # (N, 2)
    m_star: jnp.ndarray,        # (k, 2)
    sigma_star: jnp.ndarray,    # (k, 2)
    comp_weights: jnp.ndarray,  # (k,)
):
    """MSE & W₂ between numeric bary and analytic GM bary on the grid."""
    nu_star = gm_pdf_nd(grid, m_star, sigma_star, comp_weights)
    nu_star /= nu_star.sum()

    mse = jnp.sum((bary - nu_star) ** 2)

    # pair-wise squared Euclidean distance matrix (N,N)
    C = jnp.sum((grid[:, None, :] - grid[None, :, :]) ** 2, axis=-1)
    logger.info(f"C shape - {C.shape}")
    logger.info(f"{bary / bary.sum() == nu_star}")
    w2 = jnp.sqrt(
        ot.emd2(             # POT still expects NumPy
            np.asarray(bary / bary.sum()),
            np.asarray(nu_star),
            np.asarray(C),
        )
    )
    print("W2", w2)
    return mse, w2

def measure_barycenter(prob, solver, marginals, costs, **kwargs):
    instance = solver()

    start_time = time.perf_counter()
    res = instance.solve(marginals=marginals, costs=costs, **kwargs)
    _wait_jax_finish(res)
    time_end = (time.perf_counter() - start_time) * 1000

    bary, plans, us = res["barycenter"], res["transport_plans"], res["vs_final"]
    grid = marginals[0].to_discrete()[0].squeeze()

    m_star, sigma_star, comp_weights = gm_barycenter_1d(marginals, prob._weights)

    if not jnp.isnan(bary).any():
        dl2, dw2 = direct_bias(bary, grid, m_star, sigma_star, comp_weights)
    else:
        dl2, dw2 = jnp.nan, jnp.nan

    return dict(
        direct_l2_bias      = dl2,
        direct_w2_bias      = dw2,
        time                = time_end,
    )

def measure_barycenter2d(
    prob,                 # problem wrapper holding ._weights
    solver,               # callable returning a solver instance
    marginals,            # list of marginal objects
    costs,                # cost matrices for the solver (unused here)
    **kwargs,
):
    instance = solver()

    tic = time.perf_counter()
    res = instance.solve(marginals=marginals, costs=costs, **kwargs)
    _wait_jax_finish(res)
    elapsed_ms = (time.perf_counter() - tic) * 1_000

    bary  = res["barycenter"]           # (N,)
    grid  = marginals[0].to_discrete()[0]   # (N, 2)

    m_star, sigma_star, comp_weights = gm_barycenter_2d(
        marginals, prob._weights
    )

    if not jnp.isnan(bary).any():
        dl2, dw2 = direct_bias_2d(bary, grid, m_star, sigma_star, comp_weights)
    else:
        dl2, dw2 = jnp.nan, jnp.nan

    return dict(
        direct_l2_bias = dl2,
        direct_w2_bias = dw2,
        time           = elapsed_ms,
    )


import pandas as pd


def run_experiments_save_json(
    generator: ProblemGenerator,
    regs: list[float],
    solvers: list[BaseSolver],
    data_dir: str | Path = "output/data",
    **kwargs,
):
    ## Gather metrics first
    records = []
    if generator._dim == 2:
        experiment = Experiment(
            name="Barycetner tests  2d",
            solve_fn=measure_barycenter2d,
        )
    else:
        experiment = Experiment(
            name="Barycenter tests",
            solve_fn=measure_barycenter
        )
    for solver in solvers:
        for reg in regs:
            df = experiment.run_on_problems(
                problems=generator.generate_barycenter(),
                solver=solver,
                reg=reg,
                **kwargs,
            )

            records.append({
                'name': "PDLPBarycenter" if solver.__name__ == "PDLPBarycenterSolver" else "POTSinkhornBarycenter",
                'reg': reg,
                'avg_direct_l2_bias': df["direct_l2_bias"].mean(),
                'avg_direct_w2_bias': df["direct_w2_bias"].mean(),
                # 'avg_lp_l2_bias': df["lp_l2_bias"].mean(),
                # 'avg_lp_w2_bias': df["lp_w2_bias"].mean(),
                'avg_time': df["time"].mean(),
            })

    df_result = pd.DataFrame(records)

    # ── Save as JSON ───────────────────────────────────────
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / f"{generator._name}-bary.json"
    df_result.to_json(out_file, orient="records", indent=2)
    print(f"[✓] Results written to {out_file}")

    ## Plot the results


def plot_results_from_json(
    generator_name: str,
    data_dir: str | Path = "output/data",
    img_dir: str | Path = "output/images",
) -> Path:
    data_dir, img_dir = Path(data_dir), Path(img_dir)
    in_file = data_dir / f"{generator_name}-bary.json"
    if not in_file.is_file():
        raise FileNotFoundError(in_file)

    df = pd.read_json(in_file)

    # --- three side-by-side axes, tighter spacing ---------------------------
    fig, (ax_l2, ax_w2, ax_rt) = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.subplots_adjust(wspace=0.25)          # 👈 reduce horizontal gap
    fig.suptitle(generator_name, x=0.01, ha="left")

    for i, (name, sub) in enumerate(df.groupby("name")):
        colour = _PALETTE[i % len(_PALETTE)]
        style  = "-" if i % 2 == 0 else "--"

        # L2 & runtime keep the alternating style …
        ax_l2.plot(sub["reg"], sub["avg_direct_l2_bias"],
                   style="-", color=colour, marker="o", label=name)
        ax_rt.plot(sub["reg"], sub["avg_time"],
                   style="-", color=colour, marker="o", label=name)

        # … but W2 precision is *always* solid:
        ax_w2.plot(sub["reg"], sub["avg_direct_w2_bias"],
                   "-",   color=colour, marker="o", label=name)  # 👈 solid line

        # annotate minima (unchanged)
        for ax, col in ((ax_l2, "avg_direct_l2_bias"),
                        (ax_w2, "avg_direct_w2_bias"),
                        (ax_rt, "avg_time")):
            idx = sub[col].idxmin()
            ax.annotate(f"{sub[col][idx]:.1e}" if "bias" in col
                        else f"{sub[col][idx]:.1f}",
                        xy=(sub["reg"][idx], sub[col][idx]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=8, color=colour)

    # cosmetics (unchanged)
    for ax, title in (
        (ax_l2, "Mean $L_2$ precision"),
        (ax_w2, "Mean $W_2$ precision"),
        (ax_rt, "Mean runtime"),
    ):
        _apply_axes_style(ax, r"Regularisation $\varepsilon$")
        ax.set_title(title, loc="left")

    handles, labels = ax_l2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    fig.tight_layout()
    img_dir.mkdir(parents=True, exist_ok=True)
    out_path = img_dir / f"{generator_name}-bary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Plot saved to {out_path}")
    return out_path

# regs = [1e-1, 5e-1, 1.0, 5.0, 10.0, 100.0, 1000.0, 10000.0]
regs_additional = [1e-2, 1e-3]
solvers = [PDLPBarycenterSolver, POTSinkhornBarycenterSolver]

experiment_settings = {
    "Gaussian1D": [512, 1024, 2048, 4096],
    "Gaussian2D": [64],
}

if __name__ == "__main__":
    for n in experiment_settings["Gaussian1D"]:
        generator = GaussianMixtureGenerator(
            name=f"Gaussian1D-{n}p-additional",
            dim=1,
            num_marginals=2,
            num_components=1,
            n_points=n,
            num_datasets=20,
            weights=[0.5, 0.5],
            borders=[-6, 6],
            cost_fn=cost_euclid_squared,
            use_jax=False
        )

        run_experiments_save_json(
            generator=generator,
            solvers=solvers,
            regs=regs_additional,
            weights=[0.5, 0.5], maxiter=15000, tol=1e-5
        )

        # plot_results_from_json(
        #     generator_name=f"Gaussian1D-{n}p"
        # )

    # for n in experiment_settings["Gaussian2D"]:
    #     generator = GaussianMixtureGenerator(
    #         name=f"Gaussian2D-{n}p",
    #         dim=2,
    #         num_marginals=2,
    #         num_components=1,
    #         n_points=n,
    #         num_datasets=5,
    #         weights=[0.5, 0.5],
    #         borders=[-6, 6],
    #         cost_fn=cost_euclid_squared,
    #         use_jax=False
    #     )
    #
    #     run_experiments_save_json(
    #         generator=generator,
    #         solvers=solvers,
    #         regs=regs,
    #         weights=[0.5, 0.5], maxiter=15000, tol=1e-6
    #     )
    #
    #     plot_results_from_json(
    #         generator_name=f"Gaussian2D-{n}p"
    #     )
