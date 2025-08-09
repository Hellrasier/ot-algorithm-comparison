import jax
from uot.problems.generators.gaussian_mixture_generator import GaussianMixtureGenerator
from uot.problems.generators.cauchy_generator import CauchyGenerator
from uot.solvers.base_solver import BaseSolver
from uot.utils.costs import cost_euclid_squared
from uot.solvers.pdlp import PDLPSolver
from uot.solvers.sinkhorn import SinkhornTwoMarginalSolver
from uot.experiments.experiment import Experiment
from uot.experiments.measurement import measure_time_and_precision, measure_pushforward
from uot.problems.problem_generator import ProblemGenerator
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import logging

logger = logging.getLogger('uot')

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

_PALETTE = plt.cm.get_cmap("tab10").colors

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

jax.config.update("jax_enable_x64", True)

import pandas as pd

from jax import numpy as jnp
import ot

def bary_proj_2d(pi: jnp.ndarray,
                 y:  jnp.ndarray,      # (m, 2) target grid coordinates
                 a:  jnp.ndarray       # (n,)   source weights
) -> jnp.ndarray:
    """
    Barycentric projection (2-D)

       T_i = (Σ_j π_{ij} y_j) / a_i ,   i = 1…n.

    Parameters
    ----------
    pi : (n, m) transport plan
    y  : (m, 2) target grid coordinates
    a  : (n,)   source masses  (∑ a_i = 1)

    Returns
    -------
    T : (n, 2) projected coordinates.
    """
    logger.info(f"shapes: {pi.shape}, {y.shape}")
    num = pi @ y                       # (n,2)
    T   = num / (a[:, None] + 1e-12)   # row-wise division
    return T


def rebin_2d(T: jnp.ndarray,
             w: jnp.ndarray,
             grid: jnp.ndarray         # (m, 2)
) -> jnp.ndarray:
    """
    Push the masses `w` located at positions `T`
    back onto the discrete `grid` via nearest-neighbour assignment.
    Returns weights on the grid (shape (m,)).
    """
    dists2 = jnp.sum((T[:, None, :] - grid[None, :, :]) ** 2, axis=-1)
    idx    = jnp.argmin(dists2, axis=1)              # (n,)
    return jnp.zeros(grid.shape[0]).at[idx].add(w)   # (m,)


def _cost_2d(grid: jnp.ndarray) -> jnp.ndarray:
    """Pair-wise squared Euclidean distance on a 2-D grid."""
    diff = grid[:, None, :] - grid[None, :, :]
    return jnp.sum(diff ** 2, axis=-1)


def bary_bias_2d(mu_w: jnp.ndarray,         # (n,)   source weights
                 nu_x: jnp.ndarray,         # (m,2)  target grid coords
                 nu_w: jnp.ndarray,         # (m,)   target weights
                 pi:   jnp.ndarray          # (n,m)  optimal plan
):
    """
    ℓ²- and W₂-errors for the barycentric push-forward in 2-D.
    """
    T        = bary_proj_2d(pi, nu_x, mu_w)
    nu_hat_w = rebin_2d(T, mu_w, nu_x)

    l2 = jnp.linalg.norm(nu_hat_w - nu_w)

    C  = _cost_2d(nu_x)
    w2 = jnp.sqrt(ot.emd2(nu_hat_w, nu_w, C))

    return l2, w2

def measure_pushforward_2d(prob, Solver, *args, **kw):
    res   = Solver().solve(*args, **kw)

    π = res["transport_plan"]

    # discrete marginals ----------------------------------------------
    (μ_x, μ_w), (ν_x, ν_w) = (m.to_discrete() for m in prob.get_marginals())
    μ_w, ν_w = μ_w.ravel(), ν_w.ravel()        # weights only!
    # μ_x, ν_x keep shape (N², 2)

    bl2, bw2 = bary_bias_2d(μ_w, ν_x, ν_w, π)

    return dict(
        barycentric_l2_bias = bl2,
        barycentric_w2_bias = bw2,
    )


def run_experiments_prec_save_json(
    generator,
    regs: list[float],
    solvers: list[BaseSolver],
    data_dir: str | Path = "output/data",
    **kwargs,
) -> pd.DataFrame:
    """
    Execute `measure_time_and_precision` on every (solver, ε) pair,
    return the aggregated DataFrame *and* save it as JSON.

    File path:  <data_dir>/<generator._name>.json
    """
    experiment = Experiment(
        name="Time and precision",
        solve_fn=measure_time_and_precision,
    )

    records = []
    for solver in solvers:
        for reg in regs:
            df = experiment.run_on_problems(
                problems=generator.generate(),
                solver=solver,
                reg=reg,
                **kwargs,
            )
            records.append(
                {
                    "name": "PDLP" if solver.__name__ == "PDLPSolver" else "Sinkhorn",
                    "reg": reg,
                    "avg_time": df["time"].mean(),
                    "avg_cost_err": df["cost_err"].mean(),
                }
            )

    df_result = pd.DataFrame(records)

    # ── Save as JSON ───────────────────────────────────────
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / f"{generator._name}-prec.json"
    df_result.to_json(out_file, orient="records", indent=2)
    print(f"[✓] Results written to {out_file}")

    return df_result


# ──────────────────────────────────────────────────────────
# 2) JSON  →  PNG
# ──────────────────────────────────────────────────────────
def plot_results_prec_from_json(
    generator_name: str,
    data_dir: str | Path = "output/data",
    img_dir: str | Path = "output/images",
) -> Path:
    """
    Re-create runtime / precision-vs-ε plot with a cleaner aesthetic.

    Returns
    -------
    Path to the saved PNG.
    """
    data_dir, img_dir = Path(data_dir), Path(img_dir)
    in_file = data_dir / f"{generator_name}-prec.json"
    if not in_file.is_file():
        raise FileNotFoundError(in_file)

    df = pd.read_json(in_file)

    fig, (ax_time, ax_err) = plt.subplots(1, 2, figsize=(12, 4.2))
    fig.suptitle(generator_name, x=0.01, ha="left")

    for i, (name, sub) in enumerate(df.groupby("name")):
        colour = _PALETTE[i % len(_PALETTE)]
        # style = "-" if i % 2 == 0 else "--"
        ax_time.plot(sub["reg"], sub["avg_time"], color=colour, marker="o", label=name)
        ax_err.plot(sub["reg"], sub["avg_cost_err"], color=colour, marker="o", label=name)

        # annotate minimum points
        min_idx = sub["avg_time"].idxmin()
        ax_time.annotate(
            f"{sub['avg_time'][min_idx]:.1f}",
            xy=(sub["reg"][min_idx], sub["avg_time"][min_idx]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=colour,
        )
        min_idx = sub["avg_cost_err"].idxmin()
        ax_err.annotate(
            f"{sub['avg_cost_err'][min_idx]:.1e}",
            xy=(sub["reg"][min_idx], sub["avg_cost_err"][min_idx]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=colour,
        )

    # axis cosmetics
    _apply_axes_style(ax_time, r"Regularisation $\varepsilon$")
    _apply_axes_style(ax_err, r"Regularisation $\varepsilon$")
    ax_time.set_title("Mean runtime", loc="left")
    ax_err.set_title("Mean cost error", loc="left")

    for ax in (ax_time, ax_err):
        ax.legend(
            loc="upper right",
            frameon=True,
            fontsize=8,
            borderpad=0.3,
            handlelength=1.2,
        )

    fig.tight_layout()
    img_dir.mkdir(parents=True, exist_ok=True)
    out_path = img_dir / f"{generator_name}-prec.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Plot saved to {out_path}")
    return out_path

def run_experiments_pushforward_save_json(
    generator: ProblemGenerator,
    regs: list[float],
    solvers: list[BaseSolver],
    data_dir: str | Path = "output/data",
    **kwargs,
):
    ## Gather metrics first
    records = []
    experiment = Experiment(
        name="Pushforward bias",
        solve_fn=measure_pushforward if generator._dim == 1 else measure_pushforward_2d
    )
    for solver in solvers:
        for reg in regs:
            df = experiment.run_on_problems(
                problems=generator.generate(),
                solver=solver,
                reg=reg,
                **kwargs,
            )

            records.append({
                "name": "PDLP" if solver.__name__ == "PDLPSolver" else "Sinkhorn",
                'reg': reg,
                'avg_barycentric_l2_bias': df["barycentric_l2_bias"].mean(),
                'avg_barycentric_w2_bias': df["barycentric_w2_bias"].mean(),
                'avg_monge_l2_bias': df["monge_l2_bias"].mean() if "monge_l2_bias" in df else np.nan,
                'avg_monge_w2_bias': df["monge_w2_bias"].mean() if "monge_l2_bias" in df else np.nan,
            })

    df_result = pd.DataFrame(records)

    # ── Save as JSON ───────────────────────────────────────
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / f"{generator._name}-stat.json"
    df_result.to_json(out_file, orient="records", indent=2)
    print(f"[✓] Results written to {out_file}")



def plot_results_pushforward_from_json(
    generator_name: str,
    data_dir: str | Path = "output/data",
    img_dir: str | Path = "output/images",
) -> Path:
    """
    Re-create 4-panel bias plot with upgraded styling.

    Returns
    -------
    Path to the saved PNG.
    """
    data_dir, img_dir = Path(data_dir), Path(img_dir)
    in_file = data_dir / f"{generator_name}-stat.json"
    if not in_file.is_file():
        raise FileNotFoundError(in_file)

    df = pd.read_json(in_file)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(generator_name, x=0.01, ha="left")

    metrics = [
        ("avg_barycentric_l2_bias", "Barycentric $L_2$ bias"),
        ("avg_barycentric_w2_bias", "Barycentric $W_2$ bias"),
        ("avg_monge_l2_bias", "Gradient $L_2$ bias"),
        ("avg_monge_w2_bias", "Gradient $W_2$ bias"),
    ]

    for i, (name, sub) in enumerate(df.groupby("name")):
        colour = _PALETTE[i % len(_PALETTE)]
        # style = "-" if i % 2 == 0 else "--"
        for ax, (col, title) in zip(axes.flat, metrics):
            ax.plot(sub["reg"], sub[col], color=colour, marker="o", label=name)

    for ax, (_, title) in zip(axes.flat, metrics):
        _apply_axes_style(ax, r"Regularisation $\varepsilon$")
        ax.set_title(title, loc="left")

    # single shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    fig.tight_layout()
    img_dir.mkdir(parents=True, exist_ok=True)
    out_path = img_dir / f"{generator_name}-stat.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Plot saved to {out_path}")
    return out_path

regs = [1e5, 10000.0, 1000.0, 100.0, 10.0, 1.0, 1e-1, 1e-2, 1e-3, 1e-4]
# regs = [10.0, 1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
sovers = [PDLPSolver, SinkhornTwoMarginalSolver]

experiment_settings = {
    "Gaussian1D": [2048, 4096, 8192],
    "Cauchy1D": [4096, 8192],
    "Gaussian2D": [128]
}

if __name__ == "__main__":
    for n in experiment_settings["Gaussian1D"]:
        # gaussian_generator = GaussianMixtureGenerator(
        #     name=f"Gaussian-1d-{n}p",
        #     dim=1,
        #     num_components=1,
        #     n_points=n,
        #     num_datasets=10,
        #     borders=[-1, 1],
        #     cost_fn=cost_euclid_squared,
        #     use_jax=False
        # )

        # run_experiments_prec_save_json(
        #     generator=gaussian_generator,
        #     regs=regs,
        #     solvers=sovers,
        #     tol=1e-6,
        #     maxiter=100000,
        # )

        plot_results_prec_from_json(
            generator_name=f"Gaussian-1d-{n}p",
        )

        # run_experiments_pushforward_save_json(
        #     generator=gaussian_generator,
        #     regs=regs,
        #     solvers=sovers,
        #     tol=1e-6,
        #     maxiter=100000,
        # )

        # plot_results_pushforward_from_json(
        #     generator_name=f"Gaussian-1d-{n}p",
        # )

    for n in experiment_settings["Cauchy1D"]:
        # cauchy_generator = CauchyGenerator(
        #     name=f"Cauchy-1d-{n}p",
        #     dim=1,
        #     n_points=n,
        #     num_datasets=10,
        #     borders=[-1, 1],
        #     cost_fn=cost_euclid_squared,
        #     use_jax=False
        # )

        # run_experiments_prec_save_json(
        #     generator=cauchy_generator,
        #     regs=regs,
        #     solvers=sovers,
        #     tol=1e-6,
        #     maxiter=100000,
        # )

        plot_results_prec_from_json(
            generator_name=f"Cauchy-1d-{n}p",
        )

        # run_experiments_pushforward_save_json(
        #     generator=cauchy_generator,
        #     regs=regs,
        #     solvers=sovers,
        #     tol=1e-6,
        #     maxiter=100000,
        # )

        # plot_results_pushforward_from_json(
        #     generator_name=f"Cauchy-1d-{n}p",
        # )

    # for n in experiment_settings["Gaussian2D"]:
    #     gaussian_generator = GaussianMixtureGenerator(
    #         name=f"Gaussian-2d-{n}p",
    #         dim=2,
    #         num_components=1,
    #         n_points=n,
    #         num_datasets=1,
    #         borders=[-6, 6],
    #         cost_fn=cost_euclid_squared,
    #         use_jax=False
    #     )

        # run_experiments_prec_save_json(
        #     generator=gaussian_generator,
        #     regs=regs,
        #     solvers=sovers,
        #     tol=1e-6,
        #     maxiter=100000,
        # )

        # plot_results_prec_from_json(
        #     generator_name=f"Gaussian-2d-{n}p",
        # )
        #
        # run_experiments_pushforward_save_json(
        #     generator=gaussian_generator,
        #     regs=regs,
        #     solvers=sovers,
        #     tol=1e-6,
        #     maxiter=100000,
        # )
        #
        # plot_results_pushforward_from_json(
        #     generator_name=f"Gaussian-2d-{n}p",
        # )