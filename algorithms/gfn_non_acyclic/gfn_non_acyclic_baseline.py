"""
Code for Trajectory Balance (TB) training.
For further details see: https://arxiv.org/abs/2301.12594 and https://arxiv.org/abs/2501.06148
"""

from functools import partial

import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import wandb

from algorithms.common.diffusion_related.init_model import init_model_non_acyclic
from algorithms.gfn_non_acyclic.gfn_non_acyclic_rnd import rnd_mcmc
from eval import discrepancies
from eval.utils import extract_last_entry, save_plot_images
from utils.mcmc_diagnostics import (
    compute_geweke_z,
    compute_rhat_summary,
    rhat_status,
    summarize_geweke,
)
from utils.print_utils import print_results


def gfn_non_acyclic_baseline(cfg, target, exp=None):
    key_gen = jax.random.PRNGKey(cfg.seed)

    dim = target.dim
    alg_cfg = cfg.algorithm
    batch_size = alg_cfg.batch_size

    target_xs = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    initial_dist = distrax.MultivariateNormalDiag(
        jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std
    )
    aux_tuple = (alg_cfg.model.gamma,)

    # Initialize the model
    key, key_gen = jax.random.split(key_gen)
    model_state = init_model_non_acyclic(key, dim, alg_cfg)

    rnd_eval_partial_base = partial(
        rnd_mcmc,
        aux_tuple=aux_tuple,
        target=target,
        num_steps=alg_cfg.eval_max_steps,
        step_name=alg_cfg.step_name,
        initial_dist=initial_dist,
    )

    logger = {
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
        "diag/rhat_stop_step": [],
        "diag/geweke_stop_step": [],
        "diag/rhat_max_rank_split": [],
        "diag/rhat_max_split": [],
        "diag/rhat_max_classical": [],
        "diag/rhat_status": [],
        "diag/geweke_max_abs_z": [],
        "diag/geweke_frac_gt_1p96": [],
        "diag/geweke_frac_gt_2p58": [],
        "diag/geweke_status": [],
    }
    for d in cfg.discrepancies:
        logger[f"discrepancies/{d}"] = []
        logger[f"discrepancies_rhat/{d}"] = []
        logger[f"discrepancies_geweke/{d}"] = []

    def _build_t_grid(max_steps: int, min_steps: int, n_points: int):
        grid = np.linspace(min_steps, max_steps, num=n_points, dtype=int)
        grid = np.clip(grid, min_steps, max_steps)
        return np.unique(grid)

    def _samples_at_step(trajectories, t_idx):
        t_idx = int(np.clip(t_idx, 0, trajectories.shape[1] - 1))
        return trajectories[:, t_idx]

    def _plot_diagnostic_dynamics(
        t_grid, rhat_curve, split_curve, classical_curve, gw196, gw258, stop_rhat, stop_geweke
    ):
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot()
        ax.plot(t_grid, rhat_curve, label="max rank-split R-hat", color="#1f77b4")
        ax.plot(
            t_grid,
            split_curve,
            label="max split R-hat (aux)",
            color="#1f77b4",
            alpha=0.55,
        )
        ax.plot(
            t_grid,
            classical_curve,
            label="max classical R-hat (aux)",
            color="#1f77b4",
            alpha=0.35,
        )
        ax.plot(t_grid, gw196, label="frac |Z|>1.96", color="#ff7f0e")
        ax.plot(t_grid, gw258, label="frac |Z|>2.58", color="#d62728")
        ax.axvline(stop_rhat, linestyle="--", color="#1f77b4", alpha=0.7, label="R-hat stop")
        ax.axvline(
            stop_geweke,
            linestyle="--",
            color="#2ca02c",
            alpha=0.7,
            label="Geweke stop",
        )
        ax.set_xlabel("chain length")
        ax.set_ylabel("diagnostic value")
        ax.grid(alpha=0.3)
        ax.legend()
        return fig

    def _plot_trajectory_lengths(traj_lengths):
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot()
        vals = np.asarray(traj_lengths)
        ax.hist(vals, bins=30, color="#6a5acd", alpha=0.85)
        mean_v = float(np.mean(vals))
        med_v = float(np.median(vals))
        q90 = float(np.quantile(vals, 0.9))
        ax.axvline(mean_v, color="black", linestyle="--", label=f"mean={mean_v:.1f}")
        ax.axvline(med_v, color="#2ca02c", linestyle="--", label=f"median={med_v:.1f}")
        ax.axvline(q90, color="#ff7f0e", linestyle="--", label=f"q90={q90:.1f}")
        ax.set_xlabel("trajectory length")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
        ax.legend()
        return fig

    def _plot_sample_stats_dynamics(trajectories, t_grid):
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot()
        mean_norm = []
        avg_std = []
        for t in t_grid:
            samples = np.asarray(trajectories[:, t - 1, :])
            mean_norm.append(float(np.linalg.norm(samples.mean(axis=0))))
            avg_std.append(float(samples.std(axis=0).mean()))
        ax.plot(t_grid, mean_norm, label="||E[x_t]||", color="#9467bd")
        ax.plot(t_grid, avg_std, label="mean std(x_t)", color="#17becf")
        ax.set_xlabel("chain length")
        ax.set_ylabel("sample statistics")
        ax.grid(alpha=0.3)
        ax.legend()
        return fig

    def _compute_diag_tracks(trajectories):
        diag_cfg = getattr(cfg.algorithm, "early_stop", None)
        min_steps = int(getattr(diag_cfg, "min_chain_length", 20))
        n_grid = int(getattr(diag_cfg, "n_checkpoints", 25))
        max_steps = int(trajectories.shape[1])
        t_grid = _build_t_grid(max_steps, min_steps, n_grid)

        rhat_curve = []
        split_curve = []
        classical_curve = []
        rhat_statuses = []
        gw_max_curve = []
        gw196_curve = []
        gw258_curve = []
        gw_status_curve = []

        for t in t_grid:
            chain_slice = np.asarray(trajectories[:, :t, :])
            rhat_summary = compute_rhat_summary(chain_slice)
            rmax = rhat_summary.max_rank_split_rhat
            rhat_curve.append(rmax)
            split_curve.append(rhat_summary.max_split_rhat)
            classical_curve.append(rhat_summary.max_classical_rhat)
            rhat_statuses.append(rhat_status(rmax))

            z = compute_geweke_z(
                chain_slice,
                frac1=float(getattr(diag_cfg.geweke, "frac1", 0.1)),
                frac2=float(getattr(diag_cfg.geweke, "frac2", 0.5)),
                estimator=str(getattr(diag_cfg.geweke, "spectral_estimator", "ar")),
            )
            gsum = summarize_geweke(z)
            gw_max_curve.append(gsum["max_abs_z"])
            gw196_curve.append(gsum["frac_abs_z_gt_1p96"])
            gw258_curve.append(gsum["frac_abs_z_gt_2p58"])
            gw_status_curve.append(gsum["status"])

        rhat_thr = float(getattr(diag_cfg.rhat, "threshold", 1.01))
        gw196_thr = float(getattr(diag_cfg.geweke, "max_frac_gt_1p96", 0.05))
        gw258_thr = float(getattr(diag_cfg.geweke, "max_frac_gt_2p58", 0.01))
        gw_max_thr = float(getattr(diag_cfg.geweke, "max_abs_z_threshold", 4.0))

        stop_rhat_idx = next((i for i, v in enumerate(rhat_curve) if v < rhat_thr), len(t_grid) - 1)
        stop_geweke_idx = next(
            (
                i
                for i, (m, f1, f2) in enumerate(zip(gw_max_curve, gw196_curve, gw258_curve))
                if (f1 <= gw196_thr and f2 <= gw258_thr and m <= gw_max_thr)
            ),
            len(t_grid) - 1,
        )
        return {
            "t_grid": np.asarray(t_grid),
            "rhat_curve": np.asarray(rhat_curve),
            "split_curve": np.asarray(split_curve),
            "classical_curve": np.asarray(classical_curve),
            "rhat_statuses": rhat_statuses,
            "gw_max_curve": np.asarray(gw_max_curve),
            "gw196_curve": np.asarray(gw196_curve),
            "gw258_curve": np.asarray(gw258_curve),
            "gw_statuses": gw_status_curve,
            "stop_rhat": int(t_grid[stop_rhat_idx]),
            "stop_geweke": int(t_grid[stop_geweke_idx]),
        }

    def eval_fn(model_state, key):
        params = (model_state.params,)
        trajectories, _, _, trajectories_length = rnd_eval_partial_base(
            key,
            model_state,
            *params,
            batch_size=cfg.eval_samples,
        )
        samples_full = trajectories[:, -1]

        diag = _compute_diag_tracks(trajectories)
        samples_rhat = _samples_at_step(trajectories, diag["stop_rhat"] - 1)
        samples_geweke = _samples_at_step(trajectories, diag["stop_geweke"] - 1)

        eval_sets = {
            "full": samples_full,
            "rhat": samples_rhat,
            "geweke": samples_geweke,
        }

        for suffix, sample_set in eval_sets.items():
            for d in cfg.discrepancies:
                value = (
                    getattr(discrepancies, f"compute_{d}")(target_xs, sample_set, cfg)
                    if target_xs is not None
                    else jnp.inf
                )
                prefix = "discrepancies" if suffix == "full" else f"discrepancies_{suffix}"
                logger[f"{prefix}/{d}"].append(value)

        logger["diag/rhat_stop_step"].append(diag["stop_rhat"])
        logger["diag/geweke_stop_step"].append(diag["stop_geweke"])
        logger["diag/rhat_max_rank_split"].append(float(diag["rhat_curve"][-1]))
        logger["diag/rhat_max_split"].append(float(diag["split_curve"][-1]))
        logger["diag/rhat_max_classical"].append(float(diag["classical_curve"][-1]))
        logger["diag/rhat_status"].append(diag["rhat_statuses"][-1])
        logger["diag/geweke_max_abs_z"].append(float(diag["gw_max_curve"][-1]))
        logger["diag/geweke_frac_gt_1p96"].append(float(diag["gw196_curve"][-1]))
        logger["diag/geweke_frac_gt_2p58"].append(float(diag["gw258_curve"][-1]))
        logger["diag/geweke_status"].append(diag["gw_statuses"][-1])

        logger.update(target.visualise(samples=samples_full, prefix="full"))
        logger.update(target.visualise(samples=samples_rhat, prefix="rhat_stop"))
        logger.update(target.visualise(samples=samples_geweke, prefix="geweke_stop"))

        fig_diag = _plot_diagnostic_dynamics(
            diag["t_grid"],
            diag["rhat_curve"],
            diag["split_curve"],
            diag["classical_curve"],
            diag["gw196_curve"],
            diag["gw258_curve"],
            diag["stop_rhat"],
            diag["stop_geweke"],
        )
        logger.update({"figures/diag_dynamics": [wandb.Image(fig_diag)]})
        plt.close(fig_diag)

        fig_len = _plot_trajectory_lengths(trajectories_length)
        logger.update({"figures/trajectory_length_hist": [wandb.Image(fig_len)]})
        plt.close(fig_len)

        fig_stats = _plot_sample_stats_dynamics(trajectories, diag["t_grid"])
        logger.update({"figures/sample_stats_dynamics": [wandb.Image(fig_stats)]})
        plt.close(fig_stats)
        plt.close("all")
        return logger

    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)

    ### Training phase
    for it in range(alg_cfg.iters):
        if (it % eval_freq == 0) or (it == alg_cfg.iters - 1):
            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(it)
            logger["stats/nfe"].append((it + 1) * batch_size)  # FIXME
            logger.update(eval_fn(model_state, key))
            last_entry = extract_last_entry(logger)
            if getattr(cfg, "save_local_plots", True):
                save_plot_images(last_entry, cfg, it)
            if cfg.use_cometml:
                metrics = {}
                for key, value in last_entry.items():
                    if isinstance(value, wandb.Image):
                        exp.log_image(value.image, name=key, step=it)
                    else:
                        metrics[key] = value
                exp.log_metrics(metrics, step=it)
        break
