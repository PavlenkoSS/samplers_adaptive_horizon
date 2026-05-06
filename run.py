import os
from datetime import datetime

import hydra
import jax
import matplotlib

# import wandb
from comet_ml import Experiment
from omegaconf import DictConfig, OmegaConf

from utils.helper import flatten_dict, reset_device_memory
from utils.train_selector import get_train_fn


def _resolve_train_fn(alg_name: str):
    # Keep baseline and baseline_diagnostics as distinct entrypoints.
    if alg_name == "gfn_non_acyclic_baseline_diagnostics":
        from algorithms.gfn_non_acyclic.gfn_non_acyclic_baseline_diagnostics import (
            gfn_non_acyclic_baseline,
        )

        return gfn_non_acyclic_baseline
    return get_train_fn(alg_name)


@hydra.main(version_base=None, config_path="configs", config_name="base_conf")
def main(cfg: DictConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    # Load the chosen algorithm-specific configuration dynamically
    cfg = hydra.utils.instantiate(cfg)
    target = cfg.target.fn
    target_dim = int(target.dim)

    # Default checkpoint path now depends on environment + dimensionality.
    if hasattr(cfg, "algorithm") and hasattr(cfg.algorithm, "checkpoint_dir"):
        if getattr(cfg.algorithm, "checkpoint_dir", None) in (None, ""):
            cfg.algorithm.checkpoint_dir = os.path.join(
                "checkpoints",
                f"{cfg.target.name}_{target_dim}D",
                cfg.algorithm.name,
            )

    run_name = f"{cfg.cometml.prefix}_{cfg.algorithm.name}_{cfg.target.name}_{target_dim}_{datetime.now()}_seed{cfg.seed}"
    # if not cfg.wandb.get("name"):
    #     cfg.wandb.name = run_name

    print("JAX devices:", jax.devices())
    print("JAX default backend:", jax.default_backend())

    if not cfg.visualize_samples:
        matplotlib.use("agg")

    # if cfg.use_wandb:
    #     wandb.init(
    #         **cfg.wandb,
    #         group=f"{cfg.algorithm.name}",
    #         job_type=f"{cfg.target.name}_{target.dim}D",
    #         config=flatten_dict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)),
    #     )
    #     wandb.run.log_code(".")
    exp = None
    if cfg.use_cometml:
        exp = Experiment(**cfg.cometml)
        exp.set_name(run_name)
        exp.log_parameters(
            flatten_dict(
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            )
        )
    train_fn = _resolve_train_fn(cfg.algorithm.name)

    try:
        if cfg.use_jit:
            train_fn(cfg, target, exp)
        else:
            with jax.disable_jit():
                train_fn(cfg, target, exp)
        # if cfg.use_wandb:
        #     wandb.run.summary["error"] = None
        #     wandb.finish()
        if cfg.use_cometml:
            exp.log_other("error", None)
            exp.end()

    except Exception as e:
        # if cfg.use_wandb:
        #     wandb.run.summary["error"] = str(e)
        #     wandb.finish(exit_code=1)
        if cfg.use_cometml:
            exp.log_other("error", str(e))
            exp.end()
        reset_device_memory()
        raise e


if __name__ == "__main__":
    main()
