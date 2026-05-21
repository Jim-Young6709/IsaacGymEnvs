import isaacgym
import torch

import json
from datetime import datetime
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, open_dict

from isaacgymenvs.distillation.dagger_mobile_trainer import DaggerMobile
from isaacgymenvs.utils.training_utils import colorprint


def _cfg(cfg, key, default=None):
    value = cfg.get(key, default)
    if value == "":
        return default
    return value


def _resolve_checkpoint_path(cfg: DictConfig) -> Path:
    ckpt_path = _cfg(cfg.eval, "ckpt_path") or _cfg(cfg.dagger, "load_ckpt_path")
    ckpt_path = Path(ckpt_path) if ckpt_path else Path("dagger_ckpts") / str(cfg.experiment) / "latest.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Eval checkpoint not found: {ckpt_path}. "
            "Set eval.ckpt_path or dagger.load_ckpt_path."
        )
    return ckpt_path


def _apply_eval_overrides(cfg: DictConfig, ckpt_path: Path) -> None:
    debug_visuals = bool(_cfg(cfg.eval, "debug_visuals", False))
    with open_dict(cfg):
        cfg.dagger.load_ckpt_path = str(ckpt_path)

        enable_viewer = _cfg(cfg.eval, "enable_viewer")
        if enable_viewer is not None:
            cfg.headless = not bool(enable_viewer)
        elif debug_visuals:
            cfg.headless = False
        else:
            cfg.headless = True

        if debug_visuals:
            cfg.force_render = True


def _scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().item()
        return value.detach().float().mean().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _mean_logs(logs_per_episode):
    if len(logs_per_episode) == 1:
        return logs_per_episode[0]
    averaged = {}
    for key in sorted({key for episode_logs in logs_per_episode for key in episode_logs}):
        values = [episode_logs[key] for episode_logs in logs_per_episode if key in episode_logs]
        if all(isinstance(value, (int, float)) for value in values):
            averaged[key] = float(sum(values) / len(values))
    return averaged


def _run_eval(trainer, num_episodes, include_env_extras):
    logs_per_episode = []
    for _ in range(num_episodes):
        logs = {key: _scalar(value) for key, value in trainer.eval().items()}
        if include_env_extras:
            logs = {**{key: _scalar(value) for key, value in trainer.env.extras.items()}, **logs}
        logs_per_episode.append(logs)
    return _mean_logs(logs_per_episode)


def _save_eval_logs(cfg: DictConfig, ckpt_path: Path, eval_logs: dict) -> Path:
    metrics_path = _cfg(cfg.eval, "save_metrics_path")
    if metrics_path is None:
        exp_name = str(cfg.experiment or cfg.wandb_name)
        exp_name = "".join(char if char.isalnum() or char in "-_." else "_" for char in exp_name)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        metrics_path = Path("logs") / "eval" / f"{exp_name}_{timestamp}.json"
    else:
        metrics_path = Path(metrics_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(ckpt_path),
        "num_episodes": int(_cfg(cfg.eval, "num_episodes", 1)),
        "metrics": {key: _scalar(value) for key, value in sorted(eval_logs.items())},
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return metrics_path


def _print_eval_logs(eval_logs: dict) -> None:
    colorprint("Eval metrics:", color="magenta")
    for key, value in sorted(eval_logs.items()):
        if isinstance(value, float):
            colorprint(f"{key}: {value:.4f}", color="green")
        elif isinstance(value, int):
            colorprint(f"{key}: {value}", color="green")


@hydra.main(config_name="dagger_config", config_path="../cfg")
def main(cfg: DictConfig):
    ckpt_path = _resolve_checkpoint_path(cfg)
    _apply_eval_overrides(cfg, ckpt_path)

    num_episodes = int(_cfg(cfg.eval, "num_episodes", 1))
    if num_episodes <= 0:
        raise ValueError(f"eval.num_episodes must be positive, got {num_episodes}")
    if cfg["task"]["type"] != "WBC":
        raise ValueError(
            "run_dagger_eval.py supports single-GPU WBC mobile DAgger eval only. "
            f"Unsupported task.type: {cfg['task']['type']}"
        )

    trainer = DaggerMobile(cfg=cfg)
    eval_logs = _run_eval(trainer, num_episodes, bool(_cfg(cfg.eval, "include_env_extras", True)))

    _print_eval_logs(eval_logs)
    if getattr(trainer, "use_wandb", False):
        wandb.log(eval_logs, step=trainer.total_steps)

    if bool(_cfg(cfg.eval, "save_metrics", True)):
        metrics_path = _save_eval_logs(cfg, ckpt_path, eval_logs)
        colorprint(f"Saved eval metrics to {metrics_path}", color="cyan")

    if getattr(trainer, "use_wandb", False):
        wandb.finish()


if __name__ == "__main__":
    import torch._dynamo
    torch._dynamo.config.disable = True
    main()
