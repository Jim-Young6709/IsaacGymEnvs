import isaacgym
import torch
import os
import sys
import time
import subprocess
from pathlib import Path
from omegaconf import DictConfig
import hydra
from isaacgymenvs.distillation.dagger_trainer import Dagger
from isaacgymenvs.distillation.dagger_mobile_trainer import DaggerMobile


_RELAUNCH_ENV_KEY = "DAGGER_TRAINING_CHILD"
_RELAUNCH_DELAY_SECONDS = 5
_MAX_RETRY = 10
_RESUME_FROM_LATEST = True


def _get_latest_checkpoint_path(cfg: DictConfig) -> Path:
    save_dir = Path("dagger_ckpts") / str(cfg.experiment)
    return save_dir / "latest.pt"


def _maybe_set_resume_checkpoint(cfg: DictConfig) -> None:
    if not _RESUME_FROM_LATEST:
        return

    latest_ckpt = _get_latest_checkpoint_path(cfg)
    if latest_ckpt.is_file():
        cfg.dagger.load_ckpt_path = str(latest_ckpt)
        print("-----------------------------------------------------------")
        print(f"Auto-resuming from checkpoint: {latest_ckpt}")
        print("-----------------------------------------------------------")
    else:
        cfg.dagger.load_ckpt_path = None
        print("-----------------------------------------------------------")
        print(f"No checkpoint found at {latest_ckpt}. Launching from scratch.")
        print("-----------------------------------------------------------")


@hydra.main(config_name="dagger_config", config_path="../cfg")
def main(cfg: DictConfig):
    _maybe_set_resume_checkpoint(cfg)

    if 'Mobile' in cfg['task']['name']:
        dagger_trainer = DaggerMobile(
            cfg=cfg,
        )
    else:
        dagger_trainer = Dagger(
            cfg=cfg,
        )

    dagger_trainer.train()


if __name__ == "__main__":
    import torch._dynamo
    torch._dynamo.config.disable = True

    if os.environ.get(_RELAUNCH_ENV_KEY) == "1":
        main()
    else:
        child_env = os.environ.copy()
        child_env[_RELAUNCH_ENV_KEY] = "1"

        for _ in range(_MAX_RETRY):
            result = subprocess.run([sys.executable, *sys.argv], env=child_env)

            if result.returncode == 0:
                break

            if result.returncode in (-2, 130):
                print("Training stopped by user. Exiting launcher.")
                break

            print(
                f"Training crashed with code {result.returncode}. "
                f"Relaunching in {_RELAUNCH_DELAY_SECONDS}s..."
            )
            time.sleep(_RELAUNCH_DELAY_SECONDS)
