import isaacgym
import torch
from omegaconf import DictConfig
import hydra
from isaacgymenvs.distillation.dagger_trainer import Dagger
from isaacgymenvs.distillation.dagger_mobile_trainer import DaggerMobile


@hydra.main(config_name="dagger_config", config_path="../cfg")
def main(cfg: DictConfig):
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
    main()
