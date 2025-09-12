import isaacgym
import torch
from omegaconf import DictConfig
from hydra.utils import instantiate
import hydra
from isaacgymenvs.utils.common_utils import set_seed_and_precision, load_model, load_env
from isaacgymenvs.distillation.dagger_trainer import Dagger
from nmp.training.train_utils import setup_training_directory


@hydra.main(config_name="dagger_config", config_path="../cfg")
def main(cfg: DictConfig):
    dagger_trainer = Dagger(
        cfg=cfg,
    )


    
if __name__ == "__main__":
    import torch._dynamo
    torch._dynamo.config.disable = True

    main()
