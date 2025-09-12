import random
import numpy as np
import torch
from nmp.utils.visualization_utils import colorprint
from pathlib import Path
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

def set_seed_and_precision(seed=42, rank=0):
    seed = seed + rank
    
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    colorprint(f"Set random seed to {seed}", color="yellow")

def load_env(cfg):
    # load env with pcd_spec from train_config
    checkpoint_folder = cfg.checkpoint_folder
    train_config_file = Path(checkpoint_folder) / "train_config.yaml"
    if not train_config_file.exists():
        colorprint(f"Train config file {train_config_file} does not exist", color="red")
    else:
        colorprint(f"Load PCD config from train config file {train_config_file}", color="cyan")
        with open(train_config_file, "r") as f:
            train_config = yaml.safe_load(f)
        cfg.environment.task_cfg.pcd_spec = train_config["environment"]["task_cfg"]["pcd_spec"]
    env = instantiate(cfg.environment)
    return env

def load_model(checkpoint_folder, checkpoint_path):
    model_config_file = Path(checkpoint_folder) / "model_config.yaml"
    assert model_config_file.exists(), f"Model config file {model_config_file} does not exist"

    with open(model_config_file, "r") as f:
        model_config = OmegaConf.load(f)
    model = instantiate(model_config)
    
    checkpoint_path = Path(checkpoint_folder) / checkpoint_path
    assert checkpoint_path.exists(), f"Checkpoint path {checkpoint_path} does not exist"
    epoch, eval_success_rate, val_loss = model.load_checkpoint(checkpoint_path)
    colorprint(f"Loaded checkpoint from {checkpoint_path} at epoch {epoch+1}", color="cyan")
    colorprint(f"\tEval success rate: {eval_success_rate:}", color="cyan")
    colorprint(f"\tVal loss: {val_loss:}", color="cyan")
    
    model = torch.compile(model).to("cuda").float()
    
    return model, model_config
