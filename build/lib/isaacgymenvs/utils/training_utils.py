import random
import numpy as np
import torch
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

import imageio
import os
import numpy as np


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


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    num_cycles: float = 0.5, 
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def make_video(frames, logdir, name="video.mp4"):
    filename = os.path.join(logdir, name)
    frames = np.asarray(frames)
    with imageio.get_writer(filename, fps=20) as writer:
        for frame in frames:
            writer.append_data(frame)


def colorprint(text, color="blue"):
    """
    Print text in color.
    
    Args:
        text (str): Text to print
        color (str): Color to print in. Options: "red", "green", "yellow", "blue", "magenta", "cyan"
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "reset": "\033[0m"
    }
    
    if color not in colors:
        print(text)
        return
        
    print(f"{colors[color]}{text}{colors['reset']}")
