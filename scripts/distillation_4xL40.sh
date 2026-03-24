#!/bin/bash
#SBATCH --job-name=dex_dagger
#SBATCH -N 1
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=60
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# optional: create log folder
mkdir -p logs

# activate your conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dex_drp

# go to your project directory
cd /path/to/your/project

# optional but often helpful for torch distributed
export OMP_NUM_THREADS=15

torchrun --nnodes=1 --nproc_per_node=4 --master_port=29501 \
    isaacgymenvs/distillation/run_dagger.py task=DexMobileExpFull \
    num_envs=1024 seed=1 multi_gpu=True headless=True wandb_activate=True \
    task.env.video_logging.capture=True \
    teacher.ckpt=./ckpts/exp_table_Feb23.pth \
    task.task.randomize=True \
    model=transformer_mobile_local_t_auxdelta dagger.learning_rate=1e-4 \
    chunk_size=1 dagger.teacher_forcing.enable=True dagger.eval_freq=0 \
    task.env.scene.hdf5_path=./presampled_envs/final/table_multi_Mar23_hybridbaseinit0.3close_nomobile_0.0-0.8tableheight_sr_0.7852_num_8192.hdf5 \
    experiment=Mar23_dagger_mobile_full-tablemulti_mobileobs0_Mar23_4x1024_curridr200k_local_t_auxdelta_2048_0_1024_expFeb23