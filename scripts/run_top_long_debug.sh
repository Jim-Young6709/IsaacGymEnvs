#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/top_long_logs}"
LOG_PATH="${LOG_DIR}/run.log"

TEACHER_CKPT_DEFAULT="${ROOT_DIR}/good_checkpoints/rel_curr_midheightTruex10.0_mass0.1_0.9_object_axis_curl3_flat1.0_recovery0.25_25-09-31-59/nn/last_rel_curr_midheightTruex10.0_mass0.1_0.9_object_axis_curl3_flat1.0_recovery0.25_ep_6400_rew_3103.01.pth"
TEACHER_CKPT="${TEACHER_CKPT:-$TEACHER_CKPT_DEFAULT}"

mkdir -p "${LOG_DIR}"

export ISAACGYM_URDF_CACHE_ROOT="${ISAACGYM_URDF_CACHE_ROOT:-/home/rayliu/.cache/isaacgym_urdf_overrides}"
export WARP_CACHE_ROOT="${WARP_CACHE_ROOT:-/home/rayliu/.cache/warp}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -f "${TEACHER_CKPT}" ]]; then
  echo "Teacher checkpoint not found: ${TEACHER_CKPT}" >&2
  exit 1
fi

CMD=(
  torchrun
  --nnodes=1
  --nproc_per_node=1
  --master_port=29501
  isaacgymenvs/distillation/run_dagger.py
  task=DexMobileDistillationTopLong
  num_envs=4
  seed=1
  multi_gpu=False
  headless=False
  wandb_activate=False
  wandb_group=dexgymNewTopLong
  task.env.video_logging.capture=False
  task.env.video_logging.envs=16
  "teacher.ckpt=${TEACHER_CKPT}"
  task.task.randomize=True
  "task.env.object_settings.mass_range=[0.2, 0.8]"
  task.env.object_teleport.enable=True
  task.env.object_wrench.enable=True
  task.env.object_wrench.success_curriculum.enable=False
  task.env.object_wrench.curri_steps=1
  task.env.teacher_obs_action_frame=eef
  task.env.enableDebugVis=True
  task.env.enable_viser=True
  task.env.mesh.mesh_dir=./meshes_side/long
  task.env.grasp_guide_idx=1
  model=transformer_mobile_local_t_auxdelta
  dagger.learning_rate=1e-4
  chunk_size=1
  dagger.eval_freq=0
  dagger.teacher_forcing.enable=True
  experiment=top_long
)

printf -v CMD_STR '%q ' "${CMD[@]}"

echo "Logging to ${LOG_PATH}"
echo "Teacher ckpt: ${TEACHER_CKPT}"

cd "${ROOT_DIR}"
script -f "${LOG_PATH}" -c "CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 ${CMD_STR}"
