#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_FILE_NAME="${LOG_FILE_NAME:-expert}"
MASTER_PORT="${MASTER_PORT:-29501}"
NUM_ENVS="${NUM_ENVS:-500}"
MESH_DIR="${MESH_DIR:-./meshes_final/default}"
EXPERT_REACHING_CONTROLLER="${EXPERT_REACHING_CONTROLLER:-fabric}" # CODEX

run_eval() {
  local name="$1"
  local seed="$2"
  local teacher_ckpt="$3"
  local cfg_override="$4"
  local batch_idx="$5"
  local hdf5_path="$6"

  torchrun --nnodes=1 --nproc_per_node=1 --master_port="${MASTER_PORT}" \
  isaacgymenvs/distillation/run_expert_eval.py \
  task=DexMobileExpFull num_envs="${NUM_ENVS}" seed="${seed}" multi_gpu=True \
  teacher.ckpt="${teacher_ckpt}" \
  task.env.mesh.mesh_dir="${MESH_DIR}" \
  task.cfg_override="${cfg_override}" experiment="${LOG_FILE_NAME}_${name}" \
  task.env.object_wrench.enable=False task.env.object_teleport.enable=False \
  task.env.scene.batch_idx="${batch_idx}" \
  model=transformer_mobile_local_t_auxdelta \
  eval.debug_visuals=False \
  +eval.expert_reaching_controller="${EXPERT_REACHING_CONTROLLER}" \
  task.env.scene.hdf5_path="${hdf5_path}"
}

run_eval shelf 9 "./rl_ckpts/exp_shelf_May6.pth" "SideConstrained" 0 \
  "./presampled_envs/final/May16_shelf_sr_0.4054_num_3259.hdf5"

run_eval box 11 "./rl_ckpts/exp_top_Apr9.pth" "TopdownConstrained" 0 \
  "./presampled_envs/final/May16_3xtop_sr_0.6531_num_12300.hdf5"

run_eval drawer 12 "./rl_ckpts/exp_top_Apr9.pth" "TopdownConstrained" 2 \
  "./presampled_envs/final/May16_3xtop_sr_0.6531_num_12300.hdf5"

run_eval tabletop 13 "./rl_ckpts/exp_top_Apr9.pth" "TopdownConstrained" 3 \
  "./presampled_envs/final/May16_3xtop_sr_0.6531_num_12300.hdf5"
