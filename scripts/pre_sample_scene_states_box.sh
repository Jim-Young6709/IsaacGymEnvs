#!/usr/bin/env bash
set -euo pipefail

# Set these two variables only.
TABLE_HEIGHT_RANGES=(
  "0.0 0.8"
  "0.0 0.8"
  "0.0 0.8"
  "0.0 0.8"
  "0.0 0.8"
  "0.0 0.8"
  "0.0 0.8"
  "0.0 0.8"
  "0.0 0.8"
  "0.0 0.8"
)
NUM_ENVS_LIST=(1000)

BASE_INIT_RANGE="[[-1.0, -0.5, -1.5], [-0.2, 0.5, 1.5]]"
SEED=1
TASK_NAME="inhand_part2_box_top_t0inview_Apr9"
TEACHER_CKPT="./rl_ckpts/exp_top_Apr9.pth"
SCENE_DIR="./presampled_envs/scene_only"
HEADLESS="True"
SCENE_GEN_ONLY="False"
OBJECT_T0="True"

for num_envs in "${NUM_ENVS_LIST[@]}"; do
  for range_idx in "${!TABLE_HEIGHT_RANGES[@]}"; do
    range="${TABLE_HEIGHT_RANGES[$range_idx]}"
    read -r z_min z_max <<< "$range"

    if [[ "$z_min" == "$z_max" ]]; then
      height_tag="$z_min"
    else
      height_tag="${z_min}-${z_max}"
    fi

    sample_idx=$((range_idx + 1))
    hdf5_name="${TASK_NAME}_idx${sample_idx}_0.0-0.8tableheight_${num_envs}.hdf5"
    scene_hdf5_path="${SCENE_DIR}/box10k_Apr9_inhand_part2.hdf5"

    python isaacgymenvs/presampling/pre_sample_robot_init_pose.py \
    task=DexMobileExpFull num_envs="${num_envs}" seed="${SEED}" \
    teacher.ckpt="${TEACHER_CKPT}" \
    task.env.scene.hdf5_path="${scene_hdf5_path}" \
    task.env.scene.batch_idx="${range_idx}" \
    presample.assume_obj_in_view_t0="${OBJECT_T0}" \
    presample.output_hdf5_name="${hdf5_name}" \
    presample.rand_cfg.base_init_range="${BASE_INIT_RANGE}" \
    headless=True
  done
done
