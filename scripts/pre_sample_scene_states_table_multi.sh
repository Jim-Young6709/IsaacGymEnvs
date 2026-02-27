#!/usr/bin/env bash
set -euo pipefail

# Set these two variables only.
TABLE_HEIGHT_RANGES=(
  "0.7 0.8"
  "0.8 0.8"
  "0.0 0.0"
  "0.0 0.1"
  "0.0 0.8"
)
NUM_ENVS_LIST=(1024)

TASK="DexMobileExpFull"
SEED=1
TEACHER_CKPT="./ckpts/exp_table_Feb23.pth"
SCENE_DIR="./presampled_envs/scene_only"
ENABLE_VISER="False"
HEADLESS="True"
SCENE_GEN_ONLY="False"

for num_envs in "${NUM_ENVS_LIST[@]}"; do
  for range in "${TABLE_HEIGHT_RANGES[@]}"; do
    read -r z_min z_max <<< "$range"

    if [[ "$z_min" == "$z_max" ]]; then
      height_tag="$z_min"
    else
      height_tag="${z_min}-${z_max}"
    fi

    hdf5_name="table_multi_${height_tag}tableheight_${num_envs}.hdf5"
    scene_hdf5_path="${SCENE_DIR}/${hdf5_name}"

    python isaacgymenvs/distillation/pre_sample_scene_states_table_multi.py \
      num_envs="${num_envs}" task.env.scene.z_shift_range="[${z_min},${z_max}]" \
      presample.output_hdf5_name="${hdf5_name}"

    if [[ "$SCENE_GEN_ONLY" == "False" ]]; then
        python isaacgymenvs/distillation/pre_sample_robot_init_pose.py \
        task="${TASK}" num_envs="${num_envs}" seed="${SEED}" \
        teacher.ckpt="${TEACHER_CKPT}" \
        task.env.scene.hdf5_path="${scene_hdf5_path}" \
        presample.output_hdf5_name="${hdf5_name}" \
        task.env.enable_viser="${ENABLE_VISER}" headless="${HEADLESS}"
    fi
  done
done
