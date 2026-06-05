#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required to detect available GPUs." >&2
    exit 1
  fi
  mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
fi

GPU_COUNT="${#GPU_IDS[@]}"
if (( GPU_COUNT == 0 )); then
  echo "No GPUs detected." >&2
  exit 1
fi

PIDS=()

wait_for_wave() {
  local status=0
  local pid

  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  PIDS=()
  return "${status}"
}

run_sample() {
  local gpu_id="$1"
  local num_envs="$2"
  local range_idx="$3"
  local hdf5_name="$4"
  local scene_hdf5_path="$5"

  CUDA_VISIBLE_DEVICES="${gpu_id}" python isaacgymenvs/presampling/pre_sample_robot_init_pose.py \
    task=DexMobileExpFull num_envs="${num_envs}" seed="${SEED}" \
    teacher.ckpt="${TEACHER_CKPT}" \
    task.env.scene.hdf5_path="${scene_hdf5_path}" \
    task.env.scene.batch_idx="${range_idx}" \
    task.cfg_override="SideConstrained" \
    presample.assume_obj_in_view_t0="${OBJECT_T0}" \
    presample.output_hdf5_name="${hdf5_name}" \
    presample.rand_cfg.base_init_range="${BASE_INIT_RANGE}" \
    headless=True
}

# ----------------------------------------------------------------------------
TABLE_HEIGHT_RANGES=(
  "0.0 0.0"
  "0.0 0.0"
  "0.0 0.0"
  "0.0 0.0"
  "0.0 0.0"
  "0.0 0.0"
  "0.0 0.0"
  "0.0 0.0"
)
NUM_ENVS_LIST=(1250)

BASE_INIT_RANGE="[[-1.0, -0.5, -0.3], [-0.6, 0.5, 0.3]]"
SEED=1
TASK_NAME="May21_shelf_t0inview_part5"
TEACHER_CKPT="./rl_ckpts/exp_shelf_May6.pth"
SCENE_DIR="./presampled_envs/scene_only"
SCENE_HDF5="May13_shelf10k_part5_post.hdf5"
HEADLESS="True"
SCENE_GEN_ONLY="False"
OBJECT_T0="True"
# ----------------------------------------------------------------------------

job_idx=0
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
    hdf5_name="${TASK_NAME}_idx${sample_idx}_0.0tableheight_${num_envs}.hdf5"
    scene_hdf5_path="${SCENE_DIR}/${SCENE_HDF5}"

    gpu_id="${GPU_IDS[$((job_idx % GPU_COUNT))]}"
    echo "Launching ${hdf5_name} on GPU ${gpu_id} with ${num_envs} envs"
    run_sample "${gpu_id}" "${num_envs}" "${range_idx}" "${hdf5_name}" "${scene_hdf5_path}" &
    PIDS+=("$!")
    job_idx=$((job_idx + 1))

    if (( ${#PIDS[@]} == GPU_COUNT )); then
      wait_for_wave
    fi
  done
done

if (( ${#PIDS[@]} > 0 )); then
  wait_for_wave
fi
