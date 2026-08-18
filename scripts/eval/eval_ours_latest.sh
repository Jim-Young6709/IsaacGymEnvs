#!/usr/bin/env bash

LOG_FILE_NAME="ours_latest_240k"
CKPT_PATH="dagger_ckpts/grogu_ckpts/May22_wbc_shelf_top_tabletop_switch0.4_ep240.pt"
# CKPT_PATH="dagger_ckpts/grogu_ckpts/May25_wbc_7x512_shelf_top_table_switch0.4_ep250.pt"
# CKPT_PATH="dagger_ckpts/grogu_ckpts/May25_wbc_7x512_shelf_top_table_switch0.4_ep400.pt"
# CKPT_PATH="dagger_ckpts/grogu_ckpts/May25_wbc_7x512_shelf_top_table_switch0.4_ep500.pt"



# torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 \
# isaacgymenvs/distillation/run_dagger_eval.py \
# task=DexMobileExpFull num_envs=500 seed=13 multi_gpu=True \
# teacher.ckpt="./rl_ckpts/exp_shelf_May6.pth" \
# task.env.mesh.mesh_dir="./meshes_final/200_0.2_1.2" \
# task.cfg_override="SideConstrained" experiment="${LOG_FILE_NAME}_shelf" \
# task.env.object_wrench.enable=False task.env.object_teleport.enable=False \
# task.env.scene.batch_idx=0 \
# model=transformer_mobile_local_t_auxdelta \
# eval.debug_visuals=False \
# eval.ckpt_path=${CKPT_PATH} \
# task.env.scene.hdf5_path=./presampled_envs/final/May21_shelf_sr_0.3561_num_2508.hdf5


torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 \
isaacgymenvs/distillation/run_dagger_eval.py \
task=DexMobileExpFull num_envs=500 seed=3 multi_gpu=True \
teacher.ckpt="./rl_ckpts/exp_top_Apr9.pth" \
task.env.mesh.mesh_dir="./meshes_final/200_0.2_1.2" \
task.cfg_override="TopdownConstrained" experiment="${LOG_FILE_NAME}_box" \
task.env.object_wrench.enable=False task.env.object_teleport.enable=False \
task.env.scene.batch_idx=0 \
model=transformer_mobile_local_t_auxdelta \
eval.debug_visuals=False \
eval.ckpt_path=${CKPT_PATH} \
task.env.scene.hdf5_path=./presampled_envs/final/May21_box_sr_0.4540_num_512.hdf5


torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 \
isaacgymenvs/distillation/run_dagger_eval.py \
task=DexMobileExpFull num_envs=500 seed=12 multi_gpu=True \
teacher.ckpt="./rl_ckpts/exp_top_Apr9.pth" \
task.env.mesh.mesh_dir="./meshes_final/200_0.2_1.2" \
task.cfg_override="TopdownConstrained" experiment="${LOG_FILE_NAME}_drawer" \
task.env.object_wrench.enable=False task.env.object_teleport.enable=False \
task.env.scene.batch_idx=0 \
model=transformer_mobile_local_t_auxdelta \
eval.debug_visuals=False \
eval.ckpt_path=${CKPT_PATH} \
task.env.scene.hdf5_path=./presampled_envs/final/May21_drawer_sr_0.5289_num_512.hdf5


torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 \
isaacgymenvs/distillation/run_dagger_eval.py \
task=DexMobileExpFull num_envs=500 seed=13 multi_gpu=True \
teacher.ckpt="./rl_ckpts/exp_top_Apr9.pth" \
task.env.mesh.mesh_dir="./meshes_final/200_0.2_1.2" \
task.cfg_override="TopdownConstrained" experiment="${LOG_FILE_NAME}_tabletop" \
task.env.object_wrench.enable=False task.env.object_teleport.enable=False \
task.env.scene.batch_idx=0 \
model=transformer_mobile_local_t_auxdelta \
eval.debug_visuals=False \
eval.ckpt_path=${CKPT_PATH} \
task.env.scene.hdf5_path=./presampled_envs/final/May21_tabletopclutter_0.5close_sr_0.5080_num_512.hdf5

