#!/usr/bin/env bash

torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 \
isaacgymenvs/distillation/run_dagger_eval.py \
task=DexMobileExpFull num_envs=500 seed=12 multi_gpu=True \
teacher.ckpt="./rl_ckpts/exp_shelf_May6.pth" \
task.env.mesh.mesh_dir="./meshes_final/default" \
task.cfg_override="SideConstrained" experiment=shelf_nolidar \
task.env.object_wrench.enable=False task.env.object_teleport.enable=False \
task.env.scene.batch_idx=0 \
model=transformer_mobile_local_t_auxdelta \
eval.debug_visuals=False \
eval.ckpt_path=dagger_ckpts/grogu_ckpts/May19_wbc_8x512_shelf_top_table_newfabric_nolidar_ep200.pt \
task.env.scene.hdf5_path=./presampled_envs/final/May16_shelf_sr_0.4054_num_3259.hdf5


torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 \
isaacgymenvs/distillation/run_dagger_eval.py \
task=DexMobileExpFull num_envs=500 seed=11 multi_gpu=True \
teacher.ckpt="./rl_ckpts/exp_top_Apr9.pth" \
task.env.mesh.mesh_dir="./meshes_final/default" \
task.cfg_override="TopdownConstrained" experiment=box_nolidar \
task.env.object_wrench.enable=False task.env.object_teleport.enable=False \
task.env.scene.batch_idx=0 \
model=transformer_mobile_local_t_auxdelta \
eval.debug_visuals=False \
eval.ckpt_path=dagger_ckpts/grogu_ckpts/May19_wbc_8x512_shelf_top_table_newfabric_nolidar_ep200.pt \
task.env.scene.hdf5_path=./presampled_envs/final/May16_3xtop_sr_0.6531_num_12300.hdf5


torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 \
isaacgymenvs/distillation/run_dagger_eval.py \
task=DexMobileExpFull num_envs=500 seed=9 multi_gpu=True \
teacher.ckpt="./rl_ckpts/exp_top_Apr9.pth" \
task.env.mesh.mesh_dir="./meshes_final/default" \
task.cfg_override="TopdownConstrained" experiment=drawer_nolidar \
task.env.object_wrench.enable=False task.env.object_teleport.enable=False \
task.env.scene.batch_idx=2 \
model=transformer_mobile_local_t_auxdelta \
eval.debug_visuals=False \
eval.ckpt_path=dagger_ckpts/grogu_ckpts/May19_wbc_8x512_shelf_top_table_newfabric_nolidar_ep200.pt \
task.env.scene.hdf5_path=./presampled_envs/final/May16_3xtop_sr_0.6531_num_12300.hdf5


torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 \
isaacgymenvs/distillation/run_dagger_eval.py \
task=DexMobileExpFull num_envs=500 seed=13 multi_gpu=True \
teacher.ckpt="./rl_ckpts/exp_top_Apr9.pth" \
task.env.mesh.mesh_dir="./meshes_final/default" \
task.cfg_override="TopdownConstrained" experiment=tabletop_nolidar \
task.env.object_wrench.enable=False task.env.object_teleport.enable=False \
task.env.scene.batch_idx=3 \
model=transformer_mobile_local_t_auxdelta \
eval.debug_visuals=False \
eval.ckpt_path=dagger_ckpts/grogu_ckpts/May19_wbc_8x512_shelf_top_table_newfabric_nolidar_ep200.pt \
task.env.scene.hdf5_path=./presampled_envs/final/May16_3xtop_sr_0.6531_num_12300.hdf5

