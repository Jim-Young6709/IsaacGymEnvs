import subprocess
import os
import shutil
import argparse
import torch


# Get CPU count allocated by SLURM
num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
num_threads = max(1, num_cpus // torch.cuda.device_count())  # Distribute threads across GPUs

os.environ["OMP_NUM_THREADS"] = str(num_threads)

print(f"Setting OMP_NUM_THREADS to {num_threads}")


if __name__ == "__main__":
    # argparse the following args:
    parser = argparse.ArgumentParser()

    # configs that just use default for most of the time
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_learning_epochs", type=int, default=1)
    parser.add_argument("--wandb_disable", action='store_true')
    parser.add_argument("--wandb_project", type=str, default='drp_dagger')
    parser.add_argument("--train_dir", type=str, default='runs_dagger')

    parser.add_argument("--dataset_path", type=str, default='')
    parser.add_argument("--expert_base_policy_url", "-eb", type=str, default='jimyoung6709/DRP')
    parser.add_argument("--resume", type=str, default='')
    parser.add_argument("--ckpt_path", type=str, default='None')
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_learning_iters", type=int, default=1000)
    parser.add_argument("--wandb_run_name", type=str, default='dagger')
    parser.add_argument("--start_batch_iter", type=int, default=0)
    parser.add_argument("--fabric_lockin", type=float, default=0.01)
    parser.add_argument("--step_expert", action='store_true')

    parser.add_argument("--loss_type", type=str, default='l1')
    parser.add_argument("--colli_reset", action='store_true')
    parser.add_argument("--colli_stepback", type=int, default=0)
    parser.add_argument("--capture_video", action='store_true')
    parser.add_argument("--skip_init_eval", action='store_true')
    parser.add_argument("--gpu_num", type=int, default=1)

    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = args.num_envs

    gpu_id = args.gpu_id
    batch_size = args.batch_size
    lr = args.lr
    num_learning_epochs = args.num_learning_epochs
    wandb_activate = not args.wandb_disable
    wandb_project = args.wandb_project
    train_dir = args.train_dir
    dataset_path = args.dataset_path
    resume = args.resume
    ckpt_path = args.ckpt_path
    num_envs = args.num_envs
    num_learning_iters = args.num_learning_iters
    wandb_run_name = args.wandb_run_name
    start_batch_iter = args.start_batch_iter
    fabric_lockin = args.fabric_lockin
    loss_type = args.loss_type
    colli_reset = args.colli_reset
    colli_stepback = args.colli_stepback
    capture_video = args.capture_video
    skip_init_eval = args.skip_init_eval
    gpu_num = args.gpu_num

    log_dir = os.path.join(args.train_dir, args.wandb_run_name)

    if os.path.exists(log_dir) and not args.resume:
        ans = input("WARNING: training directory ({}) already exists! \n type 'y' to overwrite, 'n' to resume (y/n)\n".format(log_dir))
        if ans == "y":
            print("removing previous runs")
            shutil.rmtree(log_dir)
        else:
            print("resuming from previous runs")

    gpu_cmd = {
        # "DISABLE_LAYER_NV_OPTIMUS_1": "1",
        # "DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1": "1",
        # "MESA_VK_DEVICE_SELECT": "10de:2230",
        # "DISPLAY": ":1",
        # "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        # "ENABLE_DEVICE_CHOOSER_LAYER": "1",
        # "VULKAN_DEVICE_INDEX": f"{gpu_id}",
        # "CUDA_VISIBLE_DEVICES": f"{gpu_id}",
    }

    for outer_epoch in range(1000):
        if os.path.exists(f"{log_dir}/nn/checkpoint_latest.pth"):
            resume = f"{log_dir}/nn/checkpoint_latest.pth"
            ckpt_path = 'None'

        command_list =[
            "torchrun", f"--nproc_per_node={args.gpu_num}", "isaacgymenvs/dagger.py",
            f"multi_gpu={args.gpu_num > 0}",
            f"task.env.hdf5_path={args.dataset_path}",
            f"ckpt_path={ckpt_path}", f"resume={resume}",
            f"num_envs={args.num_envs}", f"dagger.batch_size={args.batch_size}", 
            f"dagger.lr={args.lr}", f"dagger.num_learning_iterations={args.num_learning_iters}", f"dagger.num_learning_epochs={args.num_learning_epochs}",
            f"wandb_project={args.wandb_project}", f"wandb_activate={not args.wandb_disable}",
            f"wandb_run_name={args.wandb_run_name}", f"train_dir={args.train_dir}",
            f"dagger.loss_type={args.loss_type}", f"task.env.reset_on_collision={args.colli_reset}", f"task.env.step_back_on_collision={args.colli_stepback}",
            f"task.env.capture_video={args.capture_video}", f"task.env.batch_idx={outer_epoch + start_batch_iter}",
            f"task.fabric.lock_in_pos_err={args.fabric_lockin}", f"task.env.base_policy_url={args.expert_base_policy_url}",
            f"debug_training={args.skip_init_eval}", f"dagger.step_expert={args.step_expert}",
        ]

        # check if {log_dir}/nn/checkpoint_latest.pth exists, if so resume from that path
        print("Running command:", " ".join(command_list))
        subprocess.run(command_list, env={**os.environ, **gpu_cmd})
