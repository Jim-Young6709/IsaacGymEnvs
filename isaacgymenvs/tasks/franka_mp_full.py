import time

import hydra
import isaacgym
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.franka_mp import FrankaMP
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.demo_loader import DemoLoader
from isaacgymenvs.utils.torch_jit_utils import *

from neural_mp.utils.pcd_utils import decompose_scene_pcd_params_obs, compute_scene_oracle_pcd
from collections import OrderedDict
from omegaconf import DictConfig
from tqdm import tqdm


from fabrics_sim.fabrics.franka_fabric_rl import FrankaFabricRL
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel



class FrankaMPFull(FrankaMP):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render, num_env_per_env=1):
        self.MAX_OBSTACLES = 40
        self.device = sim_device
        print("Before super init:")
        print("self.device:", self.device)
        print("sim_device:", sim_device)
        print("rl_device:", rl_device)

        # Demo loading
        hdf5_path = '/home/jimyoung/Neural_MP_Proj/neural_mp/datasets/hybrid1000.hdf5'
        self.demo_loader = DemoLoader(hdf5_path, cfg["env"]["numEnvs"])

        # need to change the logic here (3 layers of reset ; multiple start & goal in one DRP env ; multiple DRP envs in one IG env ; relaunch IG)
        self.batch = self.demo_loader.get_next_batch()

        self.start_config = torch.zeros((cfg["env"]["numEnvs"], 7), device=self.device)
        self.goal_config = torch.zeros((cfg["env"]["numEnvs"], 7), device=self.device)
        self.obstacle_configs = []

        for env_idx, demo in enumerate(self.batch):
            self.start_config[env_idx] = torch.tensor(demo['states'][0][:7], device=self.device)
            self.goal_config[env_idx] = torch.tensor(demo['states'][0][7:14], device=self.device)

            pcd_params = demo['states'][0][15:]
            obstacle_config = decompose_scene_pcd_params_obs(pcd_params)
            self.obstacle_configs.append(obstacle_config)

        self.obstacle_handles = []

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def _create_envs(self, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.cuboid_dims = []  # xyz
        self.capsule_dims = []  # r, l
        self.sphere_radii = []  # r

        # setup franka
        franka_dof_props = self._create_franka()
        franka_asset = self.franka_asset
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + self.MAX_OBSTACLES  # franka + obstacles
        max_agg_shapes = num_franka_shapes + self.MAX_OBSTACLES
        self.frankas = []
        self.envs = []

        num_robot_points = self.pcd_spec_dict['num_robot_points']
        num_scene_points = self.pcd_spec_dict['num_obstacle_points']
        num_target_points = self.pcd_spec_dict['num_target_points']
        self.combined_pcds = torch.cat(
            (
                torch.zeros(num_robot_points, 4, device=self.device),
                torch.ones(num_scene_points, 4, device=self.device),
                2 * torch.ones(num_target_points, 4, device=self.device),
            ),
            dim=0,
        ).repeat(self.num_envs, 1, 1)


        self.obstacle_count = 0
        self.max_objects_per_env = 20
        self.fabrics_world_dict = dict()

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            self.objects_per_env = 0

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            franka_actor = self.gym.create_actor(
                env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0
            )
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create obstacles using initial demo data
            env_obstacles = []
            (
                cuboid_dims, 
                cuboid_centers, 
                cuboid_quats,
                cylinder_radii, 
                cylinder_heights,
                cylinder_centers,
                cylinder_quats,
                *_
            ) = self.obstacle_configs[i]
            
            # num_cylinders = len(cylinder_radii) #pausing cylinders due to incorrect spawning. Likely an actor indexing issue.
            

            num_cubes = len(cuboid_dims)
            # cuboid_dims = cuboid_dims*0.00000001
            # TODO: GET RID OF THIS
            # cuboid_centers[:, 2] = 100
            
            
            # Create actual obstacles with proper sizes
            for j in range(self.MAX_OBSTACLES):
                if j < num_cubes:
                    # Create obstacle with actual size and position
                    obstacle_asset, obstacle_pose = self._create_cube(
                        pos=cuboid_centers[j].tolist(),
                        size=cuboid_dims[j].tolist(),
                        quat=cuboid_quats[j].tolist()
                    )
                    self._create_fabric_cubu(
                        pos=cuboid_centers[j].tolist(),
                        size=cuboid_dims[j].tolist(),
                        quat=cuboid_quats[j].tolist(),
                        env_id=i,
                    )
                    
                else:
                    pass 
                    # Create minimal placeholder obstacles far away
                    # obstacle_asset, obstacle_pose = self._create_cube(
                    #     pos=[100.0, 100.0, 100.0],
                    #     size=[0.001, 0.001, 0.001],
                    #     quat=[0, 0, 0, 1]
                    # )
                
                obstacle_actor = self.gym.create_actor(
                    env_ptr,
                    obstacle_asset,
                    obstacle_pose,
                    f"obstacle_{j}",
                    i,
                    1,
                    0
                )
                env_obstacles.append(obstacle_actor)

            # update max_objects_per_envs
            if self.objects_per_env > self.max_objects_per_env:
                self.max_objects_per_env = self.objects_per_env
                
            self.obstacle_handles.append(env_obstacles)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

            # compute the static scene pcd (currently only consider static scenes)
            static_scene_pcd = compute_scene_oracle_pcd(
                num_obstacle_points=num_scene_points,
                cuboid_dims=cuboid_dims,
                cuboid_centers=cuboid_centers,
                cuboid_quats=cuboid_quats,
            )
            self.combined_pcds[i, num_robot_points:num_robot_points+num_scene_points, :3] = torch.from_numpy(static_scene_pcd).to(self.device)

        
        
        self.fabrics_world_model = WorldMeshesModel(
            batch_size=self.num_envs,
            max_objects_per_env=self.max_objects_per_env,
            device=self.device,
            world_dict=self.fabrics_world_dict,
        )
        self.fabrics_object_ids, self.fabrics_object_indicator = self.fabrics_world_model.get_object_ids()
        
        # Create franka fabric
        self.franka_fabric = FrankaFabricRL(self.num_envs, self.device, "right_gripper")

        # Create integrator for the fabric dynamics.
        self.franka_integrator = DisplacementIntegrator(self.franka_fabric)
        
        cspace_dim = 7
        self.fabric_q = torch.zeros((self.num_envs, cspace_dim), dtype=torch.float, device=self.device)
        self.fabric_qd = torch.zeros((self.num_envs, cspace_dim), dtype=torch.float, device=self.device)
        self.fabric_qdd = torch.zeros((self.num_envs, cspace_dim), dtype=torch.float, device=self.device)
        
        
        # Setup data
        actor_num = 1 + self.MAX_OBSTACLES  # franka  + obstacles
        self.init_data(actor_num=actor_num)
    
    
    
    def _create_fabric_cubu(self, pos, size, quat, env_id):
                            # scale, pos, rot, obstacle_asset_options, env_ptr, i):
                        #         pos=[100.0, 100.0, 100.0],
                        # size=[0.001, 0.001, 0.001],
                        # quat=[0, 0, 0, 1]
        self.obstacle_count += 1
        self.objects_per_env += 1

        transform = list(pos) + list(quat)
        self.fabrics_world_dict[f"cube_{self.obstacle_count}"] = {
            "env_index": env_id,
            "type": "box",
            "scaling": " ".join(map(str, size)),
            "transform": " ".join(map(str, transform)),
        }
        return
    
    

    def update_obstacle_configs_from_batch(self, batch_data):
        """Update obstacle configurations from a new batch of demos."""
        self.obstacle_configs = []
        for demo in batch_data:
            pcd_params = demo['states'][0][15:]
            obstacle_config = decompose_scene_pcd_params_obs(pcd_params)
            self.obstacle_configs.append(obstacle_config)

    

    def plan_open_loop(self, use_controller=True):
        """
        debug base policy
        """
        obs_base = OrderedDict()

        for i in range(150):
            self.update_robot_pcds()
            obs_base["current_angles"] = self.states['q'][:, :7].clone()
            obs_base["goal_angles"] = self.goal_config.clone()
            obs_base["compute_pcd_params"] = self.combined_pcds.clone()
            base_action = self.base_model.policy.get_action(obs_dict=obs_base)
            abs_action = base_action + self.get_joint_angles()
            if use_controller:
                self.step(base_action)
                self._refresh()
            else:
                self.set_robot_joint_state(abs_action)
                self.gym.simulate(self.sim)
                self.render()
                self._refresh()

    def visualize_pcd_meshcat(self, env_idx: int=0):
        "for debug purposes"
        import meshcat
        import urchin
        from robofin.robots import FrankaRobot
        self.viz = meshcat.Visualizer()
        self.urdf = urchin.URDF.load(FrankaRobot.urdf)
        for idx, (k, v) in enumerate(self.urdf.visual_trimesh_fk(np.zeros(8)).items()):
            self.viz[f"robot/{idx}"].set_object(
                meshcat.geometry.TriangularMeshGeometry(k.vertices, k.faces),
                meshcat.geometry.MeshLambertMaterial(wireframe=False),
            )
            self.viz[f"robot/{idx}"].set_transform(v)
        pcd_rgb = np.zeros((3, 8192))
        pcd_rgb[0, :2048] = 1
        pcd_rgb[1, 2048:6144] = 1
        pcd_rgb[2, 6144:] = 1
        
        self.viz['pcd'].set_object(
            meshcat.geometry.PointCloud(
                position=self.combined_pcds[env_idx, :, :3].cpu().numpy().T,
                color=pcd_rgb,
                size=0.005,
            )
        )

    def reset_idx(self, env_ids=None):
        """
        TODO: need to re-write completely
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.start_config = tensor_clamp(self.start_config, self.franka_dof_lower_limits[:7], self.franka_dof_upper_limits[:7])

        self.goal_config = tensor_clamp(self.goal_config, self.franka_dof_lower_limits[:7], self.franka_dof_upper_limits[:7])

        self.goal_ee = self.get_ee_from_joint(self.goal_config)

        self.set_robot_joint_state(self.start_config, debug=True)
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def compute_reward(self, actions):
        self.check_robot_collision()
        current_angles = self.get_joint_angles()
        current_ee = self.get_ee_from_joint(current_angles)

        joint_err = torch.norm(current_angles - self.goal_config, dim=1)
        pos_err = torch.norm(current_ee[:, :3] - self.goal_ee[:, :3], dim=1)
        quat_err = orientation_error(self.goal_ee[:, 3:], current_ee[:, 3:])

        self.rew_buf[:], self.reset_buf[:], d = compute_franka_reward(
            self.reset_buf, self.progress_buf, joint_err, pos_err, quat_err, self.collision, self.max_episode_length
        )

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        if is_last_step:
            self.extras['successes'] = torch.mean(torch.where(d < 0.1, 1.0, 0.0)).item()
    
    
    
    def fabric_forward_kinematics(self, q):
        gripper_map = self.franka_fabric.get_taskmap("gripper")
        gripper_pos, _ = gripper_map(q, None)
        
        ee_x_axis_robot_frame = (gripper_pos[:, 3:6] - gripper_pos[:, 0:3])
        ee_x_axis_robot_frame = ee_x_axis_robot_frame / ee_x_axis_robot_frame.norm(p=2, dim=1, keepdim=True)
        ee_y_axis_robot_frame = (gripper_pos[:, 9:12] - gripper_pos[:, 0:3])
        ee_y_axis_robot_frame = ee_y_axis_robot_frame / ee_y_axis_robot_frame.norm(p=2, dim=1, keepdim=True)
        ee_z_axis_robot_frame = (gripper_pos[:, 15:18] - gripper_pos[:, 0:3])
        ee_z_axis_robot_frame = ee_z_axis_robot_frame / ee_z_axis_robot_frame.norm(p=2, dim=1, keepdim=True)

        R = torch.stack((ee_x_axis_robot_frame, ee_y_axis_robot_frame, ee_z_axis_robot_frame), dim=2)
        # convert from wxyz to xyzw
        q = matrix_to_quaternion(R)[:, [1, 2, 3, 0]] 
        ee_pose = torch.cat((gripper_pos[:, 0:3], q), dim=1)
        return ee_pose


    def pre_physics_step(self, actions):
        delta_actions = actions.clone().to(self.device)
        gripper_state = torch.Tensor([[0.035, 0.035]] * self.num_envs).to(self.device)

        delta_actions = torch.clamp(delta_actions, -self.cmd_limit, self.cmd_limit) / self.action_scale
        # delta_actions = delta_actions * self.cmd_limit / self.action_scale
        abs_actions = self.get_joint_angles() + delta_actions
        # abs_actions[:, 0:7] = self.fabric_q[:, 0:7] + delta_actions[:, 0:7]
        if abs_actions.shape[-1] == 7:
            abs_actions = torch.cat((abs_actions, gripper_state), dim=1)
       
        
        # cspace_target = torch.tensor([[0.02550676, -0.25173378, -0.3518326, -2.5239587, -0.11268669, 2.2990525, 0.5429185]], device=self.device).expand((self.num_envs, 7))
        cspace_target = abs_actions[:, 0:7]
        gripper_target = self.fabric_forward_kinematics(cspace_target) 
        
        cspace_toggle = torch.ones(self.num_envs, 1, device=self.device)           
        self.franka_fabric.set_features(
            gripper_target,
            "quaternion",
            cspace_target,
            cspace_toggle,
            self.fabric_q.detach(),
            self.fabric_qd.detach(),
            self.fabrics_object_ids,
            self.fabrics_object_indicator
        )
        
        timestep = 1/60.
        # Integrate fabric layer forward at 60 Hz. If policy action rate gets downsampled in the future, 
        # then change the value of 1 below to the downsample factor
        for i in range(1):
            self.fabric_q, self.fabric_qd, self.fabric_qdd = self.franka_integrator.step(
                self.fabric_q.detach(), self.fabric_qd.detach(), timestep # should be 1/60
            )

        abs_actions[:, :7] = torch.clone(self.fabric_q[:, 0:7]).contiguous()
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(abs_actions))
        
        
        vel_targets = torch.zeros_like(abs_actions, device=self.device)
        vel_targets[:, :7] = self.fabric_qd[:, 0:7]
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(vel_targets))
        

    def post_physics_step(self):
        super().post_physics_step()
        # reset the robot to start if collision occurs
        if sum(self.collision) > 0:
            collision_env_idx = self.collision.nonzero(as_tuple=False).flatten()
            self.set_robot_joint_state(self.start_config[collision_env_idx], env_ids=collision_env_idx)


    def update_robot_pcds(self):
        num_robot_points = self.pcd_spec_dict['num_robot_points']
        num_target_points = self.pcd_spec_dict['num_target_points']
        robot_pcd = self.gpu_fk_sampler.sample(self.get_joint_angles(), num_robot_points)
        target_pcd = self.gpu_fk_sampler.sample(self.goal_config, num_target_points)
        self.combined_pcds[:, :num_robot_points, :3] = robot_pcd
        self.combined_pcds[:, -num_target_points:, :3] = target_pcd

@hydra.main(config_name="config", config_path="../cfg/")
def launch_test(cfg: DictConfig):
    import isaacgymenvs
    from isaacgymenvs.learning import amp_continuous, amp_models, amp_network_builder, amp_players
    from isaacgymenvs.utils.rlgames_utils import (
        RLGPUAlgoObserver,
        RLGPUEnv,
        get_rlgames_env_creator,
    )
    from rl_games.algos_torch import model_builder
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    np.random.seed(0)
    torch.manual_seed(0)
    cfg_dict = omegaconf_to_dict(cfg)
    cfg_task = cfg_dict["task"]
    rl_device = cfg_dict["rl_device"]
    sim_device = cfg_dict["sim_device"]
    headless = cfg_dict["headless"]
    graphics_device_id = 0
    virtual_screen_capture = False
    force_render = False
    env = FrankaMPFull(cfg_task, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
    env.reset()

    total_error = 0
    num_failed_plans = 0
    num_plans = 1000
    for i in tqdm(range(num_plans)):
        import ipdb ; ipdb.set_trace()
        t1 = time.time()
        env.reset_idx()
        t2 = time.time()

        # print(f"Reset time: {t2 - t1}")
        # print("\nvalidation checking:")
        # env.set_robot_joint_state(env.start_config)
        # env.print_resampling_info(env.start_config)

        env.render()

    print(f"Average Error: {total_error / num_plans}")
    print(f"Percentage of failed plans: {num_failed_plans / num_plans * 100} ")

def orientation_error(desired, current):
    batch_diff = int(current.shape[0] / desired.shape[0])
    desired = desired.repeat(batch_diff, 1)
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return torch.abs((q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)).mean(dim=1))

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, joint_err, pos_err, quat_err, collision_status, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]
    exp_r = True
    if exp_r:
        exp_eef = torch.exp(-100*pos_err) + torch.exp(-100*quat_err)
        exp_joint = torch.exp(-100*joint_err)
        # exp_colli = 3*torch.exp(-100*collision_status)
        rewards = exp_eef + exp_joint # + exp_colli
    else:
        eef_reward = 1.0 - (torch.tanh(10*pos_err)+torch.tanh(10*quat_err))/2.0
        joint_reward = 1.0 - torch.tanh(10*joint_err)
        collision_reward = 1.0 - collision_status
        rewards = eef_reward + joint_reward + collision_reward
    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf, joint_err


if __name__ == "__main__":
    launch_test()
