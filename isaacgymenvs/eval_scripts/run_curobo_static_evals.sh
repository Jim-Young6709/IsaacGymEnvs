#!/bin/bash

# List of task names
task_names=(
  static_box_ff
  static_box_ft
  static_box_tf
  static_box_tt
  static_cage_ff
  static_cage_ft
  static_cage_tf
  static_cage_tt
  static_cubby_ff
  static_cubby_ft
  static_cubby_tf
  static_cubby_tt
  static_dishwasher_ff
  static_dishwasher_ft
  static_dishwasher_tf
  static_dishwasher_tt
  static_hybrid_ff
  static_hybrid_ft
  static_hybrid_tf
  static_hybrid_tt
  static_microwave_ff
  static_microwave_ft
  static_microwave_tf
  static_microwave_tt
  static_shelf_ff
  static_shelf_ft
  static_shelf_tf
  static_shelf_tt
  static_tableonly_ff
  static_wallcabinet_ff
  static_wallcabinet_ft
  static_wallcabinet_tf
  static_wallcabinet_tt
)



# DRPNeuralMP Closed Loop
# Loop through each task name and run the Python script
for task in "${task_names[@]}"; do
  echo "Running task: $task"
  python3 run_drp_evals.py headless=True task.planner="DRP_ACT" task.task_type="static" task.task_name="$task"
done

# # Curobo Closed Loop
# # Loop through each task name and run the Python script
# for task in "${task_names[@]}"; do
#   echo "Running task: $task"
#   python3 run_drp_evals.py headless=True task.planner="Curobo" task.task_type="static" task.task_name="$task"
# done

# # Curobo Open Loop
# # Loop through each task name and run the Python script
# for task in "${task_names[@]}"; do
#   echo "Running task: $task"
#   python3 run_drp_evals.py headless=True task.planner="Curobo" task.close_loop=False task.task_type="static" task.task_name="$task"
# done

# # Curobo PCD Open Loop
# # Loop through each task name and run the Python script
# for task in "${task_names[@]}"; do
#   echo "Running task: $task"
#   python3 run_drp_evals.py headless=True task.planner="Curobo_PCD" task.close_loop=False task.task_type="static" task.task_name="$task"
# done



# Curobo Closed Loop Quasi dynamic
# python3 run_drp_evals.py headless=True task.planner="Curobo" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_1" task.use_speed_norm=False
# python3 run_drp_evals.py headless=True task.planner="Curobo" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_2" task.use_speed_norm=False
# python3 run_drp_evals.py headless=True task.planner="Curobo" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_3" task.use_speed_norm=False

# python3 run_drp_evals.py headless=True task.planner="Curobo" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_3" task.use_speed_norm=True
# python3 run_drp_evals.py headless=True task.planner="Curobo" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_1" task.use_speed_norm=True
# python3 run_drp_evals.py headless=True task.planner="Curobo" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_2" task.use_speed_norm=True


# # Curobo Closed Loop Dynamic Goal Blocking without RAP
# python3 run_drp_evals.py headless=True task.planner="Curobo" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_1" task.use_speed_norm=True task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="Curobo" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_2" task.use_speed_norm=True task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="Curobo" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_3" task.use_speed_norm=True task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="Curobo" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_4" task.use_speed_norm=True task.use_artificial_potential=False


# # Curobo Closed Loop Quasi Dynamic Voxel
# python3 run_drp_evals.py headless=True task.planner="Curobo_PCD" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_1" task.use_speed_norm=True task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="Curobo_PCD" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_2" task.use_speed_norm=True task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="Curobo_PCD" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_3" task.use_speed_norm=True task.use_artificial_potential=False


# # Curobo Closed Loop Dynamic Goal Blocking with RAP Voxel
# python3 run_drp_evals.py headless=True task.planner="Curobo_PCD" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_1" task.use_speed_norm=True task.use_artificial_potential=True
# python3 run_drp_evals.py headless=True task.planner="Curobo_PCD" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_2" task.use_speed_norm=True task.use_artificial_potential=True
# python3 run_drp_evals.py headless=True task.planner="Curobo_PCD" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_3" task.use_speed_norm=True task.use_artificial_potential=True
# python3 run_drp_evals.py headless=True task.planner="Curobo_PCD" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_4" task.use_speed_norm=True task.use_artificial_potential=True


# # NeuralMP TTO Open Loop
# # Loop through each task name and run the Python script
# for task in "${task_names[@]}"; do
#   echo "Running task: $task"
#   python3 run_drp_evals.py headless=True task.planner="DRPNeuralMP" task.task_type="static" task.task_name="$task" task.close_loop=False env.episodeLength=300
# done



# # Curobo Closed Loop Quasi Dynamic Voxel
# python3 run_drp_evals.py headless=True task.planner="Curobo_PCD" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_1" task.use_speed_norm=True task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="Curobo_PCD" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_2" task.use_speed_norm=True task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="Curobo_PCD" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_3" task.use_speed_norm=True task.use_artificial_potential=False


# # Curobo Closed Loop Goal Blocking Voxel
# python3 run_drp_evals.py headless=True task.planner="Curobo_PCD" task.close_loop=True task.task_type="goal_blocker" task.task_name="level_1" task.use_speed_norm=True task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="Curobo_PCD" task.close_loop=True task.task_type="goal_blocker" task.task_name="level_2" task.use_speed_norm=True task.use_artificial_potential=False


# # NeuralMP Closed Loop Quasi Dynamic
# python3 run_drp_evals.py headless=True task.planner="DRPNeuralMP" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_1" task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="DRPNeuralMP" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_2" task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="DRPNeuralMP" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_3" task.use_artificial_potential=False



# # RMP Only Closed Loop Static
# # Loop through each task name and run the Python script
# for task in "${task_names[@]}"; do
#   echo "Running task: $task"
#   python3 run_drp_evals.py headless=True task.planner="RMP_Only" task.task_type="static" task.task_name="$task" task.close_loop=True
# done


# # RMP Only Closed Loop Quasi Dynamic
# python3 run_drp_evals.py headless=True task.planner="RMP_Only" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_1" task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="RMP_Only" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_2" task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="RMP_Only" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_3" task.use_artificial_potential=False

# # RMP Only Closed Loop Goal Blocking
# python3 run_drp_evals.py headless=True task.planner="RMP_Only" task.close_loop=True task.task_type="goal_blocker" task.task_name="level_1" task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="RMP_Only" task.close_loop=True task.task_type="goal_blocker" task.task_name="level_2" task.use_artificial_potential=False


# # RMP Only Closed Loop Dynamic Goal Blocking
# python3 run_drp_evals.py headless=True task.planner="RMP_Only" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_1" 
# python3 run_drp_evals.py headless=True task.planner="RMP_Only" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_2" 
# python3 run_drp_evals.py headless=True task.planner="RMP_Only" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_3"
# python3 run_drp_evals.py headless=True task.planner="RMP_Only" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_4"


# RMP Only Closed Loop Dynamic Goal Blocking
# python3 run_drp_evals.py headless=True task.planner="DRPNeuralMP" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_1" 
# python3 run_drp_evals.py headless=True task.planner="DRPNeuralMP" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_2" 
# python3 run_drp_evals.py headless=True task.planner="DRPNeuralMP" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_3"
# python3 run_drp_evals.py headless=True task.planner="DRPNeuralMP" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_4"

# Closed Loop Quasi Dynamic
# python3 run_drp_evals.py headless=True task.planner="DRP_ACT" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_1" task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="DRP_ACT" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_2" task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="DRP_ACT" task.close_loop=True task.task_type="quasi_dynamic" task.task_name="level_3" task.use_artificial_potential=False

# # Closed Loop Goal Blocking Voxel
# python3 run_drp_evals.py headless=True task.planner="DRP_ACT" task.close_loop=True task.task_type="goal_blocker" task.task_name="level_1" task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="DRP_ACT" task.close_loop=True task.task_type="goal_blocker" task.task_name="level_2" task.use_artificial_potential=False

# # Closed Loop Dynamic Goal Blocking
# python3 run_drp_evals.py headless=True task.planner="DRP_ACT" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_1" task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="DRP_ACT" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_2" task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="DRP_ACT" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_3" task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="DRP_ACT" task.close_loop=True task.task_type="dynamic_goal_blocker" task.task_name="level_4" task.use_artificial_potential=False

# # floating
# python3 run_drp_evals.py headless=True task.planner="DRP_ACT" task.close_loop=True task.task_type="floating" task.task_name="level_1" task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="DRP_ACT" task.close_loop=True task.task_type="floating" task.task_name="level_2" task.use_artificial_potential=False

# # floating
# python3 run_drp_evals.py headless=True task.planner="DRPNeuralMP" task.close_loop=True task.task_type="floating" task.task_name="level_1" task.use_artificial_potential=False
# python3 run_drp_evals.py headless=True task.planner="DRPNeuralMP" task.close_loop=True task.task_type="floating" task.task_name="level_2" task.use_artificial_potential=False
