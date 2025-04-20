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



# # DRP Closed Loop
# # Loop through each task name and run the Python script
# for task in "${task_names[@]}"; do
#   echo "Running task: $task"
#   python3 run_drp_evals.py headless=True task.planner="DRP" task.task_type="static" task.task_name="$task"
# done

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


# Curobo PCD Open Loop
# Loop through each task name and run the Python script
for task in "${task_names[@]}"; do
  echo "Running task: $task"
  python3 run_drp_evals.py headless=True task.planner="Curobo_PCD" task.close_loop=False task.task_type="static" task.task_name="$task"
done