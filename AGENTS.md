# Repository Guidelines

## Project Structure & Module Organization
- Core package: `isaacgymenvs/`.
- Training entry point: `isaacgymenvs/train.py`.
- Task environments: `isaacgymenvs/tasks/` (many task variants, plus `*_dummy.py` experimental copies).
- Distillation pipeline: `isaacgymenvs/distillation/`.
- Runtime/model configs: `isaacgymenvs/cfg/` (`task/`, `train/`, `model/`, `pbt/`).
- Utilities and shared math/helpers: `isaacgymenvs/utils/`, `isaacgymenvs/model/`, `isaacgymenvs/learning/`.
- Robot and scene assets: `assets/`, `meshes*/`.
- Reference docs: `docs/`.

## Build, Test, and Development Commands
- `pip install -e .`: install this repo in editable mode.
- `python isaacgymenvs/train.py task=DexExpTop headless=True`: run a training job with Hydra overrides.
- `python isaacgymenvs/distillation/run_dagger.py`: start distillation training using `cfg/dagger_config.yaml`.
- `pre-commit run --all-files`: run repository hooks (currently `codespell`).
- `python -m compileall isaacgymenvs`: quick syntax sanity check across the package.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and standard PEP 8 naming.
- Functions/variables/files: `snake_case`; classes: `PascalCase`; constants: `UPPER_SNAKE_CASE`.
- Keep task/config naming aligned (example: `cfg/task/DexExpTop.yaml` with `cfg/train/DexExpTopPPO.yaml`).
- Use Hydra overrides rather than hardcoding run-time values.

## Testing Guidelines
- There is no dedicated `tests/` suite in this branch; use targeted smoke checks.
- For environment changes, run at least one short headless training launch and confirm no startup/runtime errors.
- For config-only changes, validate by launching the affected task and train config pair.
- Before opening a PR, run `pre-commit run --all-files` and `python -m compileall isaacgymenvs`.

## Commit & Pull Request Guidelines
- Existing history favors short, imperative commit subjects (for example: `minor fix`, `update inference code`) and standard merge commits.
- Recommended format: `<area>: <brief imperative summary>` (example: `distillation: fix teacher policy device placement`).
- PRs should include: purpose/scope, key config overrides used for validation, and links to related issues.
- Attach logs or screenshots only when they clarify behavior changes (training curves, success metrics, viewer output).

## Security & Configuration Tips
- Do not commit large generated artifacts (`runs/`, `videos/`, `wandb/`, checkpoints) unless explicitly required.
- Keep machine-specific paths and secrets out of tracked configs; prefer local overrides at runtime.

# Project Overview: Multi-Teacher Distillation for Dexterous Grasping

---

## 1. Project goal

Train a **dexterous grasping policy for any object** using **sim-to-real teacher–student distillation**.

Teachers are trained with RL for each task. Relevant files:

* `/home/rayliu/grogu/IsaacGymEnvs/isaacgymenvs/tasks/franka_leap.py`
* `/home/rayliu/grogu/IsaacGymEnvs/isaacgymenvs/tasks/franka_leap_pick_table_side.py`
* `/home/rayliu/grogu/IsaacGymEnvs/isaacgymenvs/tasks/franka_leap_pick_table.py`

Multiple teachers are distilled into a **single student policy**.

The differences between teachers are based on:

* **Grasp type** (e.g. side grasp, top-down grasp)
* **Grasp mode** (e.g. open fingers envelop, curled thumb tight clench)
* **Object types** (e.g. small objects vs. large objects)
* **Environment definition** (e.g. tabletop or shelves)

---

## 2. Teacher vs. mobile environments

Each teacher environment used for distillation has a corresponding **mobile** version.

Example:

* `franka_leap_pick_table.py`
* `franka_leap_mobile_pick_table.py`

There is a shared parent class:

* `/home/rayliu/grogu/IsaacGymEnvs/isaacgymenvs/tasks/franka_leap_mobile.py`

The mobile environment adds **motion planning** using a framework called **fabrics**.

You can treat fabrics as a black box that:

* Takes in current `q`, `qd`, `qdd`, and a target pose
* Outputs motion toward the target pose

During execution:

* Fabrics is used for navigation
* The system switches to the **teacher RL policy** once the robot is close enough to the object for grasping

---

## 3. Rules for you (AI coding agent)

* You are allowed and encouraged to **read any file**
* You are only allowed to **write into dummy files**, which are copies of current files
* CODE STYLE RULE (strict config keys): unless I explicitly ask for fallback behavior, do **not** use fallback defaults for config/arg access (for example, avoid `dict.get("k", default)`); use strict key access like `dict["k"]` so missing config keys fail fast.

Example:

* Real file:
  `/home/rayliu/grogu/IsaacGymEnvs/isaacgymenvs/tasks/franka_leap.py`

* Dummy file:
  `/home/rayliu/grogu/IsaacGymEnvs/isaacgymenvs/tasks/franka_leap_dummy.py`

All changes, including inline edits and deletions, must be clearly marked with comments:

```python
# CODEX
```

I will manually review your changes and decide which ones to apply.

---

## 4. Current goals

### 4.1 Unify environments for distillation

We should follow a **tree structure**:

* For each major modification that requires overriding a parent function, create a **child file**
* Share as much logic as possible across environments

Minor differences should **not** require new env files. These include:

* Dataset paths
* Curl shapes
* Other small configuration differences

These should be handled through **configs**, which act as specifications for individual environments.

---

### 4.2 Environment specifications

The plan is to define environments using a list of specs, for example:

```json
{
  "env": "Side",
  "dataset": "ycb_small_v2",
  "variant": "default",
  "switch": "contact_then_teacher",
  "planner": "rmp_v1",
  "obs": "student_v1"
}
```

Using these specs, we:

* Find the correct env file
* Load its config
* Modify the config according to the remaining fields in the spec

---

## 5. Motivation: multi-teacher distillation

In
`/home/rayliu/grogu/IsaacGymEnvs/isaacgymenvs/distillation/dagger_mobile_trainer.py`

we currently:

* Do distributed training
* Use only a **single environment**

We should instead:

* Load **different environments using specs**
* Evenly split them across available GPUs
* Each GPU runs one environment

(We may later load multiple envs on the same GPU, but this is not required now.)

---

## 6. Student checkpoint specifications (not priority)

Another goal is to properly document student models by:

* Storing a JSON file per `.pt` checkpoint
* Including:

  * The full spec
  * Success rate

This would allow easy filtering for the highest-success student checkpoint.

This is **not a priority right now**.

For now:

* Treat the **student checkpoint** as part of the distillation spec

## 7. Confirm you read this
Upon reading this first time, say "Hello, I've read your AGENT.MD file yeah!!!" in your next reply.

## 8. Progress
<!-- CODEX+ -->
### 2026-02-25
- Side-pick distillation follow-ups:
  1. [DONE] Fix the success metric for train and eval
  2. [DONE] Quickly add a video for logging training behavior
  3. [DONE] Check EEF teacher and decide which to use
  4. [DONE] Change fabric motion planning + correct action scale
  5. [DONE] Future task: make `per_ep_instant` delayed/latching at reset (same logging style as `per_ep`) instead of immediate snapshot logging
- Success metric findings (distillation):
  - Current env `per_ep` / `per_ep_instant` are snapshot/latch style and can be misleading with variable episode lengths.
  - Trainer early reset uses integer step streak (`count_reaching >= reaching_reset_threshold`), while env success uses float duration (`success_duration >= success_timeout`).
  - Even after setting `success_timeout = reaching_reset_threshold * dt * controlFrequencyInv`, boundary precision can undercount success at reset (trainer sees success, env `per_ep` may still be false at exact threshold).
  - Recommended fix: align success criterion to step-count logic in env logging path (or use an epsilon / slightly lower timeout) so trainer reset criterion and env success metric use the same event definition.
- ~~Potential leak from non-scalar `env.extras` logging (`time_outs`)~~ [DONE: removed `time_outs` from extras; current extras/logged metrics are scalarized]

### 2026-02-25 (New TODOs)
- 1. Change teacher reward:
  - start at 0.1 and gradually increase
  - add stronger wrench
  - see if we can do this to the sigmoid gated reward
- 2. Implement multi-teacher specs
- 3. Trim RL init poses
- 4. Train left grasp and right grasp, based on an input to the policy
- 5. Random init quaternion and pose
- 6. Make target pose change
- 7. Make hand quaternion change based on object pose
- 8. Debug memory issue (still exists)

### 2026-03-12
- [DONE] RL side-pick reset curriculum supports success-gated section unlock:
  - Added `right_section_curriculum_mode: success|steps`
  - Added success-gated unlock knobs (`right_section_curriculum_success_threshold`, `right_section_curriculum_success_hold_steps`)
  - Logs active section count to wandb via env extras: `curriculum/right_section_active_sections`
- [DONE] Right-section curriculum now supports true zero start:
  - Active right sections can start at `0` and increase over time (instead of forcing min `1`)
- [DONE] RL side-pick left-only fallback clarified and preserved:
  - Left-only behavior when `side_mode=left`, `right_section_sampling_enable=False`, `right_section_curriculum_enable=False`, `debug_disable_left_pose_bank=False`
- [DONE] Distillation mass range wiring:
  - `env.object_settings.mass_range` now sampled in `franka_leap_mobile_distillation.py` and applied via cached URDF mass patching
- [DONE] URDF override cache path made user-private + configurable:
  - Default cache root: `~/.cache/isaacgym_urdf_overrides`
  - Env override: `ISAACGYM_URDF_CACHE_ROOT=/path/to/cache`
- [DONE] Distillation reset/teleport anti-collision guard:
  - Added min XY distance from EEF during object reset/teleport sampling
  - New config keys under `env.object_teleport`:
    - `min_xy_dist_to_eef`
    - `min_xy_dist_resample_rounds`
- [DONE] Eval teleport suppression in DAgger mobile trainer:
  - During `eval()`, teleport is disabled and restored after successful eval return
- [DONE] Removed mem_probe logging from DAgger mobile trainer:
  - Removed probe metric collection/printing/local CSV probe logging path
