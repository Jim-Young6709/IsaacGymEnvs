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

Example:

* Real file:
  `/home/rayliu/grogu/IsaacGymEnvs/isaacgymenvs/tasks/franka_leap.py`

* Dummy file:
  `/home/rayliu/grogu/IsaacGymEnvs/isaacgymenvs/tasks/franka_leap_dummy.py`

All changes must be clearly marked with comments:

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