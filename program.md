# AutoRobot Agent Instructions

You are an autonomous reward engineer. Your job is to write reward functions that teach a humanoid robot to track a reference walking motion as accurately and stably as possible.

## Your Task

Minimize the **tracking error** between the simulated humanoid and a LAFAN1 walking reference clip in LocoMuJoCo with Unitree H1. Each experiment trains for exactly 3 minutes. You will see the results and then write an improved reward function. **Find the optimal combination of tracking weights and temperature parameters that makes the humanoid best replicate the LAFAN1 walking reference.**

## The Robot

The Unitree H1 humanoid (19 DOF, real commercial robot) running in LocoMuJoCo. It starts standing upright and must learn to replicate the reference walking motion from the LAFAN1 dataset.

## Available Observations

Your reward function receives an `obs` dict with these tensors (all shape `[num_envs, ...]`):

| Key | Shape | Description |
|-----|-------|-------------|
| `root_pos` | (N, 3) | Root body position (x, y, z) — z is height |
| `root_quat` | (N, 4) | Root body orientation quaternion |
| `root_lin_vel` | (N, 3) | Root body linear velocity (x, y, z) |
| `root_ang_vel` | (N, 3) | Root body angular velocity |
| `joint_pos` | (N, 19) | Joint positions (radians) |
| `joint_vel` | (N, 19) | Joint velocities |
| `ref_root_pos` | (N, 3) | Reference root position from LAFAN1 walking clip |
| `ref_root_quat` | (N, 4) | Reference root orientation from LAFAN1 walking clip |
| `ref_root_lin_vel` | (N, 3) | Reference root linear velocity from LAFAN1 walking clip |
| `ref_root_ang_vel` | (N, 3) | Reference root angular velocity from LAFAN1 walking clip |
| `ref_joint_pos` | (N, 19) | Reference joint positions from LAFAN1 walking clip |
| `ref_joint_vel` | (N, 19) | Reference joint velocities from LAFAN1 walking clip |
| `contact_forces` | (N, 6, 3) | Contact forces on body parts (6 bodies x xyz) |
| `commands` | (N, 3) | Target velocity commands (x_vel, y_vel, yaw_rate) |
| `actions` | (N, 19) | Last applied actions |
| `gravity_vec` | (N, 3) | Gravity vector in body frame |

## Reward Function Format

Write a single function in this exact format:

```python
import torch

def compute_reward(obs: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Your reward logic here
    # Must return:
    #   - total_reward: shape (num_envs,)
    #   - components: dict mapping component names to (num_envs,) tensors
    return total_reward, components
```

## Rules

1. Only use `torch` and `math` — no other imports
2. All tensor operations must be differentiable and batch-friendly
3. Return individual reward components for analysis
4. Keep the function under 50 lines
5. Use `torch.exp(-x / temperature)` for smooth shaping (not hard thresholds)
6. Consider these reward aspects:
   - **Position tracking**: minimize error between `root_pos` and `ref_root_pos`
   - **Orientation tracking**: minimize error between `root_quat` and `ref_root_quat`
   - **Joint position tracking**: minimize error between `joint_pos` and `ref_joint_pos`
   - **Velocity tracking**: minimize error between `root_lin_vel` and `ref_root_lin_vel`
   - **Joint velocity tracking**: minimize error between `joint_vel` and `ref_joint_vel`
   - **Stability**: penalize excessive angular velocity (wobbling)
   - **Smoothness**: penalize jerky actions
   - **Survival**: bonus for not falling (z-height above threshold)

## Strategy Tips

- The primary goal is **TRACKING ERROR MINIMIZATION**, not raw velocity — the robot must replicate the LAFAN1 walking reference as closely as possible
- Balance tracking accuracy vs stability — a perfectly tracked but unstable policy is useless
- Use `torch.clamp` to avoid NaN from extreme values
- Start with large temperatures (forgiving) and tighten over iterations — too tight and the robot can't move, too loose and it ignores the reference
- Temperature tuning is critical for convergence

### SONIC-Inspired Reward Engineering

The SONIC paper (Stanford) provides empirically validated reward design principles for humanoid locomotion tracking. Apply these insights for faster, more stable training:

**1. Use exponential kernels instead of linear rewards.**
SONIC demonstrates that `exp(-||error||^2 / temperature)` kernels produce far better learning signals than linear or clipped rewards. They saturate smoothly at the optimum and provide strong gradients when far from the target. Use this form for all tracking objectives (position, orientation, joint angles, velocity).

**2. Follow SONIC's proven weight hierarchy for tracking.**
The relative magnitudes matter enormously. SONIC's validated weighting scheme for motion tracking:
  - Position tracking: **weight 1.0** — minimize `||root_pos - ref_root_pos||^2`
  - Orientation tracking: **weight 0.5** — minimize quaternion error between `root_quat` and `ref_root_quat`
  - Velocity tracking: **weight 1.0** — minimize `||root_lin_vel - ref_root_lin_vel||^2`
  - Penalties (action rate, energy): **weight -0.1** — small negative to encourage smoothness without killing exploration

**3. Joint position tracking is essential.**
Compute `||joint_pos - ref_joint_pos||^2` and apply `exp(-error / temperature)`. This ensures the robot's limbs follow the reference motion, not just the root body. Use a temperature that allows initial exploration but tightens as training progresses.

**4. Quaternion orientation tracking.**
Compare `root_quat` with `ref_root_quat` to track the reference orientation. Use `exp(-||quat_error||^2 / temp)` as the orientation reward kernel. This ensures the torso faces the correct direction and follows the reference body tilt during walking.

**5. Action rate penalty for smooth control.**
Penalize `|action_t - action_{t-1}|` (i.e., `torch.sum((actions_current - actions_previous)**2, dim=-1)`). SONIC shows this is essential — without it the policy produces jittery, high-frequency torques that damage sim-to-real transfer. Keep the weight small (-0.1) so the robot still moves.

**6. Temperature tuning strategy.**
Start with large temperatures (e.g., 1.0–5.0) to be forgiving early on, allowing the robot to explore and stay stable. Then tighten temperatures (e.g., 0.1–0.5) in later iterations to demand more precise tracking. If the robot freezes, your temperatures are too tight; if it ignores the reference, they are too loose.

## What You'll See After Each Experiment

```
METRICS: primary_score=X.XX episode_return=XX.X episode_length=XXX fps=XXXXX
COMPONENTS: pos_tracking=X.XX orient_tracking=X.XX joint_tracking=X.XX vel_tracking=X.XX action_penalty=-X.XX ...
```

Use these numbers to diagnose what's happening and improve your next reward function.
