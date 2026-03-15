from __future__ import annotations
import torch


def compute_reward(obs: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SONIC-inspired reward for Humanoid locomotion.

    Uses exponential kernels: exp(-||error||^2 / temperature) with
    weights inspired by SONIC Table 1.
    """
    # 1. Forward velocity reward (weight 1.0) — primary objective
    fwd_vel = obs["root_lin_vel"][:, 0]
    vel_reward = 1.0 * (1.0 - torch.exp(-fwd_vel * fwd_vel / 4.0)) * torch.sign(fwd_vel).clamp(min=0.0)

    # 2. Upright orientation reward (weight 0.5) — quat w near 1.0
    quat_w = obs["root_quat"][:, 3]
    orient_err = (1.0 - quat_w) ** 2
    upright_reward = 0.5 * torch.exp(-orient_err / 0.1)

    # 3. Healthy height reward (weight 1.0) — z in [1.0, 2.0]
    z = obs["root_pos"][:, 2]
    z_target = 1.4
    z_err = (z - z_target) ** 2
    height_reward = 1.0 * torch.exp(-z_err / 0.25)

    # 4. Energy penalty (weight -0.1) — penalize joint velocities
    jv_sq = torch.sum(obs["joint_vel"] ** 2, dim=-1)
    energy_penalty = -0.1 * torch.exp(-1.0 / (jv_sq / 100.0 + 1e-6))

    # 5. Angular velocity penalty (weight -0.1) — penalize wobbling
    ang_vel_sq = torch.sum(obs["root_ang_vel"] ** 2, dim=-1)
    angvel_penalty = -0.1 * (1.0 - torch.exp(-ang_vel_sq / 4.0))

    # 6. Action rate penalty (weight -0.1) — penalize large actions
    act_sq = torch.sum(obs["actions"] ** 2, dim=-1)
    action_penalty = -0.1 * (1.0 - torch.exp(-act_sq / 10.0))

    total_reward = (vel_reward + upright_reward + height_reward
                    + energy_penalty + angvel_penalty + action_penalty)

    components = {
        "vel_reward": vel_reward, "upright": upright_reward,
        "height": height_reward, "energy": energy_penalty,
        "angvel": angvel_penalty, "action_rate": action_penalty,
    }
    return total_reward, components
