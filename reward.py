import torch


def compute_reward(obs: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Baseline reward function for Ant locomotion.

    Simple initial reward: forward velocity + survival bonus - energy penalty.
    Intended as a starting point for iterative improvement.
    """
    # Primary objective: reward forward velocity (x-axis)
    forward_vel = obs["root_lin_vel"][:, 0]

    # Small constant survival bonus to encourage staying alive
    survival_bonus = torch.ones_like(forward_vel) * 0.5

    # Mild energy penalty based on joint velocities
    joint_vel = obs["joint_vel"]
    energy_penalty = -0.005 * torch.sum(joint_vel ** 2, dim=-1)

    # Total reward
    total_reward = forward_vel + survival_bonus + energy_penalty

    components = {
        "forward_vel": forward_vel,
        "survival": survival_bonus,
        "energy": energy_penalty,
    }

    return total_reward, components
