from __future__ import annotations
import numpy as np


def compute_reward(obs: dict) -> tuple[float, dict[str, float]]:
    """SONIC-style motion tracking reward for Unitree H1 with LAFAN1 reference.

    Computes exponential tracking kernels: exp(-||error||^2 / temperature)
    between current robot state and LAFAN1 reference motion provided by
    LocoMuJoCo in the observation dict.
    """
    # --- Tracking rewards (SONIC exponential kernels) ---

    # 1. Root position tracking (weight 1.0, temp 1.0)
    root_pos_err = np.sum((obs["root_pos"] - obs["ref_root_pos"]) ** 2)
    root_pos_tracking = 1.0 * np.exp(-root_pos_err / 1.0)

    # 2. Root orientation tracking (weight 0.5, temp 0.1)
    root_quat_err = np.sum((obs["root_quat"] - obs["ref_root_quat"]) ** 2)
    root_orient_tracking = 0.5 * np.exp(-root_quat_err / 0.1)

    # 3. Joint position tracking (weight 1.0, temp 2.0)
    joint_pos_err = np.sum((obs["joint_pos"] - obs["ref_joint_pos"]) ** 2)
    joint_pos_tracking = 1.0 * np.exp(-joint_pos_err / 2.0)

    # 4. Joint velocity tracking (weight 0.5, temp 5.0)
    joint_vel_err = np.sum((obs["joint_vel"] - obs["ref_joint_vel"]) ** 2)
    joint_vel_tracking = 0.5 * np.exp(-joint_vel_err / 5.0)

    # --- Regularization penalties ---

    # 5. Energy penalty (weight -0.1) — penalize large joint velocities
    jv_sq = np.sum(obs["joint_vel"] ** 2)
    energy_penalty = -0.1 * jv_sq / (jv_sq + 100.0)

    # 6. Action rate penalty (weight -0.05) — penalize large actions
    act_sq = np.sum(obs["actions"] ** 2)
    action_rate_penalty = -0.05 * act_sq / (act_sq + 50.0)

    total_reward = (root_pos_tracking + root_orient_tracking
                    + joint_pos_tracking + joint_vel_tracking
                    + energy_penalty + action_rate_penalty)

    components = {
        "root_pos_tracking": root_pos_tracking,
        "root_orient_tracking": root_orient_tracking,
        "joint_pos_tracking": joint_pos_tracking,
        "joint_vel_tracking": joint_vel_tracking,
        "energy_penalty": energy_penalty,
        "action_rate_penalty": action_rate_penalty,
    }
    return float(total_reward), components


def make_reward_callback(obs_keys: list[str], ref_keys: list[str],
                         obs_splits: dict, ref_splits: dict):
    """Create a LocoMuJoCo-compatible reward_callback that wraps compute_reward.

    LocoMuJoCo signature: reward_callback(state, action, next_state) -> float
    This adapter unpacks the flat obs vector into our dict format, calls
    compute_reward, and returns the scalar reward.

    Args:
        obs_keys: ordered keys for robot state slices in the obs vector
        ref_keys: ordered keys for reference trajectory slices
        obs_splits: {key: (start, end)} index ranges in the state vector
        ref_splits: {key: (start, end)} index ranges for ref data in state
    """
    def reward_callback(state, action, next_state):
        obs = {}
        for key, (start, end) in obs_splits.items():
            obs[key] = next_state[start:end]
        for key, (start, end) in ref_splits.items():
            obs[key] = next_state[start:end]
        obs["actions"] = action
        reward, _ = compute_reward(obs)
        return reward

    return reward_callback
