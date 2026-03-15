#!/usr/bin/env python3
"""Fallback PPO training using MuJoCo Humanoid-v5 + SB3. Drop-in for train.py.
Install:  pip install "gymnasium[mujoco]" stable-baselines3
Usage:    python train_mujoco.py [--time_budget=180] [--num_envs=16]
"""
import argparse, importlib.util, os, sys, time
import numpy as np, torch, yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Config ────────────────────────────────────────────────────────────────
def load_cfg():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    p = argparse.ArgumentParser()
    p.add_argument("--time_budget", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--device", default=None)
    args, t = p.parse_args(), cfg["training"]
    for attr, key in [("time_budget","time_budget_seconds"),("num_envs","num_envs"),("device","device")]:
        if getattr(args, attr) is not None:
            t[key] = getattr(args, attr)
    # MuJoCo can't run 4096 envs — default to 8 for CPU
    if t["num_envs"] > 64:
        t["num_envs"] = 8
    return cfg

# ── Reward loader ─────────────────────────────────────────────────────────
def load_reward_fn():
    spec = importlib.util.spec_from_file_location("reward_mod", os.path.join(ROOT, "reward.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_reward

# ── Map MuJoCo Humanoid-v5 obs to our obs dict format ────────────────────
def mujoco_obs_to_dict(obs: np.ndarray) -> dict:
    """Convert Humanoid-v5 flat obs to program.md obs dict (N, ...).
    Humanoid-v5 obs (348-dim when include_cinert etc. enabled, 45-dim minimal):
      qpos[2:] = z(1) + quat(4) + joints(21)
      qvel[:] = lin_vel(3) + ang_vel(3) + joint_vel(21)
      cinert, cvel, qfrc_actuator, cfrc_ext (optional)
    We use the default obs which is 67-dim:
      qpos (24) + qvel (23) + cinert (0) + cvel (0) + qfrc (0) + cfrc (0)
    Actually Humanoid-v5 default obs = 348-dim. We extract what we need.
    """
    if obs.ndim == 1:
        obs = obs[np.newaxis, :]
    n = obs.shape[0]
    dim = obs.shape[1]
    t = lambda x: torch.as_tensor(x, dtype=torch.float32)

    # Humanoid qpos layout: [0]=z, [1:5]=quat, [5:26]=joint_pos (21 joints)
    # Humanoid qvel layout: [0:3]=lin_vel, [3:6]=ang_vel, [6:27]=joint_vel (21 joints)
    # In obs vector: first 24 values are qpos[2:] (excluding x,y), next 23 are qvel
    z_pos = obs[:, 0:1] if dim >= 1 else np.zeros((n, 1))
    quat = obs[:, 1:5] if dim >= 5 else np.ones((n, 4)) * 0.25
    joint_pos = obs[:, 5:26] if dim >= 26 else np.zeros((n, 21))
    lin_vel = obs[:, 24:27] if dim >= 27 else np.zeros((n, 3))
    ang_vel = obs[:, 27:30] if dim >= 30 else np.zeros((n, 3))
    joint_vel = obs[:, 30:51] if dim >= 51 else np.zeros((n, 21))

    return {
        "root_pos": t(np.concatenate([np.zeros((n, 2)), z_pos], axis=-1)),
        "root_quat": t(quat),
        "root_lin_vel": t(lin_vel),
        "root_ang_vel": t(ang_vel),
        "joint_pos": t(joint_pos),
        "joint_vel": t(joint_vel),
        "contact_forces": t(np.zeros((n, 6, 3))),  # 6 body contacts for humanoid
        "actions": t(np.zeros((n, 17))),  # 17 actuators
        "gravity_vec": t(np.tile([0., 0., -1.], (n, 1))),
        "commands": t(np.zeros((n, 3))),
    }

# ── Time-budget callback ─────────────────────────────────────────────────
class TimeBudgetCallback(BaseCallback):
    def __init__(self, budget_seconds: float, verbose=0):
        super().__init__(verbose)
        self.budget = budget_seconds
        self.t0 = None
        self.total_steps = 0

    def _on_training_start(self):
        self.t0 = time.time()

    def _on_step(self) -> bool:
        self.total_steps += 1
        if time.time() - self.t0 >= self.budget:
            return False
        return True

# ── Evaluation ────────────────────────────────────────────────────────────
def evaluate(model, reward_fn, num_episodes=20):
    env = gym.make("Humanoid-v5")
    ep_returns, ep_lengths = [], []
    all_comps: dict[str, list[float]] = {}

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0
        ep_comps: dict[str, list[float]] = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_len += 1

            obs_dict = mujoco_obs_to_dict(obs)
            obs_dict["actions"] = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
            rew_t, comps = reward_fn(obs_dict)
            ep_ret += rew_t.item()
            for k, v in comps.items():
                ep_comps.setdefault(k, []).append(v.item())

        ep_returns.append(ep_ret)
        ep_lengths.append(ep_len)
        for k, v in ep_comps.items():
            all_comps.setdefault(k, []).append(float(np.mean(v)))

    env.close()
    comp_means = {k: float(np.mean(v)) for k, v in all_comps.items() if v}
    return {
        "primary_score": comp_means.get("forward_vel", float(np.mean(ep_returns))),
        "episode_return": float(np.mean(ep_returns)),
        "episode_length": float(np.mean(ep_lengths)),
    }, comp_means

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    cfg = load_cfg()
    t = cfg["training"]
    budget = t["time_budget_seconds"]
    num_envs = t["num_envs"]

    try:
        reward_fn = load_reward_fn()
    except Exception as e:
        print(f"ERROR: Failed to load reward.py: {e}", file=sys.stderr)
        sys.exit(1)

    make_env = lambda seed: (lambda: (lambda e: (e.reset(seed=seed), e)[-1])(gym.make("Humanoid-v5")))
    env = (SubprocVecEnv if num_envs > 1 else DummyVecEnv)(
        [make_env(42 + i) for i in range(num_envs)])
    device_str = t.get("device", "cpu")
    sb3_device = "cuda" if ("cuda" in device_str and torch.cuda.is_available()) else "cpu"

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=0,
        device=sb3_device,
    )

    # Train under time budget
    callback = TimeBudgetCallback(budget)
    t0 = time.time()
    model.learn(total_timesteps=10_000_000, callback=callback)
    elapsed = time.time() - t0
    total_steps = callback.total_steps * num_envs
    fps = int(total_steps / max(elapsed, 1e-6))

    # Evaluate
    n_eval = cfg["env"].get("eval_episodes", 20)
    metrics, comp_means = evaluate(model, reward_fn, num_episodes=n_eval)
    metrics["fps"] = fps

    # Print in the EXACT same format as train.py
    print(f"METRICS: primary_score={metrics['primary_score']:.2f} "
          f"episode_return={metrics['episode_return']:.1f} "
          f"episode_length={int(metrics['episode_length'])} "
          f"fps={metrics['fps']}")
    print(f"COMPONENTS: {' '.join(f'{k}={v:.2f}' for k, v in comp_means.items())}")

    env.close()

if __name__ == "__main__":
    main()
