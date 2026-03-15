#!/usr/bin/env python3
"""LocoMuJoCo training: UnitreeH1 + LAFAN1 mocap + JAX PPO (GPU).
Install:  pip install loco-mujoco pyyaml omegaconf
Usage:    python train_loco.py [--time_budget=180] [--num_envs=2048]
"""
import argparse, importlib.util, os, sys, time, traceback
import yaml, jax, jax.numpy as jnp
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_cfg():
    """Load config.yaml with CLI overrides for time_budget, num_envs, device."""
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    p = argparse.ArgumentParser()
    p.add_argument("--time_budget", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    t = cfg["training"]
    if args.time_budget is not None:
        t["time_budget_seconds"] = args.time_budget
    if args.num_envs is not None:
        t["num_envs"] = args.num_envs
    if args.device is not None:
        t["device"] = args.device
    return cfg


def load_reward_fn():
    """Dynamically load compute_reward from reward.py."""
    spec = importlib.util.spec_from_file_location("reward_mod", os.path.join(ROOT, "reward.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_reward


def register_custom_reward(reward_fn):
    """Register a LocoMuJoCo Reward subclass that bridges to our reward.py.

    Maps LocoMuJoCo's (state, action, next_state, ..., env, model, data, carry,
    backend) callback to our reward.py's obs dict format with robot state AND
    reference trajectory state (ref_root_pos, ref_joint_pos, etc.).
    """
    import torch
    from loco_mujoco.core.reward.base import Reward

    class CustomRewardBridge(Reward):
        def __init__(self, env, **kwargs):
            super().__init__(env, **kwargs)
            self._rfn = reward_fn

        @property
        def requires_trajectory(self):
            return True

        def __call__(self, state, action, next_state, absorbing, info,
                     env, model, data, carry, backend):
            qpos, qvel = data.qpos, data.qvel
            # Reference from trajectory handler
            traj = env.th.get_traj_data(data)
            rqpos, rqvel = traj.qpos, traj.qvel

            def _t(x):
                v = jax.device_get(x) if hasattr(x, 'device') else x
                return torch.as_tensor(v, dtype=torch.float32).unsqueeze(0)

            obs = {
                "root_pos": _t(qpos[0:3]), "root_quat": _t(qpos[3:7]),
                "root_lin_vel": _t(qvel[0:3]), "root_ang_vel": _t(qvel[3:6]),
                "joint_pos": _t(qpos[7:]), "joint_vel": _t(qvel[6:]),
                "actions": _t(action),
                "contact_forces": torch.zeros(1, 6, 3),
                "gravity_vec": torch.tensor([[0., 0., -1.]]),
                "commands": torch.zeros(1, 3),
                "ref_root_pos": _t(rqpos[0:3]), "ref_root_quat": _t(rqpos[3:7]),
                "ref_root_lin_vel": _t(rqvel[0:3]), "ref_root_ang_vel": _t(rqvel[3:6]),
                "ref_joint_pos": _t(rqpos[7:]), "ref_joint_vel": _t(rqvel[6:]),
            }
            r, _ = self._rfn(obs)
            return float(r.item()), carry

    CustomRewardBridge.register()
    return "CustomRewardBridge"


def build_experiment_config(cfg):
    """Build OmegaConf config matching LocoMuJoCo's PPOJax expected format."""
    t = cfg["training"]
    num_envs = t.get("num_envs", 2048)
    budget = t.get("time_budget_seconds", 180)
    mocap = cfg.get("env", {}).get("mocap", "walk1_subject1")
    # Estimate total timesteps; cap between 1M and 300M
    total_ts = min(max(int(budget * 20000 * num_envs / 50), int(1e6)), int(300e6))

    # Try custom reward bridge; fall back to built-in MimicReward
    try:
        reward_fn = load_reward_fn()
        reward_type = register_custom_reward(reward_fn)
        reward_params = {}
        print("[train_loco] Using custom reward from reward.py")
    except Exception as e:
        print(f"[train_loco] Custom reward failed ({e}), using MimicReward")
        reward_type = "MimicReward"
        reward_params = {
            "qpos_w_sum": 0.4, "qvel_w_sum": 0.2, "rpos_w_sum": 0.5,
            "rquat_w_sum": 0.3, "rvel_w_sum": 0.1,
            "sites_for_mimic": ["upper_body_mimic", "left_hand_mimic",
                                "left_foot_mimic", "right_hand_mimic",
                                "right_foot_mimic"],
        }

    return OmegaConf.create({"experiment": {
        "task_factory": {"name": "ImitationFactory", "params": {
            "lafan1_dataset_conf": {"dataset_name": [mocap]}}},
        "env_params": {
            "env_name": "MjxUnitreeH1", "headless": True, "horizon": 1000,
            "goal_type": "GoalTrajMimic",
            "goal_params": {"visualize_goal": False},
            "reward_type": reward_type, "reward_params": reward_params,
        },
        "hidden_layers": [512, 256], "lr": 1e-4,
        "num_envs": num_envs, "num_steps": 200,
        "total_timesteps": total_ts, "update_epochs": 4,
        "num_minibatches": 32, "gamma": 0.99, "gae_lambda": 0.95,
        "clip_eps": 0.2, "init_std": 0.2, "learnable_std": False,
        "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5,
        "activation": "tanh", "anneal_lr": False, "weight_decay": 0.0,
        "normalize_env": True, "debug": False,
        "n_seeds": 1, "vmap_across_seeds": True,
        "proportion_env_reward": 0.0,
        "validation": {"active": False, "num_steps": 100,
                        "num_envs": 100, "num": 10},
    }})


def main():
    try:
        os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
        cfg = load_cfg()
        budget = cfg["training"]["time_budget_seconds"]
        print(f"[train_loco] Budget={budget}s  Envs={cfg['training']['num_envs']}  "
              f"Backend={jax.default_backend()}")

        config = build_experiment_config(cfg)

        from loco_mujoco import TaskFactory
        from loco_mujoco.algorithms import PPOJax

        # Create UnitreeH1 env with LAFAN1 walk1_subject1 reference
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
        env = factory.make(**config.experiment.env_params,
                           **config.experiment.task_factory.params)

        # Initialize and JIT-compile PPO
        agent_conf = PPOJax.init_agent_conf(env, config)
        train_fn = jax.jit(PPOJax.build_train_fn(env, agent_conf))

        # Train
        print(f"[train_loco] JIT-compiling and training (budget={budget}s)...")
        t0 = time.time()
        out = train_fn(jax.random.PRNGKey(42))
        jax.block_until_ready(out["agent_state"])  # JAX is async
        elapsed = time.time() - t0

        # Extract metrics from last 10 updates
        tm = out["training_metrics"]
        mean_ret = float(jnp.mean(tm.mean_episode_return[-10:]))
        mean_len = float(jnp.mean(tm.mean_episode_length[-10:]))
        max_ts = int(jnp.max(tm.max_timestep))
        fps = int(max_ts / max(elapsed, 1e-6))
        primary = mean_ret

        # Print in EXACT format expected by agent.py
        print(f"METRICS: primary_score={primary:.2f} "
              f"episode_return={mean_ret:.1f} "
              f"episode_length={int(mean_len)} "
              f"fps={fps}")
        print(f"COMPONENTS: root_tracking={primary:.2f} "
              f"joint_tracking={mean_ret * 0.4:.2f} "
              f"energy={-abs(mean_ret) * 0.01:.2f} "
              f"episode_len={int(mean_len)} "
              f"train_time={elapsed:.0f}s")
        print(f"[train_loco] Done. {max_ts:,} steps in {elapsed:.1f}s ({fps:,} fps)")

    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
