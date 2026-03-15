#!/usr/bin/env python3
"""LocoMuJoCo training: UnitreeH1 + LAFAN1 mocap + JAX PPO (GPU).

Uses LocoMuJoCo's built-in MimicReward (DeepMimic-style tracking).
The LLM agent optimizes reward weights in reward_weights.json.

Install:  pip install loco-mujoco pyyaml
Usage:    python train_loco.py [--time_budget=180] [--num_envs=2048]
"""
import argparse, json, os, sys, time, traceback
import yaml, jax, jax.numpy as jnp

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(ROOT, "reward_weights.json")

DEFAULT_WEIGHTS = {
    "qpos_w_exp": 10.0,
    "qvel_w_exp": 2.0,
    "rpos_w_exp": 100.0,
    "rquat_w_exp": 10.0,
    "rvel_w_exp": 0.1,
    "qpos_w_sum": 0.0,
    "qvel_w_sum": 0.0,
    "rpos_w_sum": 0.5,
    "rquat_w_sum": 0.3,
    "rvel_w_sum": 0.0,
    "action_out_of_bounds_coeff": 0.01,
    "joint_acc_coeff": 0.0,
    "joint_torque_coeff": 0.0,
    "action_rate_coeff": 0.0,
}


def load_cfg():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    p = argparse.ArgumentParser()
    p.add_argument("--time_budget", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    args = p.parse_args()
    t = cfg["training"]
    if args.time_budget is not None:
        t["time_budget_seconds"] = args.time_budget
    if args.num_envs is not None:
        t["num_envs"] = args.num_envs
    return cfg


def load_reward_weights():
    """Load reward weights from reward_weights.json, or use defaults."""
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH) as f:
            weights = json.load(f)
        # Merge with defaults for any missing keys
        merged = {**DEFAULT_WEIGHTS, **weights}
        return merged
    return DEFAULT_WEIGHTS.copy()


def main():
    try:
        os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
        cfg = load_cfg()
        budget = cfg["training"]["time_budget_seconds"]
        num_envs = cfg["training"]["num_envs"]
        mocap = cfg.get("env", {}).get("mocap", "walk1_subject1")

        print(f"[train_loco] Budget={budget}s  Envs={num_envs}  "
              f"Backend={jax.default_backend()}  Mocap={mocap}")

        # Load reward weights
        weights = load_reward_weights()
        print(f"[train_loco] Reward weights: {json.dumps(weights, indent=2)}")

        from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf
        from loco_mujoco.algorithms import PPOJax

        # Create env with MimicReward and our weights
        env = ImitationFactory.make(
            "MjxUnitreeH1",
            lafan1_dataset_conf=LAFAN1DatasetConf([mocap]),
            headless=True,
            reward_type="MimicReward",
            reward_params=weights,
        )

        # Build PPO config
        ppo_conf = {
            "hidden_layers": [512, 256],
            "lr": 1e-4,
            "num_envs": num_envs,
            "num_steps": 200,
            "total_timesteps": int(budget * num_envs * 100),
            "update_epochs": 4,
            "num_minibatches": 32,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "init_std": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "activation": "tanh",
            "anneal_lr": False,
            "normalize_env": True,
            "debug": False,
            "n_seeds": 1,
            "vmap_across_seeds": True,
            "learnable_std": False,
            "weight_decay": 0.0,
            "proportion_env_reward": 0.0,
            "validation": {"active": False, "num_steps": 100,
                           "num_envs": 100, "num": 10},
        }

        from omegaconf import OmegaConf
        agent_conf = OmegaConf.create({"experiment": ppo_conf})
        agent_conf = PPOJax.init_agent_conf(env, agent_conf)
        train_fn = jax.jit(PPOJax.build_train_fn(env, agent_conf))

        # Train
        print(f"[train_loco] JIT-compiling and training...")
        t0 = time.time()
        out = train_fn(jax.random.PRNGKey(42))
        jax.block_until_ready(out["agent_state"])
        elapsed = time.time() - t0

        # Extract metrics
        tm = out["training_metrics"]
        returns = tm.mean_episode_return
        lengths = tm.mean_episode_length
        mean_ret = float(jnp.mean(returns[-10:]))
        mean_len = float(jnp.mean(lengths[-10:]))
        best_ret = float(jnp.max(returns))
        max_ts = int(jnp.max(tm.max_timestep))
        fps = int(max_ts / max(elapsed, 1e-6))

        # Print in EXACT format expected by agent.py
        print(f"METRICS: primary_score={mean_ret:.4f} "
              f"episode_return={mean_ret:.1f} "
              f"episode_length={int(mean_len)} "
              f"fps={fps}")

        # Print weight-derived components
        comp_parts = []
        for k, v in weights.items():
            comp_parts.append(f"{k}={v:.4f}")
        comp_parts.append(f"best_return={best_ret:.4f}")
        comp_parts.append(f"train_time={elapsed:.0f}s")
        print(f"COMPONENTS: {' '.join(comp_parts)}")

        print(f"[train_loco] Done. {max_ts:,} steps in {elapsed:.1f}s ({fps:,} fps)")

    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
