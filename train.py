#!/usr/bin/env python3
"""Fixed-budget PPO training for AutoRobot.

Loads Ant env via isaacgymenvs, injects custom reward from reward.py,
trains with rl_games PPO under a wall-clock time limit, evaluates,
and prints parseable metrics to stdout.
Usage:  python train.py [--time_budget=180] [--num_envs=4096]
"""
import argparse, importlib.util, os, sys, time, traceback
import numpy as np, torch, yaml

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Config & CLI ──────────────────────────────────────────────────────────
def load_cfg():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    p = argparse.ArgumentParser()
    p.add_argument("--time_budget", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--headless", default=None,
                       type=lambda x: x.lower() in ("true","1","yes"))
    p.add_argument("--max_iterations", type=int, default=None)
    args = p.parse_args()
    t = cfg["training"]
    for k, ck in [("time_budget","time_budget_seconds"),("num_envs","num_envs"),
                   ("device","device"),("headless","headless"),("max_iterations","max_iterations")]:
        v = getattr(args, k)
        if v is not None:
            t[ck] = v
    return cfg

# ── Reward loader ─────────────────────────────────────────────────────────
def load_reward_fn():
    spec = importlib.util.spec_from_file_location("reward_mod", os.path.join(ROOT, "reward.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_reward

# ── Observation dict builder ──────────────────────────────────────────────
def build_obs_dict(env):
    rb = env.root_states  # (N,13): pos(3) quat(4) lin_vel(3) ang_vel(3)
    dev = env.device
    n = env.num_envs
    if hasattr(env, "feet_indices"):
        cf = env.contact_forces[:, env.feet_indices, :]
    elif hasattr(env, "vec_sensor_tensor"):
        cf = env.vec_sensor_tensor.view(n, -1, 3)
    else:
        cf = torch.zeros(n, 4, 3, device=dev)
    gv = env.gravity_vec if hasattr(env, "gravity_vec") else torch.tensor([[0.,0.,-1.]], device=dev).expand(n, -1)
    return {
        "root_pos": rb[:, 0:3], "root_quat": rb[:, 3:7],
        "root_lin_vel": rb[:, 7:10], "root_ang_vel": rb[:, 10:13],
        "joint_pos": env.dof_pos, "joint_vel": env.dof_vel,
        "contact_forces": cf, "actions": env.actions,
        "gravity_vec": gv, "commands": torch.zeros(n, 3, device=dev),
    }

# ── Environment creation ─────────────────────────────────────────────────
def create_env(cfg):
    from hydra import compose, initialize_config_dir
    import isaacgymenvs
    t = cfg["training"]
    cfg_dir = os.path.join(os.path.dirname(isaacgymenvs.__file__), "cfg")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        hcfg = compose(config_name="config", overrides=[
            "task=Ant", f"num_envs={t['num_envs']}", f"headless={t['headless']}",
            "pipeline=gpu", f"sim_device={t['device']}", f"rl_device={t['device']}",
            f"train.params.config.max_epochs={t['max_iterations']}",
        ])
    env = isaacgymenvs.make(
        seed=42, task=hcfg.task_name, num_envs=t["num_envs"],
        sim_device=t["device"], rl_device=t["device"],
        graphics_device_id=-1 if t["headless"] else 0,
        headless=t["headless"], cfg=hcfg,
    )
    return env, hcfg

def patch_reward(env, reward_fn):
    def custom_compute(*args, **kwargs):
        rew, comps = reward_fn(build_obs_dict(env))
        env.rew_buf[:] = rew
        env._reward_components = comps
    env.compute_reward = custom_compute
    env._reward_components = {}

# ── PPO training with time budget ─────────────────────────────────────────
def train_with_budget(env, hcfg, cfg):
    from omegaconf import OmegaConf
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    t = cfg["training"]
    budget = t["time_budget_seconds"]

    vecenv.register("AutoRobotEnv", lambda cfg_name, num_actors, **kw: env)
    env_configurations.register("AutoRobotEnv", {
        "vecenv_type": "AutoRobotEnv",
        "env_creator": lambda **kw: env,
    })
    rlg = OmegaConf.to_container(hcfg.train, resolve=True)
    rlg["params"]["config"]["env_name"] = "AutoRobotEnv"
    rlg["params"]["config"]["num_actors"] = t["num_envs"]

    runner = Runner()
    runner.load(rlg)
    runner.reset()
    agent = runner.algo_factory.create(
        runner.algo_name, base_name='run', params=runner.params)

    t0 = time.time()
    orig_epoch = agent.train_epoch
    def timed_epoch():
        if time.time() - t0 >= budget:
            raise TimeoutError
        return orig_epoch()
    agent.train_epoch = timed_epoch

    try:
        agent.train()
    except (TimeoutError, KeyboardInterrupt):
        pass
    elapsed = time.time() - t0
    fps = int(agent.frame / max(elapsed, 1e-6)) if hasattr(agent, "frame") else 0
    return agent, elapsed, fps

# ── Evaluation ────────────────────────────────────────────────────────────
def evaluate(env, reward_fn, agent, num_episodes=50):
    dev = env.device
    ep_returns, ep_lengths = [], []
    all_comps = {}
    cur_ret = torch.zeros(env.num_envs, device=dev)
    cur_len = torch.zeros(env.num_envs, device=dev, dtype=torch.long)
    done_count = 0

    # Use trained policy if available, else random
    has_policy = hasattr(agent, "model") and agent.model is not None
    obs_dict = env.reset()
    rnn_states = None
    if has_policy:
        agent.model.eval()
        is_rnn = agent.model.is_rnn() if hasattr(agent.model, "is_rnn") else False
        if is_rnn:
            rnn_states = [torch.zeros_like(s).to(dev)
                          for s in agent.model.get_default_rnn_state()]

    while done_count < num_episodes:
        with torch.no_grad():
            if has_policy:
                obs_flat = env.obs_buf
                input_dict = {
                    "is_train": False,
                    "prev_actions": None,
                    "obs": obs_flat,
                    "rnn_states": rnn_states,
                }
                res = agent.model(input_dict)
                action = res["mus"]  # deterministic (mean) action
                if is_rnn:
                    rnn_states = res["rnn_states"]
            else:
                action = torch.clamp(torch.randn(env.num_envs, env.num_actions, device=dev), -1, 1)
            obs_dict, rew, dones, infos = env.step(action)

        rew_custom, comps = reward_fn(build_obs_dict(env))
        cur_ret += rew_custom
        cur_len += 1

        done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
        for i in done_idx:
            if done_count >= num_episodes:
                break
            ep_returns.append(cur_ret[i].item())
            ep_lengths.append(cur_len[i].item())
            for k, v in comps.items():
                all_comps.setdefault(k, []).append(v[i].item())
            cur_ret[i] = 0.0
            cur_len[i] = 0
            done_count += 1

    comp_means = {k: float(np.mean(v)) for k, v in all_comps.items() if v}
    return {
        "primary_score": comp_means.get("forward_vel", float(np.mean(ep_returns))),
        "episode_return": float(np.mean(ep_returns)),
        "episode_length": float(np.mean(ep_lengths)),
    }, comp_means

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    cfg = load_cfg()
    try:
        reward_fn = load_reward_fn()
    except Exception:
        print("ERROR: Failed to load reward.py", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    env, hcfg = create_env(cfg)
    patch_reward(env, reward_fn)
    agent, elapsed, fps = train_with_budget(env, hcfg, cfg)

    n_eval = cfg["env"].get("eval_episodes", 50)
    metrics, comp_means = evaluate(env, reward_fn, agent, num_episodes=n_eval)
    metrics["fps"] = fps

    print(f"METRICS: primary_score={metrics['primary_score']:.2f} "
          f"episode_return={metrics['episode_return']:.1f} "
          f"episode_length={int(metrics['episode_length'])} "
          f"fps={metrics['fps']}")
    print(f"COMPONENTS: {' '.join(f'{k}={v:.2f}' for k, v in comp_means.items())}")

    if hasattr(env, "close"):
        env.close()

if __name__ == "__main__":
    main()
