"""prepare.py -- Isaac Gym Ant wrapper, evaluation harness, and obs docs for AutoRobot."""

import importlib.util
import sys
import time
from typing import Callable, Optional

import torch
import yaml

# -- Graceful handling when Isaac Gym is not installed -----------------------
try:
    import isaacgym  # noqa: F401 -- must precede torch for GPU init
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    from isaacgymenvs.tasks import isaacgym_task_map
    from hydra import compose, initialize_config_dir
    ISAAC_GYM_AVAILABLE = True
except ImportError:
    ISAAC_GYM_AVAILABLE = False
    print(
        "\n[AutoRobot] Isaac Gym not installed. Steps:\n"
        "  1. Download Preview 4: https://developer.nvidia.com/isaac-gym\n"
        "  2. conda create -n autorobot python=3.8 && conda activate autorobot\n"
        "  3. cd isaacgym/python && pip install -e .\n"
        "  4. pip install -e <eureka>/isaacgymenvs && pip install -e <eureka>/rl_games\n"
    )

# -- Observation field registry (matches program.md) ------------------------
OBS_FIELDS = [
    ("root_pos",       "(N, 3)",    "Root body position (x, y, z) in world frame"),
    ("root_quat",      "(N, 4)",    "Root body orientation quaternion (x, y, z, w)"),
    ("root_lin_vel",   "(N, 3)",    "Root body linear velocity (x, y, z)"),
    ("root_ang_vel",   "(N, 3)",    "Root body angular velocity"),
    ("joint_pos",      "(N, 8)",    "Joint positions in radians (8 actuated DOFs)"),
    ("joint_vel",      "(N, 8)",    "Joint velocities in rad/s"),
    ("contact_forces", "(N, 4, 3)", "Contact forces on 4 feet (each xyz)"),
    ("commands",       "(N, 3)",    "Target velocity commands (x_vel, y_vel, yaw_rate)"),
    ("actions",        "(N, 8)",    "Last applied joint actions"),
    ("gravity_vec",    "(N, 3)",    "Gravity vector in body frame"),
]

def get_observation_docs() -> str:
    """Return a formatted string describing all observation fields."""
    lines = [
        "Available observation keys (all torch.Tensor, batch dim = num_envs):\n",
        f"{'Key':<20} {'Shape':<12} Description",
        "-" * 72,
    ]
    for key, shape, desc in OBS_FIELDS:
        lines.append(f"{key:<20} {shape:<12} {desc}")
    return "\n".join(lines)

# -- Environment wrapper ----------------------------------------------------
class AntEnvWrapper:
    """Wraps isaacgymenvs Ant and exposes a structured obs dict."""

    def __init__(self, num_envs: int = 4096, device: str = "cuda:0",
                 headless: bool = True, cfg_dir: Optional[str] = None):
        if not ISAAC_GYM_AVAILABLE:
            raise RuntimeError("Isaac Gym is not installed. See message above.")
        self.num_envs, self.device, self.headless = num_envs, device, headless
        self.env = self._create_env(num_envs, device, headless, cfg_dir)
        self.max_episode_length = int(self.env.max_episode_length)
        self._last_actions = torch.zeros(num_envs, self.env.num_actions, device=device)

    def _create_env(self, num_envs, device, headless, cfg_dir):
        import os
        if cfg_dir is None:
            import isaacgymenvs as ige
            cfg_dir = os.path.join(os.path.dirname(ige.__file__), "cfg")
        with initialize_config_dir(config_dir=cfg_dir, version_base="1.1"):
            cfg = compose(config_name="config", overrides=[
                "task=Ant", f"num_envs={num_envs}", f"sim_device={device}",
                f"rl_device={device}", f"headless={headless}", "graphics_device_id=0",
            ])
        env_cls = isaacgym_task_map.get("Ant")
        if env_cls is None:
            raise RuntimeError("Ant task not found in isaacgym_task_map.")
        return env_cls(
            cfg=omegaconf_to_dict(cfg.task), rl_device=device, sim_device=device,
            graphics_device_id=0, headless=headless,
            virtual_screen_capture=False, force_render=False,
        )

    def _extract_obs(self) -> dict[str, torch.Tensor]:
        env = self.env
        rs = env.root_states  # (N, 13): pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        return {
            "root_pos":       rs[:, 0:3],
            "root_quat":      rs[:, 3:7],
            "root_lin_vel":   rs[:, 7:10],
            "root_ang_vel":   rs[:, 10:13],
            "joint_pos":      env.dof_pos,
            "joint_vel":      env.dof_vel,
            "contact_forces": env.contact_forces[:, env.feet_indices, :],
            "commands":       self._get_commands(),
            "actions":        self._last_actions,
            "gravity_vec":    self._get_gravity_vec(),
        }

    def _get_commands(self) -> torch.Tensor:
        if hasattr(self.env, "commands"):
            return self.env.commands
        cmds = torch.zeros(self.num_envs, 3, device=self.device)
        cmds[:, 0] = 1.0
        return cmds

    def _get_gravity_vec(self) -> torch.Tensor:
        if hasattr(self.env, "projected_gravity"):
            return self.env.projected_gravity
        from isaacgym.torch_utils import quat_rotate_inverse
        quat = self.env.root_states[:, 3:7]
        grav = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        return quat_rotate_inverse(quat, grav)

    def step(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Step the simulation and return the structured obs dict."""
        self._last_actions = actions.clone()
        self.env.step(actions)
        return self._extract_obs()

    def reset(self) -> dict[str, torch.Tensor]:
        """Reset all environments and return initial obs dict."""
        self.env.reset()
        self._last_actions.zero_()
        return self._extract_obs()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

# -- Evaluation harness ------------------------------------------------------
def evaluate(reward_fn: Callable, env: AntEnvWrapper, num_episodes: int = 50) -> dict:
    """Run rollouts, score with *reward_fn*, return aggregated metrics dict."""
    total_returns, total_lengths, fwd_vels = [], [], []
    comp_acc: dict[str, list[float]] = {}

    obs = env.reset()
    ep_ret = torch.zeros(env.num_envs, device=env.device)
    ep_len = torch.zeros(env.num_envs, device=env.device)
    done_count, step_count, t0 = 0, 0, time.time()

    while done_count < num_episodes:
        actions = torch.randn(env.num_envs, env.env.num_actions, device=env.device).clamp(-1, 1)
        obs = env.step(actions)
        step_count += env.num_envs

        reward, components = reward_fn(obs)
        ep_ret += reward
        ep_len += 1
        fwd_vels.append(obs["root_lin_vel"][:, 0].mean().item())

        for k, v in components.items():
            comp_acc.setdefault(k, []).append(v.mean().item())

        dones = ep_len >= env.max_episode_length
        if dones.any():
            idx = dones.nonzero(as_tuple=False).squeeze(-1)
            for i in idx:
                total_returns.append(ep_ret[i].item())
                total_lengths.append(ep_len[i].item())
                done_count += 1
                if done_count >= num_episodes:
                    break
            ep_ret[idx] = 0.0
            ep_len[idx] = 0.0

    elapsed = time.time() - t0
    mean_comp = {k: sum(v) / max(len(v), 1) for k, v in comp_acc.items()}
    _avg = lambda lst: sum(lst) / max(len(lst), 1)
    return {
        "primary_score": _avg(fwd_vels),
        "episode_return": _avg(total_returns),
        "episode_length": _avg(total_lengths),
        "reward_components": mean_comp,
        "training_fps": step_count / max(elapsed, 1e-6),
    }

# -- Dynamic reward loading --------------------------------------------------
def load_reward_function(path: str = "reward.py") -> Callable:
    """Dynamically import compute_reward from a Python file."""
    spec = importlib.util.spec_from_file_location("reward_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reward function from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "compute_reward"):
        raise AttributeError(f"{path} must define compute_reward(obs) -> (tensor, dict)")
    return mod.compute_reward

# -- Config helper -----------------------------------------------------------
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -- CLI quick-test ----------------------------------------------------------
if __name__ == "__main__":
    print(get_observation_docs())
    if not ISAAC_GYM_AVAILABLE:
        print("\nSkipping env test -- Isaac Gym not installed.")
        sys.exit(0)
    cfg = load_config()
    env = AntEnvWrapper(
        num_envs=cfg["training"]["num_envs"], device=cfg["training"]["device"],
        headless=cfg["training"].get("headless", True),
    )
    metrics = evaluate(load_reward_function(), env, num_episodes=10)
    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for ck, cv in v.items():
                print(f"    {ck}: {cv:.4f}")
        else:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    env.close()
