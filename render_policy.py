#!/usr/bin/env python3
"""Render a trained LocoMuJoCo policy in MuJoCo.

Usage:
    python render_policy.py                          # live render from checkpoints/latest/
    python render_policy.py --path runs/my_run/      # from a specific checkpoint
    python render_policy.py --save_video results/policy.mp4
    python render_policy.py --show_reference          # also render the mocap reference
    python render_policy.py --episodes 5
"""
import argparse, os, sys, yaml

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Render a trained LocoMuJoCo policy")
    p.add_argument("--path", type=str, default=os.path.join(ROOT, "checkpoints", "latest"),
                   help="Path to saved agent checkpoint directory")
    p.add_argument("--save_video", type=str, default=None,
                   help="Save rendering to video file (e.g. results/policy.mp4)")
    p.add_argument("--show_reference", action="store_true",
                   help="Also render the reference mocap trajectory for comparison")
    p.add_argument("--episodes", type=int, default=3,
                   help="Number of episodes to render (default: 3)")
    return p.parse_args()


def load_config():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # --- Validate checkpoint path ---
    if not os.path.exists(args.path):
        print(f"[render_policy] Checkpoint not found at: {args.path}")
        print(f"  Train a policy first:  python train_loco.py")
        print(f"  Or specify a path:     python render_policy.py --path <checkpoint_dir>")
        sys.exit(1)

    # --- Ensure output directory exists for video ---
    if args.save_video:
        video_dir = os.path.dirname(args.save_video)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)

    cfg = load_config()
    mocap = cfg.get("env", {}).get("mocap", "walk1_subject1")
    headless = args.save_video is not None  # headless only when saving to video

    print(f"[render_policy] Checkpoint : {args.path}")
    print(f"[render_policy] Mocap      : {mocap}")
    print(f"[render_policy] Episodes   : {args.episodes}")
    print(f"[render_policy] Video      : {args.save_video or 'live display'}")

    from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf
    from loco_mujoco.algorithms import PPOJax
    from omegaconf import OmegaConf

    # --- Create the same env as training, but for rendering ---
    env = ImitationFactory.make(
        "UnitreeH1",  # MuJoCo (not MJX) for rendering
        lafan1_dataset_conf=LAFAN1DatasetConf([mocap]),
        headless=headless,
        reward_type="MimicReward",
    )

    # --- Load agent ---
    print(f"[render_policy] Loading agent from {args.path} ...")
    ckpt_path = args.path
    # If path is a directory, look for the .pkl file inside
    if os.path.isdir(ckpt_path):
        pkl = os.path.join(ckpt_path, "PPOJax_saved.pkl")
        if os.path.isfile(pkl):
            ckpt_path = pkl
    agent_state = PPOJax.load_agent(ckpt_path)

    # --- Build a minimal agent config matching training ---
    train_cfg = cfg.get("training", {})
    ppo_conf = {
        "hidden_layers": [512, 256],
        "activation": "tanh",
        "num_envs": 1,
        "num_steps": 200,
        "init_std": 0.2,
        "learnable_std": False,
        "normalize_env": True,
    }
    agent_conf = OmegaConf.create({"experiment": ppo_conf})
    agent_conf = PPOJax.init_agent_conf(env, agent_conf)

    # --- Render reference trajectory ---
    if args.show_reference:
        print(f"[render_policy] Rendering reference mocap trajectory ...")
        env.play_trajectory(n_episodes=1, render=not headless,
                            record=args.save_video is not None)

    # --- Render the learned policy ---
    print(f"[render_policy] Rendering policy for {args.episodes} episode(s) ...")
    PPOJax.play_policy_mujoco(
        agent_state, env, agent_conf,
        n_episodes=args.episodes,
        record=args.save_video is not None,
    )

    # --- Save video if requested ---
    if args.save_video:
        env.save_video(args.save_video)
        print(f"[render_policy] Video saved to {args.save_video}")

    print(f"[render_policy] Done.")


if __name__ == "__main__":
    main()
