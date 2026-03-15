# AutoRobot v3 — Mini-SONIC with LocoMuJoCo

**Approach:** LLM-driven autoresearch loop that discovers SONIC-style motion tracking
rewards for Unitree H1 humanoid locomotion, using LAFAN1 mocap data via LocoMuJoCo
(GPU-accelerated via JAX/MJX). Trained on Nebius H200.

**Pitch:** "SONIC needs 700hrs of mocap and 128 GPUs. We use an LLM to automatically
discover SONIC-style rewards with 1 walking clip on 1 GPU."

## Phase 1: Install & Verify (~15 min)

- [x] **T1.1** Install JAX with CUDA on VM
  ```bash
  source ~/autorobot_venv/bin/activate
  pip install -U "jax[cuda12]"
  python -c "import jax; print(jax.default_backend())"  # should print: gpu
  ```
- [x] **T1.2** Install LocoMuJoCo
  ```bash
  pip install loco-mujoco
  ```
- [x] **T1.3** Verify LAFAN1 data loads (auto-downloads from HuggingFace)
  ```python
  from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf
  env = ImitationFactory.make(
      "UnitreeH1",
      lafan1_dataset_conf=LAFAN1DatasetConf(["walk1_subject1"]),
  )
  env.play_trajectory(n_episodes=1, n_steps_per_episode=100)
  print("LAFAN1 walking data loaded successfully")
  ```
- [x] **T1.4** Install remaining deps
  ```bash
  pip install openai pyyaml matplotlib gitpython
  ```
- [x] **T1.5** Verify Nebius Token Factory API (already working — DeepSeek-V3)

## Phase 2: Understand LocoMuJoCo & Write Training Script (~30 min)

- [ ] **T2.1** Study LocoMuJoCo's built-in training examples
  - Read `examples/training_examples/` — PPO, DeepMimic, AMP, GAIL
  - Understand the env API: observation space, action space, reward interface
  - Understand how custom rewards work (`reward_type="custom"`, `reward_callback`)
- [ ] **T2.2** Write `train_loco.py` — training script using LocoMuJoCo
  - Create UnitreeH1 env with LAFAN1 walk1_subject1 reference
  - Use LocoMuJoCo's built-in JAX PPO (GPU-accelerated)
  - Dynamically load reward from `reward.py`
  - Fixed time budget (3 min per experiment)
  - Print METRICS/COMPONENTS in parseable format:
    ```
    METRICS: primary_score=X.XX episode_return=XX.X episode_length=XXX fps=XXXXX
    COMPONENTS: root_tracking=X.XX joint_tracking=X.XX energy=-X.XX ...
    ```
  - Headless, GPU-accelerated
- [ ] **T2.3** Test: run single training (3 min), verify metrics output

## Phase 3: Reward & Agent Updates (~20 min)

- [ ] **T3.1** Update `reward.py` for motion tracking
  - SONIC-style exponential tracking kernels
  - Components: root_pos_tracking, root_orient_tracking, joint_pos_tracking,
    joint_vel_tracking, action_rate_penalty
  - Obs dict includes reference state from LocoMuJoCo
  - Use SONIC weights: orientation=0.5, position=1.0, velocity=1.0, penalties=-0.1
- [ ] **T3.2** Update `program.md` for LocoMuJoCo + tracking paradigm
  - Document the obs space (robot state + LAFAN1 reference state)
  - Explain the LLM's job: optimize tracking weights, temperatures, penalty scales
  - Include SONIC-inspired tips
- [ ] **T3.3** Update `config.yaml`
  - training.script: "train_loco.py"
  - env.name: "UnitreeH1"
  - env.mocap: "walk1_subject1"
  - model: "deepseek-ai/DeepSeek-V3-0324"
- [ ] **T3.4** Update `agent.py`
  - Point to train_loco.py
  - Ensure code extraction handles LocoMuJoCo obs format

## Phase 4: End-to-End Test (~15 min)

- [ ] **T4.1** Run single iteration manually
  - LLM generates reward → train_loco.py trains 3 min → metrics printed → verify
- [ ] **T4.2** Run 3 iterations of agent.py end-to-end
  - Verify: LLM → train → evaluate → keep/revert → git commit cycle works
- [ ] **T4.3** Fix any bugs found during testing

## Phase 5: Let It Run (~60+ min)

- [ ] **T5.1** Start agent.py in tmux on VM
  ```bash
  tmux new -s autorobot
  source ~/autorobot_venv/bin/activate
  cd ~/nebiushack
  python agent.py
  ```
- [ ] **T5.2** Monitor: `watch -n 60 'cd ~/nebiushack && git log --oneline | head -20'`
- [ ] **T5.3** Let it run while preparing demo

## Phase 6: Demo & Results (~20 min)

- [ ] **T6.1** Run `visualize.py` — score evolution chart
- [ ] **T6.2** Generate summary stats (experiments, improvements, best score)
- [ ] **T6.3** Update README with actual results
- [ ] **T6.4** Push everything to GitHub

## Critical Path

```
T1.1 (JAX) → T1.2 (LocoMuJoCo) → T1.3 (LAFAN1) → T2.1 (study API) → T2.2 (train_loco.py)
                                                                              ↓
T1.5 (API) ──────────────────────────────────────→ T3.1 (reward) → T3.4 (agent) → T4.1 (test)
                                                                                       ↓
                                                                                 T5.1 (run!) → T6.1 (demo)
```

## Key Advantages of LocoMuJoCo

| Feature | Detail |
|---------|--------|
| LAFAN1 data | Auto-downloads, pre-retargeted to Unitree H1 |
| GPU training | JAX/MJX under the hood, thousands of parallel envs |
| Built-in algos | PPO, DeepMimic, AMP, GAIL — all in JAX |
| Humanoid models | Unitree H1, H1v2, G1, Atlas, Talos, and more |
| Custom rewards | `reward_callback` API for dynamic reward injection |
| Training time | ~36 min for DeepMimic walking on RTX 3080 Ti |
| Install | `pip install loco-mujoco` — clean, no tarballs |

## v1 → v3 Comparison

| Aspect | v1 (SB3 + MuJoCo) | v3 (LocoMuJoCo) |
|--------|-------------------|-----------------|
| Robot | Generic Humanoid-v5 | Unitree H1 (real robot model) |
| Physics | CPU, 1,500 fps | GPU via JAX/MJX |
| Mocap data | None | LAFAN1 walking (auto-downloaded) |
| Reward paradigm | Pure velocity reward | SONIC-style motion tracking |
| LLM's job | Invent reward from scratch | Optimize tracking weights + temps |
| Training per iter | 270K steps (useless) | Millions of steps (meaningful) |
| Expected result | Robot barely moves | Robot walks like a human |

## Fallback

If LocoMuJoCo install fails on Python 3.12 or JAX has issues:
1. Use MuJoCo Playground (`pip install playground`) — similar speed, different API
2. Fall back to current SB3 + MuJoCo (already working, just slow)




