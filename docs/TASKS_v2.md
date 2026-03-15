# AutoRobot v2 — Mini-SONIC Task List

**Approach:** LLM-driven autoresearch loop that discovers SONIC-style motion tracking
rewards for humanoid locomotion, using LAFAN1 mocap data on Isaac Lab Newton (beta).

## Phase 1: Environment Setup (~30 min)

- [ ] **T1.1** Install Isaac Lab Newton on VM (H200)
  ```bash
  sudo apt install -y python3.12 python3.12-venv git build-essential
  curl -LsSf https://astral.sh/uv/install.sh | sh
  git clone https://github.com/isaac-sim/IsaacLab.git
  cd IsaacLab && git checkout develop
  uv venv --python 3.12 --seed env_isaaclab
  source env_isaaclab/bin/activate
  uv pip install -U torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
  ./isaaclab.sh -i
  ```
- [ ] **T1.2** Verify Isaac Lab Humanoid works
  ```bash
  ./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Humanoid-Direct-v0 --num_envs 128
  ```
- [ ] **T1.3** Download LAFAN1 mocap dataset
  ```bash
  pip install huggingface-hub
  huggingface-cli download unitreerobotics/LAFAN1_Retargeting_Dataset --repo-type dataset
  ```
  OR clone raw BVH:
  ```bash
  git lfs install
  git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset
  ```
- [ ] **T1.4** Install mocap tools: `pip install upc-pymotion` (BVH loading, FK, quaternion ops)
- [ ] **T1.5** Verify Nebius Token Factory API (already working)
- [ ] **T1.6** Install remaining deps: `pip install openai pyyaml matplotlib gitpython`

## Phase 2: Motion Reference Pipeline (~30 min)

- [ ] **T2.1** Write `mocap_loader.py`
  - Load LAFAN1 walking BVH (e.g., `walk1_subject1.bvh`)
  - Extract per-frame: joint positions, joint velocities, root position, root orientation
  - Retarget to Isaac Lab Humanoid skeleton (map LAFAN1 joints → Humanoid joints)
  - Output: numpy arrays of reference trajectory [T, num_joints] for positions + velocities
  - Provide `get_reference_frame(timestep)` → dict of target poses
- [ ] **T2.2** Write `reference_trajectory.py`
  - Pre-compute a single walking cycle from LAFAN1
  - Store as .npz file for fast loading during training
  - Provide cyclic indexing: `frame = timestep % cycle_length`
  - Include root position, root quaternion, joint positions, joint velocities
- [ ] **T2.3** Test: load reference, visualize a few frames, verify joint mapping

## Phase 3: Custom Isaac Lab Environment (~45 min)

- [ ] **T3.1** Write `envs/humanoid_tracking.py` — custom DirectRLEnv
  - Subclass `DirectRLEnv` from Isaac Lab
  - Load reference trajectory in `__init__`
  - Override `_get_observations()` → return robot state + reference state
  - Override `_get_rewards()` → call our dynamic `compute_reward(obs)`
  - Override `_get_dones()` → terminate on fall (z < 0.5)
  - Support 4096 parallel envs on GPU
- [ ] **T3.2** Write `envs/__init__.py` — register custom env
- [ ] **T3.3** Update `reward.py` for tracking reward
  - SONIC-style: `exp(-||joint_pos - ref_joint_pos||^2 / temp)` for each component
  - Components: root_pos_tracking, root_orient_tracking, joint_pos_tracking,
    joint_vel_tracking, action_rate_penalty
  - Include reference state in obs dict
- [ ] **T3.4** Update `program.md` for tracking reward paradigm
  - Explain that obs now includes `ref_joint_pos`, `ref_joint_vel`, `ref_root_pos`, `ref_root_quat`
  - LLM's job: find the best combination of tracking weights and temperatures
  - Strategy tips from SONIC paper
- [ ] **T3.5** Verify: run 1 training iteration with baseline tracking reward

## Phase 4: Training Script (~30 min)

- [ ] **T4.1** Write `train_isaac.py`
  - Use RSL-RL PPO (bundled with Isaac Lab)
  - Load custom humanoid_tracking env
  - Dynamically load reward from `reward.py`
  - Fixed time budget (3 min per experiment)
  - Print METRICS/COMPONENTS in same parseable format
  - 4096 parallel envs, headless
- [ ] **T4.2** Update `config.yaml`
  - training.script: "train_isaac.py"
  - training.num_envs: 4096
  - training.time_budget_seconds: 180
  - env.name: "HumanoidTracking"
- [ ] **T4.3** Test: run single training, verify metrics output

## Phase 5: Agent Loop (~15 min)

- [ ] **T5.1** Update `agent.py` to use train_isaac.py
- [ ] **T5.2** Test full loop: 3 iterations end-to-end
  - LLM generates tracking reward → train 3 min → evaluate → keep/revert → git commit
- [ ] **T5.3** Start autonomous loop, let it run

## Phase 6: Demo & Results (~30 min)

- [ ] **T6.1** Run `visualize.py` — score evolution chart
- [ ] **T6.2** Record videos: early (falling) vs late (walking) policies
- [ ] **T6.3** Generate summary stats
- [ ] **T6.4** Update README with results
- [ ] **T6.5** Push everything to GitHub

## Critical Path

```
T1.1 (Isaac Lab) → T1.2 (verify) → T3.1 (custom env) → T4.1 (train script) → T5.2 (test loop) → T5.3 (run)
                                   ↗
T1.3 (LAFAN1) → T2.1 (loader) → T2.2 (reference)
                                   ↘
                              T3.3 (reward.py) → T3.4 (program.md)
```

## Key Differences from v1

| Aspect | v1 (Ant + SB3) | v2 (Mini-SONIC) |
|--------|---------------|-----------------|
| Robot | Generic MuJoCo Ant/Humanoid | Isaac Lab Humanoid with mocap reference |
| Physics | CPU MuJoCo (1,500 fps) | GPU Isaac Lab Newton (millions fps) |
| Reward paradigm | Pure task reward (velocity) | Motion tracking (SONIC-style) |
| LLM's job | Invent reward from scratch | Optimize tracking weights + temperatures |
| Training per iter | 270K steps (useless) | ~50M+ steps (meaningful) |
| Expected result | Robot barely moves | Robot walks matching mocap |
| Mocap data | None | LAFAN1 walking clips |

## Fallback Plan

If Isaac Lab Newton fails to install:
1. Fall back to LocoMuJoCo (`pip install loco-mujoco`) — has LAFAN1 + DeepMimic built-in
2. Fall back to current SB3 + MuJoCo setup (already working, just slow)
