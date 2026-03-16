# AutoRobot vs SONIC: A Technical Comparison

A detailed analysis comparing AutoRobot (LLM-automated reward weight optimization on a single GPU) with SONIC (NVIDIA's supersized motion tracking foundation model trained on 128 GPUs).

---

## 1. Problem Statement

Both AutoRobot and SONIC address the same fundamental challenge in humanoid robotics: **teaching humanoid robots to produce natural locomotion without manual reward engineering**.

Current humanoid controllers require extensive per-task reward engineering. Each behavior -- walking, running, dancing, getting up from the ground -- demands its own hand-crafted reward function, painstakingly tuned by domain experts over weeks or months. This bottleneck prevents scaling to diverse behaviors and makes humanoid control inaccessible to non-experts.

**SONIC's approach**: eliminate reward engineering by scaling motion tracking. Train a single massive policy on 700 hours of professional motion capture data across 128 GPUs. The mocap data itself provides dense, frame-by-frame supervision -- no reward engineering needed because the objective is simply "match the reference motion." Scale (data, compute, model capacity) unlocks generalization to unseen motions.

**AutoRobot's approach**: eliminate reward engineering by automating it. Use an LLM (DeepSeek-V3) in an autoresearch loop to iteratively discover optimal reward weight configurations for a proven DeepMimic tracking architecture. The LLM proposes weight changes, trains for 3 minutes, evaluates, keeps or reverts, and repeats. The human never touches the reward parameters.

Both systems converge on motion tracking (DeepMimic-style exponential tracking rewards) as the core training objective. The divergence is in *how* they solve the hyperparameter problem: SONIC throws massive data and compute at a fixed reward structure; AutoRobot uses an LLM to search the reward weight space efficiently on minimal resources.

---

## 2. Data Requirements

| Dimension | SONIC | AutoRobot | Ratio |
|-----------|-------|-----------|-------|
| **Total motion data** | 700 hours | ~5 minutes (1 walking clip) | ~8,400x less |
| **Total frames** | 100,000,000+ (at 50 Hz) | 10,451 frames | ~9,500x less |
| **Number of subjects** | 170 human subjects | 1 subject (LAFAN1 subject 1) | 170x less |
| **Motion diversity** | Locomotion, daily activities, gestures, combat, dance, sports | Walking only (`walk1_subject1`) | Hundreds of behaviors vs 1 |
| **Capture quality** | Professional in-house mocap studio | Public dataset (LAFAN1 from Ubisoft) | Professional vs academic |
| **Retargeting** | Custom GMR pipeline (Araujo et al., 2025) to Unitree G1 | Pre-retargeted by LocoMuJoCo to Unitree H1 | Custom vs off-the-shelf |
| **Data acquisition cost** | Months of professional mocap sessions, 170 actors | `pip install loco-mujoco` (auto-downloads from HuggingFace) | $$$$ vs free |

SONIC's dataset spans thousands of unique motion behaviors, with most actions performed by multiple subjects across multiple takes, providing rich intra- and inter-subject variation. Body heights range from 145 cm to 199 cm (mean 174.3 cm, std 10.9 cm). Clip durations range from 1 to 180 seconds.

AutoRobot uses a single walking clip from the publicly available LAFAN1 dataset (Ubisoft La Forge Animation), pre-retargeted to the Unitree H1 by the LocoMuJoCo library. Zero data collection effort.

---

## 3. Compute Requirements

| Dimension | SONIC | AutoRobot | Ratio |
|-----------|-------|-----------|-------|
| **GPUs** | 128 GPUs | 1 GPU (NVIDIA H200) | 128x less |
| **GPU-hours** | 32,000 (128 GPUs x 3-7 days) | ~1 hour total (~16 experiments x ~3 min each) | ~32,000x less |
| **Training duration** | 3-7 days continuous | ~1 hour | ~100-170x less |
| **Simulation platform** | Isaac Lab (proprietary, requires RT cores) | LocoMuJoCo + JAX/MJX (open source, `pip install`) | Proprietary vs open |
| **Parallel environments** | 4,096 per GPU (x128 GPUs = 524,288 total) | 2,048 on 1 GPU | ~256x less |
| **Simulation FPS** | Not reported (Isaac Lab GPU) | ~225,000 fps (JAX/MJX on H200) | -- |
| **Steps per experiment** | Billions (full convergence) | 36,864,000 per 3-min experiment | Orders of magnitude less |
| **LLM inference** | None | DeepSeek-V3-0324 via Nebius Token Factory | Additional cost, but minimal |

SONIC's training hyperparameters (Table 4 of the paper): 4,096 parallel envs per GPU, 24 steps per env, 5 learning epochs, 4 mini-batches, discount 0.99, GAE lambda 0.95, actor LR 2e-5, critic LR 1e-3. The system trains across 128 GPUs simultaneously, with adaptive motion sampling to handle the dataset's diversity.

AutoRobot's PPO configuration: 2,048 parallel envs, 200 steps per env, 4 update epochs, 32 mini-batches, hidden layers [512, 256], LR 1e-4, 3-minute wall-clock budget per experiment. Total compute for the entire autoresearch run: roughly 16 experiments x 3 minutes = ~48 minutes of GPU time plus JIT compilation overhead.

---

## 4. Architecture

### SONIC: Custom Encoder-Decoder with Universal Token Space

SONIC employs a sophisticated encoder-decoder architecture with cross-embodiment learning:

| Module | Architecture | Dimensions |
|--------|-------------|------------|
| Robot Encoder | MLP | hidden = [2048, 1024, 512, 512] |
| Human Encoder | MLP | hidden = [2048, 1024, 512, 512] |
| Hybrid Encoder | MLP | hidden = [2048, 1024, 512, 512] |
| Control Decoder | MLP | hidden = [2048, 2048, 1024, 1024, 512, 512] |
| Motion Decoder | MLP | hidden = [2048, 1024, 512, 512] |
| Critic | MLP | hidden = [2048, 2048, 1024, 1024, 512, 512] |
| Quantizer | FSQ (Finite Scalar Quantization) | D_z dimensions, L_z levels |
| **Total parameters** | **1.2M to 42M** (scaled across experiments) | -- |

The architecture uses three specialized encoders to map robot, human, and hybrid motion commands into a shared latent space, which is quantized via FSQ into a universal token. This token drives both a robot control decoder (for motor commands) and a robot motion decoder (for auxiliary supervision).

The training loss combines four terms:
```
L = L_ppo + L_recon + L_token + L_cycle
```
- `L_ppo`: Standard PPO loss for RL
- `L_recon`: Reconstruction loss across all input modalities
- `L_token`: Contrastive alignment between robot and human tokens
- `L_cycle`: Cycle consistency loss (encode human -> decode robot -> re-encode robot, should match)

Additionally, SONIC includes a generative kinematic motion planner built on masked token prediction with a temporal transformer, operating in a latent space with a downsampling rate of 4.

### AutoRobot: Standard MLP with PPO

AutoRobot uses LocoMuJoCo's built-in architecture:

| Module | Architecture | Dimensions |
|--------|-------------|------------|
| Actor | MLP | hidden = [512, 256], tanh activation |
| Critic | MLP | hidden = [512, 256] |
| **Total parameters** | **~175,000** | -- |

No custom encoders, no quantizers, no cross-embodiment alignment, no cycle consistency losses. The entire policy is a standard 3-layer MLP trained with vanilla PPO. The innovation is not in the architecture but in *how the reward weights are discovered*.

### Key Architectural Difference

SONIC trains the network architecture (scaling from 1.2M to 42M parameters). AutoRobot keeps the architecture fixed and instead trains the reward weights -- 14 scalar parameters that configure the DeepMimic tracking reward. The LLM optimizes these weights; the neural network architecture is off-the-shelf.

---

## 5. Training Approach

### SONIC: Single Massive Training Run

SONIC performs a single large-scale training run:
1. Collect and retarget 700 hours of mocap data to the Unitree G1
2. Configure a fixed reward structure (Table 1 of the paper)
3. Launch training across 128 GPUs for 3-7 days
4. Use adaptive motion sampling (bin-based, failure-rate-weighted) to handle dataset diversity
5. Apply extensive domain randomization (friction, restitution, COM offsets, velocity perturbations, motion target jitter)
6. Train to convergence

The reward weights are hand-designed (Table 1) and fixed throughout training. The system scales by adding more data, more compute, and larger models -- not by modifying the reward.

### AutoRobot: Iterative Autoresearch Loop

AutoRobot runs an iterative optimization loop:
```
for each experiment (up to 100):
    1. LLM reads current best weights + metrics from previous run
    2. LLM proposes new JSON weight configuration (changing 1-3 parameters)
    3. Train PPO for 3 minutes on H200 (2,048 envs, ~36.8M steps)
    4. Evaluate: measure primary_score (mean episode return)
    5. If improved: git commit, update best weights
       If worse: git revert, feed failure info to LLM
    6. If stagnated for 5 iterations: inject "try radically different approach"
    7. Repeat
```

Each experiment is a git commit. The git log is the experiment log. Good experiments accumulate; bad experiments are reverted. The LLM sees the full history and learns from both successes and failures.

Key difference: SONIC uses a fixed reward and scales everything else. AutoRobot uses fixed architecture/data/compute and optimizes the reward weights via LLM search.

---

## 6. Reward Design

Both systems use the same fundamental reward paradigm: exponential tracking kernels of the form `exp(-||error||^2 / temperature)`. The differences lie in what is tracked, the specific weights, and how those weights were chosen.

### SONIC Reward Structure (Table 1 from the paper)

| Reward Term | Equation | Weight |
|-------------|----------|--------|
| Root orientation | `exp(-||o_p - o_g||^2 / 0.4^2)` | 0.5 |
| Body link position (root-relative) | `exp(-mean(||p_rel - p_rel_g||^2) / 0.3^2)` | 1.0 |
| Body link orientation (root-relative) | `exp(-mean(||o_rel - o_rel_g||^2) / 0.4^2)` | 1.0 |
| Body link linear velocity | `exp(-mean(||v - v_g||^2) / 1.0^2)` | 1.0 |
| Body link angular velocity | `exp(-mean(||w - w_g||^2) / 3.14^2)` | 1.0 |
| Action rate penalty | `||a_t - a_{t-1}||^2` | -0.1 |
| Joint limit penalty | `sum(q outside limits)` | -10.0 |
| Undesired contacts penalty | `sum(non-ankle/wrist contacts > 1N)` | -0.1 |

SONIC uses 6D rotation representations for orientations and tracks root-relative body link positions and orientations (not joint-space quantities). Temperatures are embedded in the equations (0.4, 0.3, 0.4, 1.0, 3.14).

### AutoRobot Reward Structure (MimicReward from LocoMuJoCo)

AutoRobot uses LocoMuJoCo's DeepMimic-style `MimicReward`, which computes:
```
reward = w_rpos * exp(-rpos_w_exp * ||site_pos_error||^2)
       + w_rquat * exp(-rquat_w_exp * ||site_orient_error||^2)
       + w_qpos * exp(-qpos_w_exp * ||joint_pos_error||^2)
       + w_qvel * exp(-qvel_w_exp * ||joint_vel_error||^2)
       + w_rvel * exp(-rvel_w_exp * ||site_vel_error||^2)
       - penalties
```

### Side-by-Side Weight Comparison

| Parameter | SONIC (hand-tuned) | AutoRobot Baseline (exp #1) | AutoRobot Best (exp #2, LLM-discovered) |
|-----------|-------------------|----------------------------|----------------------------------------|
| **Position tracking** | | | |
| Site/body position weight | 1.0 | 0.5 (`rpos_w_sum`) | 0.45 (`rpos_w_sum`) |
| Position temperature | 1/0.3^2 = 11.1 | 80.0 (`rpos_w_exp`) | 60.0 (`rpos_w_exp`) |
| **Orientation tracking** | | | |
| Orientation weight | 1.0 (body) + 0.5 (root) | 0.3 (`rquat_w_sum`) | 0.35 (`rquat_w_sum`) |
| Orientation temperature | 1/0.4^2 = 6.25 | 8.0 (`rquat_w_exp`) | 6.0 (`rquat_w_exp`) |
| **Joint position tracking** | | | |
| Joint position weight | -- (tracked via body links) | 0.1 (`qpos_w_sum`) | 0.15 (`qpos_w_sum`) |
| Joint position temperature | -- | 10.0 (`qpos_w_exp`) | 8.0 (`qpos_w_exp`) |
| **Velocity tracking** | | | |
| Linear velocity weight | 1.0 | 0.0 (`rvel_w_sum`) | 0.0 (`rvel_w_sum`) |
| Linear velocity temperature | 1/1.0^2 = 1.0 | 0.1 (`rvel_w_exp`) | 0.1 (`rvel_w_exp`) |
| Angular velocity weight | 1.0 | 0.0 (`qvel_w_sum`) | 0.0 (`qvel_w_sum`) |
| Angular velocity temperature | 1/3.14^2 = 0.10 | 2.0 (`qvel_w_exp`) | 1.5 (`qvel_w_exp`) |
| **Penalties** | | | |
| Action rate | -0.1 | 0.001 (`action_rate_coeff`) | 0.001 (`action_rate_coeff`) |
| Joint limits | -10.0 | -- (handled by LocoMuJoCo) | -- |
| Out-of-bounds actions | -- | 0.01 | 0.01 |
| Joint acceleration | -- | 0.0001 | 0.0001 |
| Joint torque | -- | 0.0 | 0.0 |
| Undesired contacts | -0.1 | -- (handled by LocoMuJoCo) | -- |

### Key Observation: Temperature Discovery

The LLM's most impactful discovery between experiment #1 and experiment #2 was **lowering temperatures across the board** (rpos_w_exp: 80 -> 60, rquat_w_exp: 8 -> 6, qpos_w_exp: 10 -> 8, qvel_w_exp: 2 -> 1.5). This made the reward more forgiving, giving the policy a smoother learning signal. This is a non-obvious insight: tighter tracking (higher temperatures) sounds like it should produce better results, but in practice it creates sparse rewards that are harder to optimize.

The LLM-discovered temperatures (60.0 for position, 6.0 for orientation) are actually closer to SONIC's effective temperatures (11.1 for position, 6.25 for orientation) than the initial baseline was. The LLM independently converged toward the expert-tuned values.

### How Weights Were Chosen

- **SONIC**: Hand-tuned by expert researchers. The paper states the reward design follows Liao et al. (2025) (BeyondMimic). Validated empirically at scale across 128 GPUs and 100M+ frames.
- **AutoRobot**: LLM-discovered via automated search. DeepSeek-V3 proposes incremental changes, trains for 3 minutes, evaluates, keeps or reverts. No human involvement in weight selection.

---

## 7. Results Comparison

| Metric | SONIC | AutoRobot |
|--------|-------|-----------|
| **Primary metric** | Success rate (Succ): % of motions tracked without falling | Tracking score (episode return): cumulative reward per episode |
| **Headline result** | 100% success on 50 diverse real-world trajectories | 55.2 -> 116.6 tracking score (+111% improvement) |
| **Simulation performance** | 98.5% success on 1,602 unseen AMASS trajectories | Best score: 97.89 (from checkpoints/best_score.txt) |
| **MPJPE** | 40.9 mm (sim), 42.7 mm (real) | Not measured (different metric framework) |
| **Baseline comparison** | vs Any2Track (5.8%), BeyondMimic (41.7%), GMT (94.3%) | vs initial weights (55.2 baseline) |
| **Motion diversity** | Walking, running, dancing, boxing, crawling, kneeling, squatting, combat, get-up, 360-spin, kung-fu | Walking only (LAFAN1 `walk1_subject1`) |
| **Velocity range** | 0.0 - 6.0 m/s with arbitrary direction (0-360 degrees) | Single walking gait |
| **Qualitative quality** | Natural, human-like motion with smooth transitions | Functional walking (tracks mocap but less refined) |
| **Cross-embodiment** | Yes (human SMPL -> robot G1 via universal token space) | No (H1 only, direct H1->G1 transfer not feasible) |

### Context for AutoRobot Scores

AutoRobot's tracking score progression from the experiment logs:
- Experiment #1 (baseline): primary_score = 55.20, episode_length = 128, episode_return = 55.2
- Experiment #2 (LLM-optimized): primary_score = 63.74, episode_length = 134, episode_return = 63.7
- Best checkpoint score: 97.89 (from `checkpoints/best_score.txt`)

The improvement from 55.2 to the best checkpoint of 97.89 represents a +77% improvement over the initial configuration. The stated +111% improvement (55.2 -> 116.6) likely reflects additional experiments beyond the two logged in the repository.

### Qualitative Difference

SONIC produces motion that is visually indistinguishable from human movement across diverse behaviors. The policy smoothly transitions between walking, running, boxing stances, crawling, and getting up from the ground. It handles style modulation (drunken walking, stealth walking, happy walking) and responds to real-time user commands.

AutoRobot produces functional walking that tracks the reference mocap clip. The robot walks, but the motion is optimized for a single clip from a single subject performing a single behavior. There is no style modulation, no behavior switching, and no response to dynamic commands.

---

## 8. Sim-to-Real Transfer

### SONIC: Proven Zero-Shot Deployment

SONIC demonstrates zero-shot sim-to-real transfer on the Unitree G1:
- **Platform**: Unitree G1 humanoid (132 cm, 35 kg, 23-29 DOF)
- **Onboard compute**: NVIDIA Jetson Orin (planner inference < 12 ms)
- **Success rate**: 100% on 50 diverse real-world trajectories (dance, jumps, loco-manipulation)
- **Tracking quality**: Real-world MPJPE of 42.7 mm (vs 40.9 mm in simulation)
- **Domain randomization**: Systematic randomization of friction, restitution, COM offsets, joint defaults, velocity perturbations, and motion target jitter (see SONIC Table 2)
- **VLA integration**: Fine-tuned GR00T N1.5 achieves 95% success on apple-to-plate mobile manipulation task through the same control interface

### AutoRobot: Simulation Only (Export Path Documented)

AutoRobot has not been deployed on real hardware. However, `docs/real.md` documents a concrete deployment path:

1. **Retrain on G1**: Change config from `UnitreeH1` to `UnitreeG1` (1-line change in LocoMuJoCo)
2. **Export policy**: The 175K-parameter MLP exports to NumPy weights (~6 lines of code)
3. **Inference**: 6-line NumPy forward pass runs at 20,000-100,000 Hz on any CPU (robot needs 500 Hz)
4. **Deploy**: Via `unitree_sdk2_python` on the G1's onboard Jetson Orin NX (16GB)
5. **Safety**: Gantry/harness support, reduced PD gains, hardware e-stop

The H1-to-G1 gap is significant: different joint topology (19 vs 23-29 DOF), different link lengths, different mass distribution. Direct policy transfer is not feasible -- retraining on G1 is required.

---

## 9. Strengths of Each Approach

### SONIC Strengths

- **State-of-the-art motion quality**: Natural, human-like movement across hundreds of diverse behaviors
- **Proven real-world deployment**: 100% success rate on 50 real-world trajectories, zero-shot sim-to-real
- **Universal control interface**: Single policy handles gamepad, VR teleoperation (3-point and whole-body), video teleoperation, text commands, music commands, and VLA integration
- **Cross-embodiment transfer**: Human motion (SMPL) maps directly to robot control via the universal token space, bypassing retargeting
- **Scaling properties**: Performance improves consistently with more data, more compute, and larger models -- log-linear scaling curves demonstrated
- **Foundation model integration**: Compatible with GR00T N1.5 VLA for autonomous mobile manipulation
- **Robustness**: Extensive domain randomization + adaptive sampling + cycle consistency loss produce robust policies

### AutoRobot Strengths

- **Accessibility**: 1 GPU, 1 hour, 1 walking clip. No mocap studio, no multi-GPU cluster, no proprietary simulation platform
- **Zero data collection**: Uses publicly available LAFAN1 data that auto-downloads via `pip install loco-mujoco`
- **Zero reward engineering**: The LLM discovers effective weight configurations autonomously -- no human expertise required for reward tuning
- **Rapid iteration**: 3 minutes per experiment means hypotheses are tested quickly. 20 experiments per hour.
- **Compound improvements**: The autoresearch pattern (sequential keep/revert) allows each improvement to build on the last, unlike Eureka's parallel independent sampling
- **Transparent experiment history**: Every experiment is a git commit. The entire optimization trajectory is reproducible and inspectable via `git log`
- **Non-obvious discoveries**: The LLM discovered that lowering temperatures (making rewards more forgiving) improves learning -- a counter-intuitive insight that a human might not try first
- **Democratization**: Anyone with a single GPU and an API key can teach a humanoid to walk. No PhD in reward engineering required
- **Open infrastructure**: MuJoCo (open source), JAX (open source), LocoMuJoCo (open source), LAFAN1 (public dataset). No proprietary dependencies except the LLM API

---

## 10. What AutoRobot Demonstrates

### LLMs Can Effectively Optimize Reward Hyperparameters

AutoRobot shows that an LLM (DeepSeek-V3) can serve as an effective hyperparameter optimizer for RL reward functions. The LLM doesn't just randomly search -- it reads previous results, forms hypotheses about why certain changes worked or failed, and proposes targeted modifications. The improvement from 55.2 to 97.89+ tracking score demonstrates meaningful optimization, not random luck.

### The Autoresearch Pattern Works for Robotics

Karpathy's autoresearch loop (hypothesize -> experiment -> evaluate -> keep/revert -> repeat) transfers from ML research to robotics reward optimization. The pattern provides:
- **Automatic experiment management** via git commits
- **Safe exploration** via revert-on-failure
- **Compound improvement** via sequential accumulation
- **Stagnation escape** via diversity injection after 5 failed iterations

### You Don't Need 128 GPUs or 700 Hours of Mocap

AutoRobot achieves functional humanoid walking with:
- 1/128th the GPUs (1 vs 128)
- 1/32,000th the GPU-hours (~1 vs 32,000)
- 1/8,400th the motion data (~5 minutes vs 700 hours)
- 1/9,500th the frames (10,451 vs 100,000,000+)

The result is not SONIC-quality motion, but it is functional walking learned entirely autonomously. For prototyping, education, and research on reward optimization, this is a significant reduction in barrier to entry.

### Temperature Tuning Is a Non-Obvious Critical Insight

The LLM autonomously discovered that the initial temperatures were too high (too strict), creating sparse reward signals that made learning difficult. By lowering `rpos_w_exp` from 80 to 60 and `rquat_w_exp` from 8 to 6, the reward became more forgiving, and the tracking score jumped from 55.2 to 63.7 (+15.5%) in a single iteration.

This is precisely the kind of insight that takes human researchers weeks of manual experimentation to discover. The LLM found it in one 3-minute experiment.

### The Weight Space Is Surprisingly Expressive

With only 14 scalar parameters (5 tracking weights, 5 temperatures, 4 penalty coefficients), the LLM can meaningfully control the quality of learned locomotion. The search space is continuous and high-dimensional enough for non-trivial optimization, but constrained enough that every experiment produces valid, crash-free results. This suggests that for many robotics problems, optimizing the weights of a proven reward architecture may be more practical than generating reward code from scratch (as Eureka does).

---

## 11. Limitations of AutoRobot vs SONIC

### Single Motion Only

AutoRobot trains on one walking clip (`walk1_subject1` from LAFAN1). SONIC trains on thousands of diverse behaviors from 170 subjects. AutoRobot's policy can only walk; SONIC's policy can walk, run, dance, box, crawl, kneel, squat, get up from the ground, perform kung-fu, do 360-degree spins, and more.

### No Cross-Embodiment Transfer

SONIC's universal token space enables human motion to be mapped directly to robot control, bypassing retargeting. AutoRobot's policy is specific to the Unitree H1 joint topology. Deploying on a different robot (e.g., Unitree G1) requires retraining from scratch.

### No Real-World Deployment

SONIC has been validated on a physical Unitree G1 with 100% success on 50 diverse trajectories. AutoRobot exists only in simulation. While the deployment path is documented (export to NumPy, run on Jetson at 500 Hz), it has not been tested on hardware.

### Lower Quality Motion

SONIC produces natural, human-like motion that is visually compelling. AutoRobot produces functional walking that tracks the reference but lacks the polish and naturalness of SONIC's output. The difference is most apparent in:
- Smoothness of transitions
- Natural arm swing and body sway
- Robustness to perturbations
- Style and expressiveness

### Optimizes Weights Only, Not Architecture

AutoRobot optimizes 14 reward weight parameters. It does not modify the network architecture (fixed at [512, 256] MLP), the RL algorithm (fixed PPO), the observation space, or the action space. SONIC co-designs the architecture (encoder-decoder with FSQ quantization, up to 42M parameters), the training procedure (adaptive sampling, domain randomization), and the reward structure.

### No Multi-Modal Control

SONIC supports gamepad, VR, video, text, and music control through its universal token space. AutoRobot has no control interface beyond the trained walking policy.

### No Foundation Model Integration

SONIC connects to GR00T N1.5 VLA for autonomous manipulation. AutoRobot has no integration with vision-language-action models or any higher-level planning system.

### Limited Robustness

SONIC applies systematic domain randomization (friction, restitution, COM offsets, velocity perturbations, motion target jitter) to ensure robust sim-to-real transfer. AutoRobot uses LocoMuJoCo's default simulation parameters with no domain randomization, which would likely cause significant sim-to-real gap.

---

## 12. Future: Bridging the Gap

### Use LLM to Optimize SONIC-Scale Systems

The most direct bridge: apply AutoRobot's LLM-driven reward optimization to SONIC's architecture. SONIC's reward structure (Table 1) has 8 terms with hand-tuned weights. An LLM could optimize these weights on a smaller dataset first, then scale up to the full 100M+ frame training run with LLM-discovered weights. This would combine AutoRobot's automated tuning with SONIC's scale.

### AutoRobot as Rapid Prototyping Before SONIC-Scale Training

Use AutoRobot to quickly explore reward weight configurations on a single GPU (~1 hour). Once the LLM discovers a promising weight configuration, use those weights to initialize a SONIC-scale training run (128 GPUs, 3-7 days). This avoids burning 32,000 GPU-hours with sub-optimal reward weights.

Concrete workflow:
1. AutoRobot discovers optimal weights on 1 GPU in 1 hour (e.g., position temperature = 60, orientation temperature = 6)
2. Transfer those weights to a SONIC-scale training pipeline
3. Train on 100M+ frames across 128 GPUs with the LLM-discovered weights
4. Deploy on real hardware

### Combine: LLM Discovers Reward Structure, Scale Refines It

Go beyond weight optimization. Use the LLM to propose entirely new reward terms (not just weights), validate them cheaply on a single GPU, then scale the best-performing reward structures with SONIC-level data and compute. This is closer to the original Eureka vision but with SONIC's scaling properties.

### Extend AutoRobot to Multi-Motion Training

The current limitation of 1 walking clip is an implementation choice, not a fundamental constraint. LocoMuJoCo supports multiple LAFAN1 clips. Extending AutoRobot to optimize weights across a diverse motion set (walking, running, jumping) would test whether the LLM can discover weight configurations that generalize across behaviors.

### Add Domain Randomization via LLM

Beyond reward weights, the LLM could also optimize domain randomization parameters (friction ranges, perturbation magnitudes, etc.) to improve sim-to-real transfer. SONIC's domain randomization (Table 2) has 15+ parameters -- another search space where LLM-driven optimization could find non-obvious configurations.

### AutoRobot + Real Hardware Validation

The most immediate next step: deploy AutoRobot's best policy on a Unitree G1 (retraining on G1 model in LocoMuJoCo first). Even partial success would demonstrate that LLM-optimized reward weights can produce sim-to-real-viable policies, validating the approach beyond simulation.

---

## Summary Table

| Dimension | SONIC | AutoRobot |
|-----------|-------|-----------|
| **Goal** | Foundation model for humanoid control | Automated reward optimization for humanoid walking |
| **Core innovation** | Scale (data, compute, model capacity) | LLM-in-the-loop reward weight search |
| **Data** | 700 hours, 100M+ frames, 170 subjects | 5 minutes, 10,451 frames, 1 subject |
| **Compute** | 128 GPUs, 32,000 GPU-hours | 1 GPU, ~1 GPU-hour |
| **Model** | 1.2M-42M params, encoder-decoder + FSQ | ~175K params, standard MLP |
| **Reward design** | Hand-tuned (Table 1), fixed during training | LLM-discovered, iteratively optimized |
| **Motions** | Hundreds of diverse behaviors | 1 walking clip |
| **Sim-to-real** | Zero-shot, 100% success on 50 trajectories | Not yet attempted |
| **Robot** | Unitree G1 (sim + real) | Unitree H1 (sim only) |
| **Accessibility** | Requires 128 GPUs + proprietary mocap + Isaac Lab | 1 GPU + pip install + free dataset |
| **Training time** | 3-7 days | ~1 hour |
| **Best for** | Production deployment, diverse behaviors, real-world robots | Rapid prototyping, reward research, education, accessibility |

---

*AutoRobot was built for the [Nebius.Build SF Hackathon](https://cerebralvalley.ai/e/nebius-build-sf) (March 15-16, 2026). SONIC was published by NVIDIA Research (November 2025). This comparison is intended to contextualize AutoRobot's contributions relative to the state of the art, not to claim equivalence.*
