# AutoRobot: The Journey

A detailed account of building AutoRobot for the Nebius.Build SF hackathon (March 15-16, 2026) -- every decision, every pivot, every failure, and what we learned.

---

## 1. The Idea

It started with HIMLoco. We had been poking at legged robot locomotion repos -- HIMLoco, walk-these-ways, legged_gym -- fascinated by how simulated robots learn to walk through reinforcement learning. The reward functions in these repos were hand-crafted by PhDs, tuned over months, full of magic numbers. We thought: what if an LLM could do that?

Then we found the [Nebius.Build SF hackathon](https://cerebralvalley.ai/e/nebius-build-sf). The theme was robotics + inference optimization, and Nebius was offering GPU cloud credits and access to their Token Factory (hosted LLM inference). Perfect alignment.

We brainstormed three ideas:

1. **Voice-commanded robot** -- speak natural language commands, LLM translates to robot actions. Cool demo, but incremental over existing work.
2. **Inference-optimized VLA** -- a vision-language-action model optimized for fast inference on Nebius. Technically interesting but too deep for a hackathon.
3. **LLM reward designer** -- combine Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) pattern with [Eureka](https://github.com/eureka-research/Eureka)'s LLM reward generation. An AI that teaches itself to teach a robot to walk.

We chose idea 3. The pitch wrote itself: *"Zero human reward engineering. The AI ran 87 experiments in 6 hours. It discovered 14 additive improvements. Final velocity: 3.2 m/s vs 1.1 m/s baseline."*

The key insight was combining two existing ideas that hadn't been put together before:

- **Autoresearch** (Karpathy): write code, run it, evaluate results, keep or revert, repeat. Every experiment is a git commit. The git log IS the paper.
- **Eureka** (NVIDIA): use an LLM to write reward functions for robotic RL, instead of having humans hand-craft them.

Eureka runs 16 reward candidates in parallel and picks the best. We chose the autoresearch pattern instead -- sequential, one experiment at a time -- because each successful experiment builds on the last. Compound improvements over independent sampling. And the git history tells a linear story that's way more compelling for a demo.

---

## 2. Initial Architecture

We designed the system around five core files and a strict separation of concerns:

| File | Owner | Role |
|------|-------|------|
| `prepare.py` | Human | Environment wrapper, observation space docs, evaluation harness |
| `reward.py` | **LLM** | The reward function -- the only file the agent writes |
| `train.py` | Human | PPO training loop with fixed 3-minute wall-clock budget |
| `agent.py` | Human | Orchestration: call LLM, write reward, train, evaluate, git commit/revert |
| `visualize.py` | Human | Parse git log + metrics into a score evolution chart |

The loop:

```
Agent reads program.md (instructions)
    -> reads prepare.py (robot observations)
    -> writes reward.py (new reward function)
    -> runs train.py (3-min PPO training)
    -> reads metrics (velocity, success rate)
    -> keeps (git commit) or reverts
    -> hypothesizes next improvement
    -> repeat
```

Every experiment is a git commit. Good experiment? `exp-42: score=2.31 (improvement, was 1.89)`. Bad experiment? `exp-43: score=0.45 (reverted, best=2.31)`. Run `git log --oneline` and you see the entire evolution of how the AI taught a robot to move. That IS the demo.

We originally planned to use:
- **Isaac Gym Preview 4** for GPU-accelerated simulation (4,096 parallel environments on a single GPU)
- **Ant quadruped** as the robot (simple enough to learn in 3 minutes, dramatic enough for a demo)
- **rl_games** for PPO training (the RL library Eureka uses)
- **Nebius Token Factory** for LLM inference (Qwen3-235B-Thinking at $0.20/$0.80 per million tokens)

The config was straightforward:

```yaml
training:
  time_budget_seconds: 180  # 3 minutes per experiment
  num_envs: 4096
  device: "cuda:0"
agent:
  max_iterations: 100
  retry_on_error: 3
  stagnation_threshold: 5
```

3 minutes per experiment = 20 experiments/hour = 120 in 6 hours. More than enough for a compelling demo.

---

## 3. First Pivot: Isaac Gym to MuJoCo

The plan hit its first wall immediately: **Isaac Gym requires a manual tarball download from NVIDIA's developer portal**. You can't `pip install` it. You can't automate it. You have to manually log in, accept a license, download a 2GB tarball, extract it, and install from source. On a cloud VM, this is painful. Isaac Gym Preview 4 is also deprecated -- NVIDIA has moved on to Isaac Lab.

We pivoted to **MuJoCo Humanoid-v5 + Stable-Baselines3**. The reasoning:
- MuJoCo is `pip install gymnasium[mujoco]` -- one command
- SB3 is `pip install stable-baselines3` -- one command
- Humanoid-v5 is a built-in Gymnasium environment
- No tarballs, no developer accounts, no deprecation warnings

We also upgraded from the Ant quadruped to the Humanoid. It's a bigger challenge but a much more impressive demo -- a bipedal robot learning to walk looks way more compelling than a four-legged ant.

But switching from Isaac Gym to MuJoCo uncovered a **critical bug** in our design. Here's what happened:

We built a `CustomRewardWrapper` that replaced the environment's built-in reward with our LLM-generated reward function. The wrapper intercepted `step()` calls, computed the custom reward, and returned it instead of the original. Sounds correct, right?

**The bug**: the custom reward was only used during **evaluation**, not during **training**. SB3's PPO was optimizing against Humanoid-v5's built-in reward (which rewards forward velocity + staying alive), while our evaluation measured performance with the LLM's custom reward. The agent would "improve" its reward function, but the trained policy hadn't actually been shaped by that reward.

The fix was wrapping the environment at creation time, before passing it to PPO:

```python
class CustomRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, _original_reward, terminated, truncated, info = self.env.step(action)
        obs_dict = mujoco_obs_to_dict(obs)
        reward_tensor, components = self.reward_fn(obs_dict)
        custom_reward = float(reward_tensor.item())
        return obs, custom_reward, terminated, truncated, info
```

Each subprocess in `SubprocVecEnv` loaded its own copy of `reward.py` to avoid pickling issues. This was commit `580da7e`: *"Fix critical bug: custom reward now used during training, not just eval"*.

Then came the second problem with MuJoCo: **speed**. CPU MuJoCo runs at about 1,500 simulation fps. Isaac Gym on a GPU runs at 300,000+ fps. That's a 200x difference. With a 3-minute training budget and 8 parallel CPU environments, we could get maybe 270,000 total steps. That's not enough for a humanoid to learn anything meaningful.

We needed GPU-accelerated physics. MuJoCo on CPU was a dead end for our hackathon timeline.

---

## 4. The SONIC Paper

While we were wrestling with the MuJoCo speed problem, a teammate committed `sonic_paper.pdf` to the repo (commit `9c74cde`). SONIC is an NVIDIA paper that builds a humanoid foundation model via motion tracking. Instead of hand-crafting task rewards (go forward, turn left, jump), SONIC trains a single policy to track thousands of hours of motion capture data.

Key insights from SONIC that shaped our approach:

1. **Motion tracking rewards work better than hand-crafted task rewards**. Instead of defining "walk forward" as "maximize x-velocity minus energy penalty," you define "match this mocap clip of a human walking." The robot learns natural-looking gaits instead of exploit-y waddles.

2. **Exponential kernel rewards**: `exp(-||error||^2 / temperature)`. This gives a smooth reward signal that goes from 1.0 (perfect tracking) to 0.0 (way off), with temperature controlling how forgiving the reward is. SONIC uses separate temperatures for position, orientation, velocity, and joint angles.

3. **Weight hierarchy**: SONIC weights position tracking highest (0.5), then orientation (0.3), then velocity (0.0-0.2). This tells the policy what matters most.

We incorporated SONIC's insights into our reward design (commit `400a908`). The reward function became a weighted sum of exponential tracking terms:

```
reward = w_rpos * exp(-rpos_temp * ||site_pos_error||^2)
       + w_rquat * exp(-rquat_temp * ||site_orient_error||^2)
       + w_qpos * exp(-qpos_temp * ||joint_pos_error||^2)
       - penalties
```

This was a philosophical shift: the LLM's job was no longer "invent a reward function from scratch" but "optimize the weights and temperatures of a proven reward architecture." A more constrained problem, but one where small changes have measurable effects.

---

## 5. GPU Reality Check

We provisioned a Nebius VM with an NVIDIA H200 GPU. The H200 is a compute GPU -- a monster for matrix multiplication -- but it has **NO RT (ray-tracing) cores**.

This matters because:

- **Isaac Lab stable** (the successor to Isaac Gym) requires **Isaac Sim**, which requires RT cores for rendering. It literally won't launch on an H200. Dead end.

- **Isaac Lab Newton** (beta branch) was designed specifically for headless GPU compute on cards like H200. But we ran into three problems:
  1. Humanoid physics had bugs in the beta -- joints would explode on reset
  2. JIT compilation took over an hour on first run
  3. Beta instability -- random crashes during training

- **GEAR-SONIC**: the implementation of the SONIC paper. But the training code was **NOT released** -- only inference code for the pre-trained model. We could demo the result but couldn't train anything ourselves.

We researched every alternative:

| Option | Verdict |
|--------|---------|
| Isaac Lab stable | Requires RT cores. Won't run on H200. |
| Isaac Lab Newton (beta) | Humanoid bugs, 1hr+ JIT, beta crashes. |
| GEAR-SONIC | Inference only. No training code released. |
| MJX + Brax | GPU-accelerated MuJoCo via JAX. Works but no built-in humanoid training pipeline. |
| MuJoCo Playground | Similar to MJX+Brax but different API. |
| LocoMuJoCo | pip-installable, LAFAN1 mocap pre-retargeted to Unitree H1, GPU via JAX/MJX, built-in DeepMimic reward. |
| EnvPool | Fast vectorized envs but still CPU physics. |

---

## 6. Second Pivot: LocoMuJoCo + LAFAN1

LocoMuJoCo won. Here's why it was perfect for our use case:

- **`pip install loco-mujoco`** -- no tarballs, no developer accounts, no manual downloads
- **LAFAN1 mocap data** auto-downloads from HuggingFace, **pre-retargeted to Unitree H1** (a real humanoid robot)
- **GPU-accelerated** via JAX/MJX under the hood -- thousands of parallel environments on one GPU
- **Built-in DeepMimic reward** (`MimicReward`) with configurable weights -- exactly the SONIC-style exponential tracking we wanted
- **Built-in PPO** in JAX -- no need for a separate RL library
- **Training time**: ~36 minutes for DeepMimic walking on an RTX 3080 Ti. On an H200, much faster.

The Unitree H1 was also a better demo robot than a generic MuJoCo humanoid. It's a real production robot -- showing an LLM teaching a Unitree H1 to walk is way more compelling than teaching a stick-figure humanoid.

This was commit `e4f7bd0`: *"Pivot to LocoMuJoCo: Unitree H1 + LAFAN1 mocap tracking"*.

We wrote a new task list (`TASKS_v3.md`) from scratch. The v1 plan had 5 phases and assumed Isaac Gym. The v3 plan had 6 phases and was built around LocoMuJoCo's API:

```python
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf
from loco_mujoco.algorithms import PPOJax

env = ImitationFactory.make(
    "MjxUnitreeH1",
    lafan1_dataset_conf=LAFAN1DatasetConf(["walk1_subject1"]),
    headless=True,
    reward_type="MimicReward",
    reward_params=weights,
)
```

Three lines to create a GPU-accelerated Unitree H1 environment with LAFAN1 walking mocap and a DeepMimic reward. Compare that to the 50+ lines of Isaac Gym setup we had before.

---

## 7. Third Pivot: Code to JSON Weights

The original autoresearch loop had the LLM writing full Python reward functions -- arbitrary code that would be executed during training. This worked conceptually but had practical problems:

1. **Injecting custom reward into LocoMuJoCo's JAX pipeline is complex.** LocoMuJoCo's training is JIT-compiled JAX. You can't just swap in a Python function that uses numpy -- it needs to be JAX-compatible, traced, compiled.

2. **Full code generation is fragile.** The LLM might import wrong modules, use wrong tensor shapes, introduce NaN-producing operations. Every broken reward function wastes a 3-minute training cycle.

3. **LocoMuJoCo's built-in MimicReward is already really good.** It implements exactly the SONIC-style exponential tracking we wanted. The architecture is proven. We just need to find the right weights.

So we pivoted the LLM's job. Instead of writing Python reward functions, it optimizes **14 JSON weight parameters**:

```json
{
    "qpos_w_exp": 10.0,       // Joint position tracking temperature
    "qvel_w_exp": 2.0,        // Joint velocity tracking temperature
    "rpos_w_exp": 100.0,      // Site position tracking temperature
    "rquat_w_exp": 10.0,      // Site orientation tracking temperature
    "rvel_w_exp": 0.1,        // Site velocity tracking temperature
    "qpos_w_sum": 0.0,        // Joint position weight in total reward
    "qvel_w_sum": 0.0,        // Joint velocity weight
    "rpos_w_sum": 0.5,        // Site position weight
    "rquat_w_sum": 0.3,       // Site orientation weight
    "rvel_w_sum": 0.0,        // Site velocity weight
    "action_out_of_bounds_coeff": 0.01,
    "joint_acc_coeff": 0.0,
    "joint_torque_coeff": 0.0,
    "action_rate_coeff": 0.0
}
```

This was commit `b98d48f`: *"Refactor to JSON weight optimization with LocoMuJoCo MimicReward"*.

The trade-off: the LLM has less creative freedom (it can't invent entirely new reward structures), but the system is more robust (JSON can't crash, weights can be validated before training). And the LLM can still explore a massive search space -- 14 continuous parameters with different scales and interactions.

`agent.py` was refactored: instead of extracting Python code blocks, it extracts JSON, validates the keys, writes `reward_weights.json`, and trains. The commit/revert logic stayed the same. The git log still tells the story.

---

## 8. Debugging and Lessons Learned

Every "simple" step had at least one unexpected blocker. Here's the catalog of pain:

### API key got overwritten by git pull

We committed a `.env` file with a placeholder API key so the team could share config. Then someone did `git pull` on the VM and it overwrote the real API key with the placeholder. The agent ran, got 401 errors from Nebius, and we spent 20 minutes wondering why.

**Fix**: `agent.py` loads `.env` using `os.environ.setdefault()` -- it only sets the variable if it's not already set. So `export NEBIUS_API_KEY=real-key` in your shell takes precedence over whatever is in `.env`. But the real fix is: **never commit API keys to git**. Even placeholder `.env` files are dangerous because `git pull` can silently overwrite your real keys.

### Thinking models return content=None

We initially configured Qwen3-235B-Thinking as our primary model. "Thinking" models (those with chain-of-thought reasoning) on Nebius return `content=None` in the response -- the reasoning tokens are in a separate field. Our code did `r.choices[0].message.content`, got `None`, and couldn't extract any reward function.

Commit `36dcb4a` switched us to DeepSeek-V3, a non-thinking model that reliably returns content in the standard field. We also added a defensive `or ""` fallback:

```python
content = r.choices[0].message.content or ""
```

### Model names differ between docs and API catalog

The Nebius documentation listed model names like `Qwen/Qwen3-235B-Thinking`. The actual API catalog uses `Qwen/Qwen3-235B-Thinking` but with specific version suffixes. We kept getting "model not found" errors until we tested with `scripts/test_nebius_api.py` and mapped the correct model identifiers.

Commit `ffcd963`: *"Fix model IDs to match Nebius Token Factory catalog"*.

### Python output buffering hid agent.py's output for hours

We started `agent.py` in a tmux session, detached, and came back an hour later. No output in the terminal. Nothing in the logs. We thought it was stuck.

It wasn't stuck -- it was running fine. Python's default stdout buffering was holding all the output in a buffer because tmux's pseudo-terminal triggered block buffering instead of line buffering. The metrics were being computed and the git commits were happening, but we couldn't see anything.

**Fix**: run with `PYTHONUNBUFFERED=1 python agent.py` or `python -u agent.py`. From then on, every print statement appeared immediately.

### conda TOS acceptance required on fresh installs

We scripted the VM setup: download Miniconda, create environment, install packages. On a fresh install, Miniconda requires you to accept the Terms of Service interactively. Our automated script hung waiting for input.

**Fix**: use `bash Miniconda3-latest-Linux-x86_64.sh -b` (the `-b` flag accepts the license automatically for batch installs). Or even better, use `uv` instead of conda (which is what TASKS_v2 eventually switched to).

### VM had no public IP initially

We provisioned a Nebius GPU VM through the console. It launched fine -- but it had no public IP address. We couldn't SSH into it. The Nebius console showed the VM running, but there was no way to connect.

**Fix**: you have to explicitly add a public IP through the Nebius console after provisioning. It's not assigned by default for GPU VMs (probably a cost/security consideration).

### SSH key mismatch

After getting the public IP, we still couldn't SSH in. The teammate who provisioned the VM used their SSH key, not ours. Nebius VMs don't have password authentication enabled by default.

**Fix**: the teammate had to add our public key through the Nebius console's metadata settings. This took another 15 minutes of back-and-forth.

### PPO config keys missing

When we switched to LocoMuJoCo's built-in PPO, we hit errors about missing configuration keys: `validation`, `n_seeds`, `vmap_across_seeds`, `learnable_std`, `weight_decay`, `proportion_env_reward`. These weren't in the LocoMuJoCo docs but were required by the PPO implementation.

Commit `0ce1548`: *"Fix missing PPO config keys (validation, n_seeds, etc.)"* -- we had to read the LocoMuJoCo source code to find all the required keys and their default values:

```python
ppo_conf = {
    "hidden_layers": [512, 256],
    "lr": 1e-4,
    "num_envs": num_envs,
    "num_steps": 200,
    "total_timesteps": int(budget * num_envs * 100),
    "update_epochs": 4,
    "num_minibatches": 32,
    "n_seeds": 1,
    "vmap_across_seeds": True,
    "learnable_std": False,
    "weight_decay": 0.0,
    "proportion_env_reward": 0.0,
    "validation": {"active": False, "num_steps": 100,
                   "num_envs": 100, "num": 10},
    # ... plus all the standard PPO params
}
```

### 11 bugs found during deep validation

Commit `ea0da1c` was titled *"Fix 11 bugs found during deep validation testing"*. We don't have the detailed list, but this was the result of running the entire pipeline end-to-end for the first time and fixing everything that broke. The lesson: **test the full pipeline end-to-end before building features**.

---

## 9. Technical Decisions and Tradeoffs

### Sequential (autoresearch) vs parallel (Eureka)

Eureka runs 16 reward candidates in parallel and picks the best. We chose sequential for three reasons:

1. **Simpler code** -- no GPU multiplexing, no process management, no result aggregation
2. **Compound improvements** -- each experiment builds on the last successful one, enabling incremental refinement
3. **Better narrative** -- the git log tells a linear story with a clear progression

The trade-off: sequential is slower at exploration. If the agent gets stuck in a local optimum, it takes several "stagnation" iterations before it tries something radically different. We mitigated this with a stagnation detector:

```python
if no_improve >= stag_thresh:
    # inject "try a radically different approach" into the prompt
```

### 3-minute training budget

Each experiment trains for exactly 3 minutes of wall-clock time. This was a deliberate choice:

- **Pro**: 20 experiments/hour. Over a 6-hour hackathon, that's 120 experiments. Big number = impressive demo.
- **Pro**: fast iteration. Bad reward functions fail fast, good ones show signal quickly.
- **Con**: complex behaviors need more training time. A humanoid walking might need 30+ minutes to converge.
- **Con**: 3-minute scores are noisy. A reward that looks bad at 3 minutes might be great at 30 minutes.

With LocoMuJoCo's GPU-accelerated training, 3 minutes gives us millions of simulation steps, which is enough to see a meaningful learning signal even for humanoid walking.

### DeepSeek-V3 over Qwen3-Thinking

We switched from Qwen3-235B-Thinking to DeepSeek-V3-0324 for a simple reason: **non-thinking models return content reliably**. Thinking models stream their reasoning into a separate field and sometimes return `content=None` in the main response body. For an autonomous agent that runs unattended for hours, reliability beats reasoning quality.

DeepSeek-V3 was also cheaper on Nebius and produced perfectly adequate JSON weight suggestions.

### JSON weights over Python code

This was our biggest design trade-off. Python code generation gives the LLM maximum creative freedom but introduces fragility (syntax errors, wrong imports, NaN rewards). JSON weight optimization constrains the LLM to a fixed reward architecture but guarantees that every experiment produces valid results.

For a hackathon, robustness wins. We wanted 100 experiments with zero crashes, not 20 experiments with 15 crashes. The JSON approach delivered that.

### Unitree H1 over generic humanoid

The Unitree H1 is a production humanoid robot. Saying "we taught a Unitree H1 to walk using LLM-optimized rewards" sounds way more compelling than "we taught a stick-figure MuJoCo humanoid to walk." The H1 also had pre-retargeted LAFAN1 mocap data available through LocoMuJoCo, which saved us the entire motion retargeting pipeline we would have had to build otherwise (see TASKS_v2, which had a whole phase dedicated to mocap loading and retargeting).

---

## 10. What We'd Do Differently

If we were starting this project again tomorrow:

### Start with LocoMuJoCo from day 1

We burned hours on Isaac Gym (can't automate install), then MuJoCo (too slow on CPU), then Isaac Lab Newton (beta bugs), before landing on LocoMuJoCo. LocoMuJoCo was `pip install loco-mujoco` with GPU acceleration, built-in mocap data, and a configurable reward system. We should have surveyed the landscape more thoroughly before writing any code.

### Never commit API keys or .env files to git

Even placeholder `.env` files are dangerous. Use `.env.example` with fake values and add `.env` to `.gitignore`. We committed a real API key to git history (visible in `.env`) that should have been rotated immediately.

### Test the full pipeline end-to-end before building features

We built each component in isolation (LLM call, training script, agent loop, git integration) and assumed they'd work together. They didn't. Eleven bugs emerged during end-to-end testing. We should have had a minimal end-to-end loop running by hour 1, then iterated on it.

### Use PYTHONUNBUFFERED=1 for all background processes

Python's output buffering is invisible until you run something in the background (tmux, nohup, screen) and wonder why there's no output. Add `PYTHONUNBUFFERED=1` to every launch script, every systemd unit, every tmux session.

### Research GPU compatibility BEFORE provisioning VMs

We provisioned an H200 and then discovered that Isaac Lab stable requires RT cores. That's a $10+/hour mistake. Before provisioning any GPU VM, check:
- Does your framework require specific GPU features (RT cores, Tensor cores, specific CUDA compute capability)?
- Does your framework support headless mode?
- Is your framework's version compatible with the GPU driver version on the VM?

### Build the agent loop around JSON from the start

We designed the system for full Python code generation, then pivoted to JSON weights when we realized LocoMuJoCo's built-in reward was better than anything the LLM could write from scratch. If we'd started with the "optimize parameters of a proven architecture" framing, we would have saved a full refactor cycle.

---

## Timeline of Pivots

| Phase | What | Duration | Outcome |
|-------|------|----------|---------|
| v1 | Isaac Gym + Ant + Python reward code | Design phase | Blocked: Isaac Gym can't be automated |
| v1.5 | MuJoCo Humanoid-v5 + SB3 | Implementation | Working but 200x too slow on CPU |
| v2 | Isaac Lab Newton + LAFAN1 + SONIC-style tracking | Research | Blocked: H200 has no RT cores, beta bugs |
| v3 | LocoMuJoCo + Unitree H1 + LAFAN1 + JSON weights | Implementation | **Shipped** |

The final architecture (v3) was simpler, more robust, and more impressive than anything we originally designed. Sometimes the best engineering is knowing when to stop building and start constraining.

---

## The v1 to v3 Evolution at a Glance

| Aspect | v1 (Isaac Gym + Ant) | v3 (LocoMuJoCo + H1) |
|--------|---------------------|----------------------|
| Robot | Generic Ant quadruped | Unitree H1 (real robot model) |
| Physics | Isaac Gym GPU (4096 envs) | JAX/MJX GPU (2048 envs) |
| Install | Manual tarball download | `pip install loco-mujoco` |
| Mocap data | None | LAFAN1 walking (auto-downloaded) |
| Reward paradigm | LLM writes Python from scratch | LLM optimizes JSON weight parameters |
| Training per iter | N/A (install blocked) | Millions of GPU-accelerated steps |
| LLM model | Qwen3-235B-Thinking | DeepSeek-V3-0324 |
| Expected result | Ant runs forward | Humanoid walks like a human |

---

## The Git Log Tells the Story

Every commit in this repo reflects a real decision:

```
0ce1548 Fix missing PPO config keys (validation, n_seeds, etc.)
b98d48f Refactor to JSON weight optimization with LocoMuJoCo MimicReward
e4f7bd0 Pivot to LocoMuJoCo: Unitree H1 + LAFAN1 mocap tracking
a0070e0 Add v3 task list: LocoMuJoCo + LAFAN1 + autoresearch
81c64c0 Add v2 task list: Mini-SONIC with LAFAN1 + Isaac Lab Newton
580da7e Fix critical bug: custom reward now used during training, not just eval
400a908 Incorporate SONIC paper insights into reward and agent instructions
36dcb4a Switch to Humanoid + DeepSeek-V3 + fix LLM None response bug
925cf18 Add MuJoCo fallback trainer and configurable train script
ffcd963 Fix model IDs to match Nebius Token Factory catalog
ea0da1c Fix 11 bugs found during deep validation testing
9c74cde sonic paper
a93c233 Add Phase 1 core files: prepare, reward, train, agent, visualize
e808efb Add .env with placeholder Nebius API key for team sharing
d65f45c Fix Phase 0 scripts after validation testing
2fef5e0 Add Phase 0 setup scripts: VM setup, API test, env verification
f8f2728 Add project docs: design, tasks, setup, and agent instructions
```

Read it bottom-to-top and you see the journey: docs first, then setup scripts, then core files, then bug fixes, then the SONIC pivot, then the MuJoCo fallback, then the LocoMuJoCo pivot, then the JSON refactor, then more bug fixes.

Three task lists (`TASKS.md`, `TASKS_v2.md`, `TASKS_v3.md`) chronicle the three pivots in planning. Three training scripts (`train.py`, `train_mujoco.py`, `train_loco.py`) chronicle the three pivots in implementation.

The messiness is the point. Building for a hackathon means making fast decisions with incomplete information, hitting walls, pivoting, and shipping what works. AutoRobot shipped.

---

## For Anyone Building Something Similar

If you want to build an LLM-driven RL reward optimization system, here is what we learned:

1. **Use a proven reward architecture and optimize its parameters.** Don't ask the LLM to write arbitrary reward code. Give it a well-designed reward structure (like DeepMimic exponential tracking) and let it tune the weights. More robust, faster iteration, fewer crashes.

2. **LocoMuJoCo is currently the easiest path to GPU-accelerated humanoid locomotion.** `pip install`, built-in mocap, built-in training algorithms, real robot models. Start here.

3. **The autoresearch pattern (sequential keep/revert) works well for reward optimization.** Each improvement compounds. The git history is a natural experiment log. The stagnation detector prevents getting stuck.

4. **Budget 50% of your time for debugging infrastructure, not algorithms.** API keys, GPU compatibility, package versions, output buffering, SSH access -- these "simple" things will eat your hackathon if you don't plan for them.

5. **Test end-to-end immediately.** Build the minimal loop (LLM call -> write config -> train 30 seconds -> read metrics -> print result) on day 0 and iterate from there. Don't build components in isolation.

6. **Non-thinking LLM models are more reliable for structured output.** If you need the LLM to return JSON or code blocks reliably, use a standard model (DeepSeek-V3, Llama-3.3-70B) instead of a thinking/reasoning model that might return `content=None`.
