# AutoRobot — Design Document

## 1. Vision

An autonomous AI scientist that teaches simulated robots to move. No human reward engineering. The LLM agent iterates on reward functions, trains RL policies, evaluates results, and accumulates improvements — all on Nebius infrastructure.

**One-liner**: "Autoresearch for robotics — an LLM runs 100 experiments to teach a robot to walk."

## 2. Core Insight

Eureka generates 16 reward functions in parallel and picks the best. Autoresearch makes one change at a time, evaluates, keeps/reverts, and accumulates. AutoRobot uses the autoresearch pattern because:

- **Compound improvements**: each successful experiment builds on the last
- **Deeper reasoning**: agent reflects on one result before proposing the next
- **Git history as demo**: the commit log IS the presentation — early falls → stumbles → runs
- **Simpler infra**: no need to orchestrate 16 parallel GPU jobs

## 3. Architecture

### 3.1 System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        NEBIUS CLOUD                          │
│                                                              │
│  ┌─────────────────┐         ┌─────────────────────────┐    │
│  │  Token Factory   │         │     GPU VM (H100)       │    │
│  │  (LLM Inference) │         │                         │    │
│  │                  │         │  ┌───────────────────┐  │    │
│  │  Qwen3-235B or   │◄───────►│  │    agent.py       │  │    │
│  │  DeepSeek-R1     │  API    │  │  (orchestrator)   │  │    │
│  │                  │         │  └─────┬─────────────┘  │    │
│  └─────────────────┘         │        │                 │    │
│                              │        ▼                 │    │
│                              │  ┌───────────────────┐  │    │
│                              │  │    train.py        │  │    │
│                              │  │  (PPO + Isaac Gym) │  │    │
│                              │  │  3-min budget      │  │    │
│                              │  └─────┬─────────────┘  │    │
│                              │        │                 │    │
│                              │        ▼                 │    │
│                              │  ┌───────────────────┐  │    │
│                              │  │   Isaac Gym        │  │    │
│                              │  │   4096 parallel    │  │    │
│                              │  │   environments     │  │    │
│                              │  └───────────────────┘  │    │
│                              └─────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 File Responsibilities

| File | Owner | Purpose | Modifiable by Agent? |
|------|-------|---------|---------------------|
| `program.md` | Human | Agent's instructions, research goals, constraints | No |
| `prepare.py` | Human | Isaac Gym env wrapper, observation space, evaluation harness, data loading | No |
| `reward.py` | Agent | Reward function — the only file the agent writes | **Yes** |
| `train.py` | Human | PPO training loop with fixed 3-min wall clock budget, metrics reporting | No |
| `agent.py` | Human | Orchestration loop: call LLM → write reward → run training → parse metrics → git commit/revert | No |
| `config.yaml` | Human | All hyperparameters: LLM model, API endpoint, training budget, num envs, task selection | No |

### 3.3 The Agent Loop (agent.py)

```python
# Pseudocode
for iteration in range(max_iterations):
    # 1. Build prompt
    context = read("prepare.py")       # env observations
    current_reward = read("reward.py") # current best reward
    history = git_log(last=5)          # recent experiments
    metrics = last_metrics             # from previous run

    # 2. Ask LLM for improved reward
    new_reward = llm.generate(
        system=read("program.md"),
        user=f"{context}\n\nCurrent reward:\n{current_reward}\n\nMetrics:\n{metrics}\n\nWrite improved reward."
    )

    # 3. Write reward and train
    write("reward.py", new_reward)
    result = subprocess.run(["python", "train.py"], timeout=200)
    new_metrics = parse_metrics(result.stdout)

    # 4. Keep or revert
    if new_metrics.primary > best_metrics.primary:
        git_commit(f"improvement: {new_metrics.primary:.3f} (was {best_metrics.primary:.3f})")
        best_metrics = new_metrics
    else:
        git_revert("reward.py")
        git_commit(f"reverted: {new_metrics.primary:.3f} < {best_metrics.primary:.3f}")
```

### 3.4 Reward Function Format

The agent writes a single function in `reward.py`:

```python
import torch

def compute_reward(obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        obs: dict with keys like 'root_lin_vel', 'root_ang_vel', 'joint_pos',
             'joint_vel', 'contact_forces', 'commands', etc.
    Returns:
        total_reward: (num_envs,) tensor
        components: dict of named reward components for logging
    """
    # Example: reward forward velocity, penalize energy
    forward_vel = obs['root_lin_vel'][:, 0]
    energy = torch.sum(obs['joint_vel'] ** 2, dim=-1)

    vel_reward = forward_vel
    energy_penalty = -0.001 * energy

    total = vel_reward + energy_penalty
    return total, {"forward_vel": vel_reward, "energy": energy_penalty}
```

### 3.5 Evaluation Metrics

| Metric | Description | Used For |
|--------|-------------|----------|
| `primary_score` | Task-specific: forward velocity (m/s) for locomotion, success rate for manipulation | Keep/revert decision |
| `episode_return` | Cumulative reward per episode | Debugging reward scale |
| `episode_length` | Steps before termination | Stability indicator |
| `reward_components` | Per-component breakdown (e.g., vel=2.1, energy=-0.3) | Fed back to LLM for reflection |
| `training_fps` | Simulation steps per second | Performance monitoring |

### 3.6 LLM Integration

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ["NEBIUS_API_KEY"]
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-Thinking",
    messages=messages,
    temperature=0.7,
    max_tokens=4096
)
```

**Why Qwen3-235B**: Strong code generation, reasoning mode for analyzing training metrics, available on Nebius at $0.20/$0.80 per million tokens (input/output).

**Fallback**: DeepSeek-R1 ($0.80/$2.40) or Llama-3.3-70B ($0.25/$0.75) for faster iteration.

## 4. Design Decisions

### 4.1 Why Ant (Not Humanoid or ShadowHand)?

| Factor | Ant | Humanoid | ShadowHand |
|--------|-----|----------|------------|
| Train time (3-min run) | Learns meaningful behavior | Barely starts | Barely starts |
| Visual clarity | Clear — 4 legs, runs | Good but slower | Hard to see progress |
| Observation space | Small (60 dims) | Medium (108 dims) | Large (211 dims) |
| Demo impact | High — dramatic improvement curve | Higher but slower | Highest but impractical |

Ant gives the best experiments-per-hour ratio for a 6-hour hackathon. We can always switch to Humanoid if time allows.

### 4.2 Why Sequential (Not Parallel)?

Eureka runs 16 candidates in parallel. We deliberately chose sequential because:
- **Simpler code**: no GPU multiplexing, no process management
- **Better narrative**: git log tells a linear story
- **Compound gains**: each step builds on the last, not independent sampling
- **Single GPU friendly**: works on 1x H100, no need for 8x A100

### 4.3 Why 3-Minute Budget (Not 5)?

- Ant converges to meaningful behavior in ~90 seconds on H100
- 3 minutes = 20 experiments/hour = 120 in 6 hours
- More experiments = more impressive demo numbers

### 4.4 Why Git-Per-Experiment?

- **Demo**: `git log --oneline` IS the presentation
- **Reproducibility**: any experiment can be replayed
- **Revert safety**: bad reward → `git checkout reward.py`
- **Diff review**: judges can see exactly what the AI changed each step

## 5. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Isaac Gym install fails on Nebius | Pre-test on local GPU first. Have a Docker fallback. |
| LLM generates broken reward code | Wrap `train.py` in try/catch. Log error, feed traceback to LLM, retry. |
| Reward hacking (robot exploits physics) | Add hard constraints in `prepare.py` (max episode length, contact checks) |
| LLM goes in circles (same reward) | Track hash of reward.py. If repeated, inject "try a fundamentally different approach" prompt. |
| 3 minutes too short for learning | Fall back to 5 minutes, accept fewer total experiments. |
| Nebius API rate limits | Use base tier (not fast), add exponential backoff. |

## 6. Demo Plan

1. **Live terminal**: show agent.py running — LLM thinking → reward written → training → metrics
2. **Git timelapse**: `git log --oneline` showing 50+ experiments with improving scores
3. **Side-by-side video**: early reward (robot falls) vs final reward (robot runs)
4. **Metrics chart**: primary_score over experiments (hockey stick curve)
5. **The pitch**: "Zero human reward engineering. The AI ran 87 experiments in 6 hours. It discovered 14 additive improvements. Final velocity: 3.2 m/s vs 1.1 m/s baseline."
