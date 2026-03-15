# AutoRobot

**AI agent that autonomously discovers robot locomotion behaviors — zero human reward engineering.**

Combines [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) loop with [Eureka](https://github.com/eureka-research/Eureka)'s LLM reward generation, running on [Nebius](https://nebius.com) GPU cloud.

The agent reads a robot simulation, writes a reward function, trains an RL policy for 3 minutes, evaluates the result, keeps or reverts, and repeats. ~12 experiments/hour, ~100 overnight. Each experiment is a git commit — you can replay the entire evolution of how the AI taught a robot to move.

## Quickstart

```bash
# 1. Clone
git clone https://github.com/anjan04/nebiushack.git
cd nebiushack

# 2. Install dependencies
pip install -e .

# 3. One-time env setup (downloads Isaac Gym assets)
python prepare.py

# 4. Set API keys
export NEBIUS_API_KEY="your-key"

# 5. Manual test run (single 3-min training)
python train.py

# 6. Start the autonomous agent loop
python agent.py
```

## How It Works

```
Agent reads program.md (instructions)
    → reads prepare.py (robot observations)
    → writes reward.py (new reward function)
    → runs train.py (3-min PPO on Isaac Gym)
    → reads metrics (velocity, success rate)
    → keeps (git commit) or reverts
    → hypothesizes next improvement
    → repeat
```

## Project Structure

```
autorobot/
├── program.md        # Agent instructions (human-editable)
├── prepare.py        # Env setup, observations, evaluation (immutable)
├── reward.py         # Reward function (agent's sole canvas)
├── train.py          # PPO training loop, fixed time budget
├── agent.py          # Orchestrates the autoresearch loop
├── config.yaml       # Hyperparameters, API settings, GPU config
└── docs/
    ├── DESIGN.md     # Architecture & design decisions
    ├── TASKS.md      # Implementation task list
    └── SETUP.md      # Detailed setup guide
```

## Built With

- **Nebius Token Factory** — LLM inference (Qwen3-235B / DeepSeek-R1)
- **Nebius GPU Cloud** — H100s for RL training
- **Isaac Gym Preview 4** — GPU-accelerated robot simulation
- **PPO (rl_games)** — Reinforcement learning algorithm

## Built For

[Nebius.Build SF Hackathon](https://cerebralvalley.ai/e/nebius-build-sf) — March 15-16, 2026

## License

MIT
