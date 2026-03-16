# Getting Started — AutoRobot

Pick up where we left off, or start fresh on a new VM. This guide covers everything from zero to a running autonomous training loop.

## What You Need

- A Nebius GPU VM (H200, H100, or any NVIDIA GPU with CUDA)
- A Nebius Token Factory API key
- SSH access to the VM
- ~15 minutes for setup

## Quick Start (Existing VM)

If the VM already has everything installed, just SSH in and run:

```bash
ssh ubuntu@<VM_IP>
cd ~/nebiushack
git pull origin main

# Set API key
echo 'NEBIUS_API_KEY=<your-key>' > .env

# Set git config (needed for experiment commits)
git config user.email "autorobot@nebiushack.dev"
git config user.name "AutoRobot Agent"

# Activate environment
source ~/autorobot_venv/bin/activate

# Start the agent
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
nohup python3 -u agent.py > /tmp/agent_output.log 2>&1 &

# Monitor
tail -f /tmp/agent_output.log
```

## Fresh VM Setup (From Scratch)

### Step 1: Provision a Nebius GPU VM

In [Nebius Console](https://console.nebius.com):
- Platform: `gpu-h200-sxm` (or H100)
- Preset: `1gpu-16vcpu-200gb`
- Boot disk: Ubuntu 24.04 with NVIDIA GPUs (CUDA 13), 200+ GB SSD
- Network: Enable public IP
- Add your SSH public key

### Step 2: SSH in and clone

```bash
ssh ubuntu@<VM_IP>
git clone https://github.com/anjan04/nebiushack.git
cd nebiushack
```

### Step 3: Create Python virtual environment

```bash
python3 -m venv ~/autorobot_venv
source ~/autorobot_venv/bin/activate
pip install --upgrade pip
```

### Step 4: Install dependencies

```bash
# JAX with CUDA (GPU-accelerated training)
pip install -U "jax[cuda12]"

# LocoMuJoCo (humanoid env + LAFAN1 mocap + PPO)
# PyPI version may have bugs — install from source:
pip install git+https://github.com/robfiras/loco-mujoco.git

# Other deps
pip install openai pyyaml matplotlib gitpython
```

### Step 5: Verify GPU

```bash
python3 -c "import jax; print('Backend:', jax.default_backend()); print('Devices:', jax.devices())"
# Should print: Backend: gpu
```

### Step 6: Verify LocoMuJoCo + LAFAN1

```bash
python3 -c "
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf
env = ImitationFactory.make('UnitreeH1', lafan1_dataset_conf=LAFAN1DatasetConf(['walk1_subject1']))
print('Environment created successfully')
"
# First run downloads LAFAN1 data (~10K frames) from HuggingFace
```

### Step 7: Test training

```bash
python3 train_loco.py --time_budget=30 --num_envs=512
# Should print METRICS: primary_score=XX.XX ... after ~60s (includes JIT compilation)
```

### Step 8: Configure and start the agent

```bash
# Set API key
echo 'NEBIUS_API_KEY=<your-key>' > .env

# Git config for experiment commits
git config user.email "autorobot@nebiushack.dev"
git config user.name "AutoRobot Agent"

# Install xvfb + ffmpeg for video rendering
sudo apt-get install -y xvfb ffmpeg

# Start the autonomous loop
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
nohup python3 -u agent.py > /tmp/agent_output.log 2>&1 &
```

## Monitoring

```bash
# Watch the agent output
tail -f /tmp/agent_output.log

# Check experiment history
git log --oneline | grep exp- | head -20

# Check best score
cat checkpoints/best_score.txt

# Check GPU usage
nvidia-smi
```

## Key Files

| File | What It Does |
|------|-------------|
| `agent.py` | Main loop: LLM → write weights → train → evaluate → keep/revert → git commit |
| `train_loco.py` | Trains Unitree H1 with MimicReward + LAFAN1 walking via LocoMuJoCo PPOJax |
| `reward_weights.json` | The 14 DeepMimic reward weights the LLM optimizes |
| `program.md` | Instructions the LLM reads (what to optimize, strategy tips) |
| `config.yaml` | All settings: LLM model, training budget, num envs, etc. |
| `render_policy.py` | Renders trained policy to video |
| `visualize.py` | Generates score evolution chart from experiment logs |
| `checkpoints/best/` | Best trained policy checkpoint |
| `logs/` | JSON logs for each experiment |

## Rendering a Video

```bash
# Install rendering deps (once)
sudo apt-get install -y xvfb ffmpeg

# Render the best policy
source ~/autorobot_venv/bin/activate
xvfb-run -a python3 -c "
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf
agent_conf, agent_state = PPOJax.load_agent('checkpoints/best/PPOJax_saved.pkl')
env = ImitationFactory.make('UnitreeH1', lafan1_dataset_conf=LAFAN1DatasetConf(['walk1_subject1']), headless=True)
PPOJax.play_policy_mujoco(env, agent_conf, agent_state, n_steps=500, render=False, record=True)
"
# Video saved to LocoMuJoCo_recordings/*/recording.mp4
```

## Changing the Robot

To train on Unitree G1 instead of H1:

1. In `train_loco.py`, change `"MjxUnitreeH1"` to `"MjxUnitreeG1"`
2. In `config.yaml`, change `name: "UnitreeH1"` to `name: "UnitreeG1"`
3. Reset `reward_weights.json` to defaults (the optimal weights may differ)
4. Restart the agent

LocoMuJoCo supports: UnitreeH1, UnitreeH1v2, UnitreeG1, Atlas, Talos, and more.

## Changing the LLM

Edit `config.yaml`:

```yaml
llm:
  model: "deepseek-ai/DeepSeek-V3-0324"  # current
  # Alternatives on Nebius Token Factory:
  # model: "Qwen/Qwen3-Coder-480B-A35B-Instruct"
  # model: "meta-llama/Llama-3.3-70B-Instruct"
  # model: "moonshotai/Kimi-K2-Instruct"
```

Run `python3 scripts/test_nebius_api.py` to verify the model works.

## Changing the Mocap Clip

LAFAN1 has multiple walking clips. Edit `config.yaml`:

```yaml
env:
  mocap: "walk1_subject1"  # current
  # Try: "walk1_subject2", "walk1_subject3", etc.
```

## Common Issues

### GPU out of memory
```
jax.errors.JaxRuntimeError: INTERNAL: no supported devices found for platform CUDA
```
Orphaned GPU processes. Fix:
```bash
pkill -9 -f python3
sleep 3
nvidia-smi  # should show 0 MiB used
```

### API key not working (401 error)
The `.env` file may have been overwritten by `git pull`. Re-set it:
```bash
echo 'NEBIUS_API_KEY=<your-key>' > .env
```

### Training times out
JIT compilation on first run takes 30-60s. The timeout is set to `time_budget + 120s` in `agent.py`. If still timing out, increase the buffer.

### Python output buffering
Always use `PYTHONUNBUFFERED=1` or `python3 -u` when running in the background, otherwise the log file stays empty.

### Git branch conflicts
The agent creates an `experiments` branch. If you need to pull updates from main:
```bash
pkill -f agent.py
git stash
git checkout main
git pull origin main
git branch -D experiments
# Restart agent
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    NEBIUS CLOUD                          │
│                                                         │
│  ┌─────────────────┐         ┌───────────────────────┐ │
│  │  Token Factory   │         │   GPU VM (H200)       │ │
│  │  DeepSeek-V3     │◄───────►│                       │ │
│  │  (LLM inference) │  API    │   agent.py            │ │
│  └─────────────────┘         │     ↓                  │ │
│                              │   train_loco.py        │ │
│                              │     ↓                  │ │
│                              │   LocoMuJoCo + JAX/MJX │ │
│                              │   2048 parallel envs   │ │
│                              │   Unitree H1 + LAFAN1  │ │
│                              └───────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Results So Far

| Metric | Value |
|--------|-------|
| Best score | 117.2 (from 55.2 baseline, +112%) |
| Improvements found | 7 |
| Total experiments | 26+ |
| Training throughput | 222,000 fps on H200 |
| Time per experiment | ~3 minutes |
| GPU memory per experiment | ~108 GB |
