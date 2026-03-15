# AutoRobot — Setup Guide

## Prerequisites

- NVIDIA GPU with CUDA 11.x+ (H100, A100, RTX 4090, or RTX 3090)
- Ubuntu 20.04 or 22.04
- Python 3.8 (Isaac Gym requirement)
- ~20 GB disk space
- Nebius account with API key

## Option A: Nebius GPU Cloud (Recommended)

### 1. Provision a VM

```bash
# Install Nebius CLI
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash

# Create a GPU VM (1x H100)
# Use the web console at console.nebius.com:
#   - Platform: gpu-h100-sxm
#   - Preset: 1gpu-16vcpu-200gb
#   - Boot image: Ubuntu 22.04 with GPU drivers
#   - Disk: 100 GB SSD
```

### 2. SSH into the VM

```bash
ssh -i ~/.ssh/nebius_key ubuntu@<VM_IP>
```

### 3. Install Conda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda create -n autorobot python=3.8 -y
conda activate autorobot
```

### 4. Install Isaac Gym Preview 4

```bash
# Download from https://developer.nvidia.com/isaac-gym
# (requires NVIDIA developer account — free)
tar -xvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .

# Verify
python -c "import isaacgym; print('Isaac Gym OK')"
```

### 5. Install Eureka Dependencies

```bash
cd ~
git clone https://github.com/eureka-research/Eureka.git
cd Eureka
pip install -e .
cd isaacgymenvs && pip install -e .
cd ../rl_games && pip install -e .

# Verify Ant training
cd ~/Eureka/isaacgymenvs
python train.py task=Ant headless=True max_iterations=200
```

### 6. Install AutoRobot

```bash
cd ~
git clone https://github.com/anjan04/nebiushack.git
cd nebiushack
pip install openai pyyaml matplotlib gitpython
```

### 7. Configure API Keys

```bash
# Nebius Token Factory
export NEBIUS_API_KEY="your-key-here"

# Verify
python -c "
from openai import OpenAI
import os
client = OpenAI(base_url='https://api.studio.nebius.ai/v1/', api_key=os.environ['NEBIUS_API_KEY'])
r = client.chat.completions.create(model='Qwen/Qwen3-235B-Thinking', messages=[{'role':'user','content':'Say hello'}], max_tokens=10)
print(r.choices[0].message.content)
"
```

## Option B: Local GPU

Same steps as above but skip the Nebius VM provisioning. Ensure your local machine has:
- NVIDIA driver 515+
- CUDA 11.7+
- Vulkan SDK (for Isaac Gym rendering)

## Running

### Manual Test (Single Training Run)

```bash
cd ~/nebiushack
python train.py
# Should output METRICS: primary_score=X.XX ... after 3 minutes
```

### Autonomous Agent Loop

```bash
cd ~/nebiushack
# Start in tmux so it survives SSH disconnects
tmux new -s autorobot
python agent.py

# In another tmux pane, monitor:
watch -n 30 'git log --oneline | head -20'
```

### Visualization

```bash
# After agent has run for a while:
python visualize.py
# Outputs: results/score_evolution.png
```

## Troubleshooting

### Isaac Gym "No GPU found"
```bash
nvidia-smi  # verify GPU is visible
# If using Docker, ensure --gpus all flag
```

### Isaac Gym "Vulkan not found"
```bash
# For headless servers (no display):
python train.py headless=True graphics_device_id=-1
```

### LLM generates invalid reward code
The agent.py loop catches syntax errors and feeds the traceback back to the LLM for self-correction. Check `logs/` for details.

### Training NaN/Inf
Usually means the reward function has unbounded values. The agent should detect this via metrics parsing and ask the LLM to fix it. If persistent, add reward clipping in `prepare.py`.

## GPU Memory Budget

| Component | Memory |
|-----------|--------|
| Isaac Gym (4096 envs) | ~4 GB |
| PPO policy + value networks | ~1 GB |
| rl_games overhead | ~1 GB |
| **Total** | **~6 GB** |

Fits easily on any modern GPU. Scale `num_envs` down to 1024 for GPUs with <12 GB VRAM.
