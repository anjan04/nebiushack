# Deploying to a Real Unitree G1

## The Problem: We Trained on H1, Want to Deploy on G1

| | Unitree H1 | Unitree G1 |
|--|-----------|-----------|
| Height | 180 cm | 132 cm |
| Weight | 47 kg | 35 kg |
| Total DOF | 19 | 23-29 |
| Legs | 5 DOF/leg (1 ankle) | 6 DOF/leg (2 ankle) |
| Arms | 4 DOF/arm (no wrist) | 5-7 DOF/arm (with wrist) |
| Waist | 1 joint | 1-3 joints |
| Price | ~$90,000 | ~$16,000-35,000 |

**Direct H1→G1 policy transfer is not feasible.** The joint topologies, link lengths, mass distributions, and dynamics are all different. The action space is a different size (19 vs 23-29).

## Recommended Path

### Step 1: Retrain on G1 in LocoMuJoCo

LocoMuJoCo already has a `UnitreeG1` environment with LAFAN1 data pre-retargeted. Just change the config:

```yaml
# config.yaml
env:
  name: "UnitreeG1"   # was UnitreeH1
  mocap: "walk1_subject1"
```

And in `train_loco.py`, change `"MjxUnitreeH1"` to `"MjxUnitreeG1"`.

Training takes ~30 minutes on a single GPU. The autoresearch loop works identically — the LLM optimizes the same 14 MimicReward weight parameters.

### Step 2: Export Policy as NumPy Weights

The trained policy is a small MLP (512→256 hidden layers, tanh activation, ~175K parameters). Export is trivial:

```python
from loco_mujoco.algorithms import PPOJax
import numpy as np

agent_conf, agent_state = PPOJax.load_agent("checkpoints/best/PPOJax_saved.pkl")
params = agent_state.train_state.params['params']

# Extract actor weights
W0 = np.array(params['FullyConnectedNet_0']['Dense_0']['kernel'])
b0 = np.array(params['FullyConnectedNet_0']['Dense_0']['bias'])
W1 = np.array(params['FullyConnectedNet_0']['Dense_1']['kernel'])
b1 = np.array(params['FullyConnectedNet_0']['Dense_1']['bias'])
W2 = np.array(params['FullyConnectedNet_0']['Dense_2']['kernel'])
b2 = np.array(params['FullyConnectedNet_0']['Dense_2']['bias'])

# Extract normalization stats
obs_mean = np.array(params['RunningMeanStd_0']['mean'])
obs_var = np.array(params['RunningMeanStd_0']['var'])

np.savez("policy_weights.npz", W0=W0, b0=b0, W1=W1, b1=b1, W2=W2, b2=b2,
         obs_mean=obs_mean, obs_var=obs_var)
```

### Step 3: Inference on the Robot (Only Needs NumPy)

The entire policy inference is 6 lines:

```python
import numpy as np

data = np.load("policy_weights.npz")
def get_action(obs):
    x = (obs - data['obs_mean']) / np.sqrt(data['obs_var'] + 1e-8)
    x = np.tanh(x @ data['W0'] + data['b0'])
    x = np.tanh(x @ data['W1'] + data['b1'])
    return x @ data['W2'] + data['b2']
```

Performance: **20,000-100,000 Hz** on any CPU. The robot needs 500 Hz. No ML framework needed.

### Step 4: Deploy via Unitree SDK

The G1 EDU has an onboard **NVIDIA Jetson Orin NX** (16GB). Use `unitree_sdk2_python`:

```bash
pip install unitree-sdk2py
```

Control loop at 500 Hz:
```python
# Pseudocode — runs on the robot's Jetson at 500 Hz
while running:
    # 1. Read robot state
    state = read_lowstate()  # joint positions, velocities, IMU

    # 2. Construct observation (same format as simulation)
    obs = build_obs(state)  # angular vel, gravity, joint pos/vel, commands

    # 3. Run policy (< 0.1ms)
    action = get_action(obs)

    # 4. Send joint commands
    for i, joint in enumerate(G1_JOINTS):
        cmd.q = default_pos[i] + action[i] * action_scale
        cmd.kp = PD_GAINS[i][0]  # e.g., 100 for hip, 150 for knee
        cmd.kd = PD_GAINS[i][1]  # e.g., 2 for hip, 4 for knee
    publish_lowcmd(cmd)

    sleep(0.002)  # 500 Hz
```

### Step 5: Safety

Before running on the real robot:

1. **Validate in MuJoCo first** — use `render_policy.py` to visually confirm the policy walks
2. **Use the state machine** — start in ZERO_TORQUE → move to DEFAULT_POSITION → then MOTION_CONTROL
3. **Add a kill switch** — hardware e-stop or software watchdog
4. **Start with low PD gains** — reduce kp/kd by 50% for the first real-world test
5. **Use a gantry or harness** — suspend the robot so it can't fall during initial tests

## G1 Joint Mapping (29 DOF)

| Index | Joint | PD Gains (kp/kd) |
|-------|-------|-------------------|
| 0 | LEFT_HIP_PITCH | 100 / 2 |
| 1 | LEFT_HIP_ROLL | 100 / 2 |
| 2 | LEFT_HIP_YAW | 100 / 2 |
| 3 | LEFT_KNEE | 150 / 4 |
| 4 | LEFT_ANKLE_PITCH | 40 / 2 |
| 5 | LEFT_ANKLE_ROLL | 40 / 2 |
| 6-11 | RIGHT LEG (same) | same |
| 12 | WAIST_YAW | 250 / 5 |
| 13-14 | WAIST_ROLL/PITCH | locked on 29-DOF |
| 15-21 | LEFT ARM (7 DOF) | 50 / 1 |
| 22-28 | RIGHT ARM (7 DOF) | 50 / 1 |

## Alternative Export Paths

| Method | Robot Dependencies | Speed | When to Use |
|--------|-------------------|-------|-------------|
| **NumPy .npz** | numpy only | 20K-100K Hz | Default — simplest |
| ONNX | onnxruntime | 10K-50K Hz | If robot stack uses ONNX |
| PyTorch | torch | 10K-50K Hz | If robot stack uses PyTorch |
| TensorRT | tensorrt (Jetson) | 50K+ Hz | Maximum performance |

## Existing End-to-End Pipelines

| Project | What It Does | Link |
|---------|-------------|------|
| **unitree_rl_gym** | Train + deploy G1 locomotion (Isaac Gym) | github.com/unitreerobotics/unitree_rl_gym |
| **GEAR-SONIC** | NVIDIA's whole-body control (inference only) | github.com/NVlabs/GR00T-WholeBodyControl |
| **H-Zero** | Cross-embodiment H1↔G1 transfer | arxiv.org/abs/2512.00971 |
| **unitree_mujoco** | MuJoCo sim validation before real deploy | github.com/unitreerobotics/unitree_mujoco |

## What We'd Do Next

1. Switch training to `UnitreeG1` in LocoMuJoCo (1 line config change)
2. Run the autoresearch loop for 20+ experiments on G1
3. Export best checkpoint to `.npz`
4. Validate in MuJoCo with `render_policy.py`
5. Flash JetPack 6 on the G1's Jetson Orin
6. Deploy the numpy inference loop via `unitree_sdk2_python`
7. Test with gantry support, low PD gains
8. Iterate: sim → real → adjust domain randomization → retrain
