# AutoRobot — Implementation Task List

## Phase 0: Environment Setup (Hour 1)

- [ ] **T0.1** Provision Nebius GPU VM (1x H100, Ubuntu, pre-installed NVIDIA drivers)
- [ ] **T0.2** Install Isaac Gym Preview 4 on Nebius VM
  - Download tarball from developer.nvidia.com
  - `conda create -n autorobot python=3.8 && conda activate autorobot`
  - `cd isaacgym/python && pip install -e .`
  - Verify: `python examples/joint_monkey.py --headless`
- [ ] **T0.3** Install rl_games and isaacgymenvs from Eureka repo
  - `git clone https://github.com/eureka-research/Eureka.git /tmp/eureka`
  - `cd /tmp/eureka/isaacgymenvs && pip install -e .`
  - `cd /tmp/eureka/rl_games && pip install -e .`
- [ ] **T0.4** Verify Ant training works
  - `cd /tmp/eureka/isaacgymenvs && python train.py task=Ant headless=True max_iterations=500`
  - Confirm GPU utilization, training FPS, reward curve
- [ ] **T0.5** Get Nebius Token Factory API key, test LLM call
  - Sign up at tokenfactory.nebius.com
  - Test: `curl https://api.studio.nebius.ai/v1/chat/completions -H "Authorization: Bearer $NEBIUS_API_KEY" ...`
- [ ] **T0.6** Clone nebiushack repo on the VM

## Phase 1: Core Files (Hour 2)

- [ ] **T1.1** Write `prepare.py`
  - Import Isaac Gym Ant environment
  - Define observation keys the agent can reference (root_lin_vel, joint_pos, etc.)
  - Write `evaluate(reward_fn, num_episodes=50)` → returns metrics dict
  - Write `get_observation_docs()` → returns string describing all obs fields (fed to LLM)
  - Keep it under 200 lines
- [ ] **T1.2** Write initial `reward.py`
  - Simple baseline: `reward = forward_velocity`
  - Must follow the function signature: `compute_reward(obs) -> (tensor, dict)`
- [ ] **T1.3** Write `train.py`
  - Load Ant env from Isaac Gym
  - Import reward function from `reward.py`
  - Run PPO (rl_games) with fixed 3-minute wall clock timeout
  - Print metrics to stdout in parseable format:
    ```
    METRICS: primary_score=1.23 episode_return=45.6 episode_length=200 fps=50000
    COMPONENTS: forward_vel=1.5 energy=-0.3 alive=0.03
    ```
  - Exit cleanly after timeout
- [ ] **T1.4** Write `program.md`
  - System prompt for the agent
  - Explain the task: "maximize forward velocity of a quadruped (Ant)"
  - Explain the observation space (reference prepare.py docs)
  - Explain the reward function format
  - Constraints: use torch tensors, keep it under 50 lines, return components dict
  - Tips: use `torch.exp(-x/temp)` for smooth rewards, penalize energy, reward survival
- [ ] **T1.5** Write `config.yaml`
  - LLM: model name, base_url, api_key env var, temperature, max_tokens
  - Training: time_budget_seconds=180, num_envs=4096, device=cuda:0
  - Agent: max_iterations=100, retry_on_error=3
  - Task: env_name=Ant, primary_metric=primary_score
- [ ] **T1.6** Manual test: run `train.py` with baseline `reward.py`, confirm metrics output

## Phase 2: Agent Orchestration (Hour 3)

- [ ] **T2.1** Write `agent.py` — the main loop
  - Parse `config.yaml`
  - Initialize OpenAI client with Nebius base_url
  - Main loop:
    1. Build prompt (program.md + env docs + current reward + last metrics + git log)
    2. Call LLM → extract reward function code from response
    3. Write to `reward.py`
    4. Run `python train.py` as subprocess with timeout
    5. Parse metrics from stdout
    6. Compare to best: keep (git commit) or revert (git checkout reward.py)
    7. Log iteration results
  - Error handling: if train.py crashes, feed traceback to LLM, retry up to 3 times
  - Stagnation detection: if same primary_score for 5 iterations, inject "try radically different approach"
- [ ] **T2.2** Implement code extraction from LLM response
  - Parse markdown code blocks (```python ... ```)
  - Validate: must define `compute_reward` function
  - Validate: function signature matches expected format
  - Validate: no imports beyond torch and math
- [ ] **T2.3** Implement metrics parsing from train.py stdout
  - Regex parse `METRICS:` and `COMPONENTS:` lines
  - Handle training failures (timeout, crash, NaN)
- [ ] **T2.4** Implement git integration
  - `git add reward.py && git commit -m "exp-{N}: score={X} (improvement)"` for keeps
  - `git checkout reward.py` for reverts, then commit with "reverted" message
  - Tag best result: `git tag best-v{N}`
- [ ] **T2.5** Test full loop: run 3 iterations end-to-end

## Phase 3: Let It Run (Hours 4-5)

- [ ] **T3.1** Start agent.py in a tmux session on the Nebius VM
- [ ] **T3.2** Monitor progress: `watch -n 30 'git log --oneline | head -20'`
- [ ] **T3.3** Log all LLM prompts/responses to `logs/` for debugging
- [ ] **T3.4** If agent gets stuck, adjust `program.md` hints and restart

## Phase 4: Demo & Visualization (Hour 5-6)

- [ ] **T4.1** Build `visualize.py` — parse git log + metrics into a chart
  - X axis: experiment number
  - Y axis: primary_score
  - Color: green for kept, red for reverted
  - Output: `results/score_evolution.png`
- [ ] **T4.2** Record sim videos of best vs worst reward
  - Run `train.py` with `headless=False` or video recording for top-3 and bottom-3 rewards
  - Save as `results/best.mp4`, `results/worst.mp4`
- [ ] **T4.3** Generate summary stats
  - Total experiments run
  - Num improvements found
  - Best score vs baseline
  - Most interesting reward function discovered
- [ ] **T4.4** Build pitch deck (3-5 slides)
  - Slide 1: The problem — "hand-crafting reward functions is tedious and brittle"
  - Slide 2: The approach — autoresearch loop diagram
  - Slide 3: Results — score evolution chart + videos
  - Slide 4: Nebius stack — Token Factory for LLM, H100 for training
  - Slide 5: What's next — harder tasks, sim-to-real, multi-robot

## Phase 5: Polish & Submit (Hour 6)

- [ ] **T5.1** Clean up git history (squash setup commits, keep experiment commits)
- [ ] **T5.2** Update README with actual results
- [ ] **T5.3** Push to GitHub
- [ ] **T5.4** Rehearse 3-minute pitch
- [ ] **T5.5** Prepare backup: if live demo fails, have pre-recorded video ready

---

## Dependencies

```
# Python packages
torch>=2.0
isaacgym  # from Preview 4 tarball
isaacgymenvs  # from Eureka repo
rl_games  # from Eureka repo
openai>=1.0
pyyaml
matplotlib
gitpython
```

## Critical Path

```
T0.2 (Isaac Gym) → T0.4 (verify Ant) → T1.1 (prepare.py) → T1.3 (train.py) → T2.1 (agent.py) → T3.1 (run)
     ↘                                    T1.2 (reward.py) ↗         ↗
      T0.5 (Nebius API)          →        T1.4 (program.md) → T2.2 (code extraction)
```

The longest pole is Isaac Gym installation (T0.2-T0.4). Everything else can be developed locally and deployed.
