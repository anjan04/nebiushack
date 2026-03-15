#!/usr/bin/env python3
"""AutoRobot Agent — autonomous reward-engineering loop.

Reads config.yaml and program.md, iteratively asks an LLM for improved
reward functions, trains them via train.py, and keeps or reverts based
on the primary_score metric.  Every decision is recorded as a git commit.

Usage:  python agent.py
"""
from __future__ import annotations
import hashlib, json, os, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parent
CONFIG_PATH = REPO / "config.yaml"
PROGRAM_PATH = REPO / "program.md"
REWARD_PATH = REPO / "reward.py"
ENV_PATH = REPO / ".env"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def git(*args: str) -> str:
    r = subprocess.run(["git"] + list(args), cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[git] WARNING: git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()

def git_commit_reward(msg: str) -> None:
    git("add", "reward.py"); git("commit", "-m", msg)

def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def parse_kv_line(stdout: str, prefix: str) -> dict | None:
    last = None
    for ln in stdout.splitlines():
        if ln.strip().startswith(prefix):
            last = ln
    if not last:
        return None
    d = {}
    for tok in last.split(prefix, 1)[1].strip().split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try: d[k] = float(v)
            except ValueError: d[k] = v
    return d or None

# ── LLM ──────────────────────────────────────────────────────────────────────

def make_client(cfg: dict):
    from openai import OpenAI
    api_key = os.environ.get(cfg["llm"]["api_key_env"], "")
    if not api_key:
        sys.exit(f"ERROR: env var {cfg['llm']['api_key_env']} not set")
    return OpenAI(base_url=cfg["llm"]["base_url"], api_key=api_key)

def call_llm(client, cfg: dict, messages: list[dict]) -> str:
    model = cfg["llm"]["model"]
    for i in range(3):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages,
                temperature=cfg["llm"]["temperature"],
                max_tokens=cfg["llm"]["max_tokens"],
            )
            return r.choices[0].message.content
        except Exception as e:
            wait = 2 ** (i + 1)
            print(f"[llm] attempt {i+1}/3 failed: {e}  (retry in {wait}s)")
            if i == 2:
                fb = cfg["llm"].get("fallback_model")
                if fb and fb != model:
                    print(f"[llm] trying fallback {fb}")
                    try:
                        r = client.chat.completions.create(
                            model=fb, messages=messages,
                            temperature=cfg["llm"]["temperature"],
                            max_tokens=cfg["llm"]["max_tokens"],
                        )
                        return r.choices[0].message.content
                    except Exception as e2:
                        raise RuntimeError(f"LLM failed (fallback): {e2}") from e2
                raise RuntimeError(f"LLM failed after 3 retries: {e}") from e
            time.sleep(wait)
    return ""

def extract_code(response: str) -> str | None:
    # Find all python code blocks (case-insensitive language marker)
    blocks = re.findall(r"```[Pp]ython\s*\n(.*?)```", response, re.DOTALL)
    if not blocks:
        # Fallback: try bare ``` blocks
        blocks = re.findall(r"```\s*\n(.*?)```", response, re.DOTALL)
    if not blocks:
        return None
    # Prefer the block containing compute_reward; fall back to the last block
    for b in blocks:
        if "def compute_reward" in b:
            return b.strip()
    return blocks[-1].strip()

_ALLOWED_IMPORTS = {"torch", "math"}

def validate_code(code: str) -> str | None:
    if "def compute_reward" not in code:
        return "Missing compute_reward function."
    for line in code.splitlines():
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            # Extract the top-level module name(s) being imported
            if s.startswith("from "):
                # "from X import ..." -> check X
                mod = s.split()[1].split(".")[0]
                if mod not in _ALLOWED_IMPORTS:
                    return f"Disallowed import: '{mod}'. Only torch and math are allowed."
            else:
                # "import X, Y, Z" or "import X.sub"
                for tok in s.replace(",", " ").split():
                    if tok in ("import", "as") or tok.startswith("as"):
                        continue
                    mod = tok.split(".")[0]
                    if mod and mod not in _ALLOWED_IMPORTS:
                        return f"Disallowed import: '{mod}'. Only torch and math are allowed."
    return None

# ── training ─────────────────────────────────────────────────────────────────

def run_training(cfg: dict) -> tuple[dict | None, dict | None, str]:
    timeout = cfg["training"]["time_budget_seconds"] + 30
    try:
        r = subprocess.run(
            [sys.executable, str(REPO / "train.py")],
            cwd=REPO, capture_output=True, text=True, timeout=timeout,
        )
        out = r.stdout + "\n" + r.stderr
        if r.returncode != 0:
            return None, None, f"CRASH (rc={r.returncode})\n{out}"
    except subprocess.TimeoutExpired:
        return None, None, f"TIMEOUT after {timeout}s"
    metrics = parse_kv_line(r.stdout, "METRICS:")
    components = parse_kv_line(r.stdout, "COMPONENTS:")
    return metrics, components, r.stdout

# ── prompt ───────────────────────────────────────────────────────────────────

def build_prompt(system: str, reward: str, metrics: dict | None,
                 components: dict | None, error: str | None,
                 stagnation: bool, history: str) -> list[dict]:
    parts: list[str] = []
    parts.append("## Current reward.py\n```python\n" + reward + "\n```")
    if metrics:
        parts.append("## Last metrics\n" +
                      " ".join(f"{k}={v}" for k, v in metrics.items()))
    if components:
        parts.append("## Reward components\n" +
                      " ".join(f"{k}={v}" for k, v in components.items()))
    if error:
        parts.append(f"## Error\nYour reward function caused this error:\n"
                      f"```\n{error}\n```\nFix and rewrite.")
    if stagnation:
        parts.append(
            "## STAGNATION\nScore has not improved for several iterations. "
            "Try a **radically different approach**: different structure, "
            "different obs features, very different coefficients.")
    if history:
        parts.append(f"## Recent experiments\n```\n{history}\n```")
    parts.append("Write an improved reward function. "
                 "Output exactly one ```python block (include imports).")
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(parts)}]

# ── logging ──────────────────────────────────────────────────────────────────

def log_iter(log_dir: Path, n: int, data: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"experiment_{n:04d}.json", "w") as f:
        json.dump(data, f, indent=2, default=str)

# ── main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    load_env(ENV_PATH)
    cfg = load_config(CONFIG_PATH)
    system_prompt = PROGRAM_PATH.read_text()
    log_dir = REPO / cfg["agent"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    (REPO / cfg["agent"]["results_dir"]).mkdir(parents=True, exist_ok=True)

    client = make_client(cfg)
    max_iter       = cfg["agent"]["max_iterations"]
    retry_limit    = cfg["agent"]["retry_on_error"]
    stag_thresh    = cfg["agent"]["stagnation_threshold"]
    prim_metric    = cfg["env"]["primary_metric"]

    # git branch setup
    branch = cfg["git"]["branch"]
    if git("rev-parse", "--abbrev-ref", "HEAD") != branch:
        if branch in git("branch", "--list", branch):
            git("checkout", branch)
        else:
            git("checkout", "-b", branch)
        print(f"[agent] on branch '{branch}'")

    best_score: float | None = None
    best_reward = REWARD_PATH.read_text()
    last_metrics: dict | None = None
    last_components: dict | None = None
    last_error: str | None = None
    seen_hashes: set[str] = set()
    no_improve = 0
    total_improve = 0
    baseline: float | None = None
    tag_ctr = 0

    print("=" * 60)
    print(f"  AutoRobot Agent — up to {max_iter} experiments")
    print("=" * 60)

    for it in range(1, max_iter + 1):
        t0 = datetime.now(timezone.utc)
        print(f"\n{'─'*50}\n  Experiment {it}/{max_iter}\n{'─'*50}")

        cur_reward = REWARD_PATH.read_text()
        hist = git("log", "--oneline", "-5")
        stagnation = no_improve >= stag_thresh

        rh = file_hash(cur_reward)
        if rh in seen_hashes and not stagnation:
            stagnation = True
        seen_hashes.add(rh)

        msgs = build_prompt(system_prompt, cur_reward, last_metrics,
                            last_components, last_error, stagnation, hist)
        last_error = None

        # ── get code from LLM ──
        new_code = None; llm_resp = ""
        for attempt in range(retry_limit):
            print(f"  [llm] attempt {attempt+1}...")
            try:
                llm_resp = call_llm(client, cfg, msgs)
            except RuntimeError as e:
                print(f"  [llm] FATAL: {e}"); continue

            new_code = extract_code(llm_resp)
            if new_code is None:
                msgs += [{"role": "assistant", "content": llm_resp},
                         {"role": "user", "content":
                          "No ```python block found. Resend full function in a code block."}]
                continue
            err = validate_code(new_code)
            if err:
                msgs += [{"role": "assistant", "content": llm_resp},
                         {"role": "user", "content": f"Problem: {err}\nFix and resend."}]
                new_code = None; continue
            break

        if new_code is None:
            print("  [llm] no valid code, skipping.")
            log_iter(log_dir, it, {"iteration": it, "status": "llm_failure",
                                   "timestamp": t0.isoformat(), "llm_response": llm_resp})
            continue

        # ── write & train ──
        backup = REWARD_PATH.read_text()
        REWARD_PATH.write_text(new_code + "\n")
        print("  [write] reward.py updated")

        metrics = components = None; train_out = ""; train_err = None
        for erri in range(retry_limit + 1):
            print(f"  [train] running... (attempt {erri+1})")
            metrics, components, train_out = run_training(cfg)
            if metrics is not None:
                break
            train_err = train_out
            if erri >= retry_limit:
                break
            print(f"  [train] failed, asking LLM to fix...")
            fix_msgs = build_prompt(system_prompt, REWARD_PATH.read_text(),
                                    None, None, train_err[:3000], False, hist)
            try:
                fix_resp = call_llm(client, cfg, fix_msgs)
            except RuntimeError:
                break
            fix_code = extract_code(fix_resp)
            if fix_code and validate_code(fix_code) is None:
                REWARD_PATH.write_text(fix_code + "\n")
                print("  [write] reward.py patched")
            else:
                break

        # ── evaluate ──
        decision = "error"; score = None
        if metrics and prim_metric in metrics:
            score = metrics[prim_metric]
            if baseline is None:
                baseline = score
            if best_score is None or score > best_score:
                decision = "improvement"
                old = best_score or 0.0
                best_score = score
                best_reward = REWARD_PATH.read_text()
                total_improve += 1; no_improve = 0
                print(f"  >>> IMPROVEMENT: {score:.3f} (was {old:.3f})")
                if cfg["git"]["auto_commit"]:
                    git_commit_reward(
                        f"exp-{it}: score={score:.3f} (improvement, was {old:.3f})")
                if cfg["git"]["tag_improvements"]:
                    tag_ctr += 1; git("tag", "-f", f"best-v{tag_ctr}")
            else:
                decision = "reverted"
                REWARD_PATH.write_text(backup); no_improve += 1
                print(f"  <<< REVERTED: {score:.3f} < best {best_score:.3f}")
                if cfg["git"]["auto_commit"]:
                    git_commit_reward(
                        f"exp-{it}: score={score:.3f} (reverted, best={best_score:.3f})")
        else:
            REWARD_PATH.write_text(backup); no_improve += 1
            print("  !!! ERROR: no metrics, reverted")
            if cfg["git"]["auto_commit"]:
                git_commit_reward(f"exp-{it}: error (reverted)")
            err_text = train_err or "No parseable METRICS line."
            last_error = err_text[:3000]  # truncate to avoid filling LLM context

        last_metrics = metrics; last_components = components

        # ── log ──
        log_iter(log_dir, it, {
            "iteration": it, "timestamp": t0.isoformat(),
            "duration_s": (datetime.now(timezone.utc) - t0).total_seconds(),
            "decision": decision, "score": score, "best_score": best_score,
            "metrics": metrics, "components": components,
            "reward_code": REWARD_PATH.read_text(),
            "llm_response": llm_resp, "train_stdout": train_out[:2000],
        })
        print(f"  [log] experiment_{it:04d}.json")
        if best_score is not None:
            print(f"  [status] best={best_score:.3f} | "
                  f"improvements={total_improve} | stagnation={no_improve}")

    # ── summary ──
    print("\n" + "=" * 60 + "\n  AGENT FINISHED\n" + "=" * 60)
    print(f"  Total experiments:  {max_iter}")
    print(f"  Improvements found: {total_improve}")
    print(f"  Best score:         {best_score}")
    if baseline is not None and best_score is not None:
        d = best_score - baseline
        pct = (d / abs(baseline) * 100) if baseline else float("inf")
        print(f"  Baseline:           {baseline:.3f}")
        print(f"  Improvement:        +{d:.3f} ({pct:.1f}%)")
    print(f"  Logs:               {log_dir}")
    print(f"  Git log:\n{git('log', '--oneline', '-20')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
