#!/usr/bin/env python3
"""Test Nebius Token Factory API connection (Task T0.5).

Verifies both primary and fallback models, tests reward-function generation,
and reports latency for each call.
"""

import os
import sys
import time

import yaml
from openai import OpenAI


def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")
    if not os.path.isfile(path):
        print(f"FAIL: config file not found: {path}")
        sys.exit(1)
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"FAIL: could not parse config file: {e}")
        sys.exit(1)
    if not isinstance(cfg, dict) or "llm" not in cfg:
        print("FAIL: config.yaml is missing the required 'llm' section")
        sys.exit(1)
    llm = cfg["llm"]
    for key in ("base_url", "api_key_env", "model", "fallback_model"):
        if key not in llm:
            print(f"FAIL: config.yaml llm section is missing required key '{key}'")
            sys.exit(1)
    return cfg


def make_client(cfg):
    api_key = os.environ.get(cfg["llm"]["api_key_env"])
    if not api_key:
        print(f"FAIL: env var {cfg['llm']['api_key_env']} is not set")
        sys.exit(1)
    return OpenAI(base_url=cfg["llm"]["base_url"], api_key=api_key)


def timed_chat(client, model, messages, temperature=0.7, max_tokens=1024):
    """Send a chat completion and return (response_text, latency_seconds)."""
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.time() - t0
    text = resp.choices[0].message.content
    return text, latency


def test_model(client, model, label):
    """Basic connectivity test: ask for a one-liner Python function."""
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function that returns the factorial of n. Reply with only the code."},
    ]
    try:
        text, latency = timed_chat(client, model, messages)
        ok = "def " in text and "factorial" in text.lower()
        status = "PASS" if ok else "FAIL (unexpected response)"
        print(f"  [{status}] {label} ({model}) — {latency:.2f}s")
        if not ok:
            print(f"    Response preview: {text[:200]}")
        return ok
    except Exception as e:
        print(f"  [FAIL] {label} ({model}) — {e}")
        return False


def test_reward_generation(client, model):
    """Ask the LLM to write a reward function and validate the output."""
    prompt = (
        "Write a reward function for a quadruped robot that rewards forward velocity. "
        "The function signature must be:\n\n"
        "  def compute_reward(obs: dict) -> tuple[torch.Tensor, dict]\n\n"
        "obs contains key 'root_lin_vel' (shape [N,3]) where index 0 is forward velocity. "
        "Return (reward_tensor, components_dict). Use torch only. Reply with only the code."
    )
    messages = [
        {"role": "system", "content": "You are an expert reinforcement-learning engineer."},
        {"role": "user", "content": prompt},
    ]
    try:
        text, latency = timed_chat(client, model, messages, max_tokens=2048)
        has_signature = "compute_reward" in text
        has_def = "def " in text
        has_torch = "torch" in text
        ok = has_signature and has_def and has_torch
        status = "PASS" if ok else "FAIL"
        detail = []
        if not has_signature:
            detail.append("missing compute_reward")
        if not has_def:
            detail.append("missing def")
        if not has_torch:
            detail.append("missing torch usage")
        extra = f" ({', '.join(detail)})" if detail else ""
        print(f"  [{status}] Reward-function generation — {latency:.2f}s{extra}")
        if not ok:
            print(f"    Response preview: {text[:300]}")
        return ok
    except Exception as e:
        print(f"  [FAIL] Reward-function generation — {e}")
        return False


def main():
    cfg = load_config()
    client = make_client(cfg)

    results = []
    print("Nebius Token Factory API — connection tests\n")

    # Test 1: primary model
    print("1. Primary model:")
    results.append(test_model(client, cfg["llm"]["model"], "Primary"))

    # Test 2: fallback model
    print("2. Fallback model:")
    results.append(test_model(client, cfg["llm"]["fallback_model"], "Fallback"))

    # Test 3: reward-function generation (using primary model)
    print("3. Reward-function generation (primary model):")
    results.append(test_reward_generation(client, cfg["llm"]["model"]))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\nResults: {passed}/{total} passed")

    if all(results):
        print("All tests PASSED.")
        sys.exit(0)
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
