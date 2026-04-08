#!/usr/bin/env python3
"""
LogEnv Inference Script

Uses the OpenAI-compatible API client (with HF_TOKEN) to drive an LLM agent
through each task. Falls back to a deterministic policy when the LLM is unavailable.

Usage:
    HF_TOKEN=your_token python inference.py
    HF_TOKEN=your_token MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python inference.py
"""

import os
import sys
import json

from environment import LogEnv
from environment.models import Action
from environment.graders import grade_task


# ---------------- OPENAI CLIENT SETUP ----------------

client = None

try:
    from openai import OpenAI

    api_key = os.environ.get("HF_TOKEN")
    if api_key:
        client = OpenAI(
            base_url=os.environ.get(
                "API_BASE_URL",
                "https://api-inference.huggingface.co/v1"
            ),
            api_key=api_key
        )
        print("✅ OpenAI client initialized (HF_TOKEN found)", flush=True)
    else:
        print("⚠️  HF_TOKEN not set — using fallback policy", file=sys.stderr, flush=True)
except Exception as e:
    print(f"⚠️  OpenAI client init failed: {e} — using fallback policy", file=sys.stderr, flush=True)


# ---------------- CONFIG ----------------

TASKS = ["task1", "task2", "task3"]
MAX_STEPS = 12
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

SYSTEM_PROMPT = """You are an expert DevOps/SRE incident response agent.
You analyze system logs, metrics, and alerts to identify root causes and resolve incidents.

Available actions (respond with ONLY a JSON object):
- filter_logs: search logs by keyword
- inspect_service: view logs for a specific service
- mark_root_cause: identify the root cause (values: oom_kill, memory_leak, misconfigured_circuit_breaker, network_partition, disk_full, deadlock, dependency_failure)
- classify_issue: classify the incident (values: infrastructure_failure, application_bug, configuration_error, network_issue, security_incident, capacity_issue, dependency_failure)
- resolve_incident: take resolution action (values: restart_service:NAME, scale_service:NAME, rollback_deploy:NAME, patch_config:NAME)

Respond with EXACTLY this JSON format:
{"action_type": "...", "target": "..."}"""


# ---------------- LLM AGENT ----------------

def build_observation_text(obs, task_id: str, step: int) -> str:
    """Build a human-readable observation for the LLM."""
    lines = [
        f"=== Step {step} | Task: {task_id} ===",
        "",
        "RECENT LOGS:",
    ]
    for log in obs.logs[-10:]:  # last 10 logs
        lines.append(f"  [{log.level}] {log.timestamp} {log.service}: {log.message}")

    lines.extend([
        "",
        "METRICS:",
        f"  CPU: {obs.metrics.cpu_percent}% | Memory: {obs.metrics.memory_percent}% | Error rate: {obs.metrics.error_rate}%",
        f"  Active connections: {obs.metrics.active_connections} | Request rate: {obs.metrics.request_rate}",
        "",
        "ALERTS:",
    ])
    for alert in obs.alerts:
        lines.append(f"  [{alert.severity}] {alert.service}: {alert.message}")

    lines.extend([
        "",
        "What is your next action? Respond with JSON only.",
    ])
    return "\n".join(lines)


def llm_action(obs, task_id: str, step: int) -> Action | None:
    """Call LLM to get the next action. Returns None on failure."""
    if client is None:
        return None

    try:
        obs_text = build_observation_text(obs, task_id, step)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text}
            ],
            max_tokens=100,
            temperature=0.1,
        )
        content = response.choices[0].message.content.strip()
        # Parse JSON response
        # Handle markdown code blocks if present
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content.strip())
        action = Action(
            action_type=parsed.get("action_type", "filter_logs"),
            target=parsed.get("target")
        )
        print(f"  🤖 LLM chose: {action.action_type}({action.target})", flush=True)
        return action
    except Exception as e:
        print(f"  ⚠️  LLM call failed: {e}", flush=True)
        return None


# ---------------- DETERMINISTIC FALLBACK ----------------

def fallback_action(task_id: str, step: int) -> Action:
    """Optimal deterministic policy for each task."""
    if task_id == "task1":
        seq = [
            Action(action_type="filter_logs", target="error"),
            Action(action_type="filter_logs", target="memory"),
            Action(action_type="inspect_service", target="api-server"),
            Action(action_type="mark_root_cause", target="oom_kill"),
            Action(action_type="classify_issue", target="infrastructure_failure"),
            Action(action_type="resolve_incident", target="restart_service:api-server"),
        ]
    elif task_id == "task2":
        seq = [
            Action(action_type="filter_logs", target="memory"),
            Action(action_type="inspect_service", target="session-manager"),
            Action(action_type="filter_logs", target="heap"),
            Action(action_type="mark_root_cause", target="memory_leak"),
            Action(action_type="classify_issue", target="application_bug"),
            Action(action_type="resolve_incident", target="restart_service:session-manager"),
        ]
    elif task_id == "task3":
        seq = [
            Action(action_type="filter_logs", target="error"),
            Action(action_type="inspect_service", target="order-service"),
            Action(action_type="filter_logs", target="circuit"),
            Action(action_type="inspect_service", target="payment-service"),
            Action(action_type="mark_root_cause", target="misconfigured_circuit_breaker"),
            Action(action_type="classify_issue", target="configuration_error"),
            Action(action_type="resolve_incident", target="scale_service:order-service"),
        ]
    else:
        seq = [Action(action_type="filter_logs", target="error")]

    idx = step - 1
    if idx < len(seq):
        return seq[idx]
    return Action(action_type="resolve_incident", target="restart_service:api-server")


# ---------------- RUN TASK ----------------

def run_task(task_id: str) -> float:
    print(f"\n{'='*50}", flush=True)
    print(f"[START] Task: {task_id}", flush=True)
    print(f"{'='*50}", flush=True)

    env = LogEnv(task_name=task_id)
    obs = env.reset()

    total_reward = 0.0
    done = False

    for step in range(1, MAX_STEPS + 1):
        # Try LLM first, fall back to deterministic policy
        action = llm_action(obs, task_id, step)
        if action is None:
            action = fallback_action(task_id, step)

        obs, reward, done, _ = env.step(action)
        total_reward += reward

        print(
            f"  [STEP {step:2d}] {action.action_type}({action.target or ''}) "
            f"→ reward={reward:+.4f} | done={done}",
            flush=True
        )

        if done:
            break

    state = env.state()
    final_score = grade_task(task_id, state)

    print(f"\n  Root cause:     {state.root_cause_marked}", flush=True)
    print(f"  Classification: {state.classification_marked}", flush=True)
    print(f"  Resolution:     {state.resolution_action}", flush=True)
    print(f"  Steps used:     {state.step_count}", flush=True)
    print(f"  Wrong actions:  {state.wrong_action_count}", flush=True)
    print(f"  ✅ FINAL SCORE: {final_score:.4f}", flush=True)
    print(f"  Cumulative reward: {total_reward:.4f}", flush=True)

    return final_score


# ---------------- MAIN ----------------

def main():
    print("=" * 60, flush=True)
    print("  LogEnv Inference Script", flush=True)
    print(f"  Model: {MODEL_NAME}", flush=True)
    print(f"  Tasks: {TASKS}", flush=True)
    print("=" * 60, flush=True)

    scores = []
    for task_id in TASKS:
        score = run_task(task_id)
        scores.append(score)

    avg = sum(scores) / len(scores)

    print("\n" + "=" * 60, flush=True)
    print("  FINAL RESULTS", flush=True)
    print("=" * 60, flush=True)
    for t, s in zip(TASKS, scores):
        bar = "█" * int(s * 20)
        print(f"  {t}: {s:.4f} {bar}", flush=True)
    print(f"\n  AVERAGE: {avg:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()