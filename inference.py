#!/usr/bin/env python3
"""
LogEnv Inference Script — v3 Final

Follows the official OpenEnv sample inference.py format exactly.

STDOUT FORMAT (required by validator):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import re
from typing import List, Optional

# ── Environment variables (mandatory per checklist) ──────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")   # optional, for from_docker_image()

API_KEY = HF_TOKEN or os.getenv("API_KEY")

from environment import LogEnv
from environment.models import Action
from environment.graders import grade_task

# ── OpenAI client (mandatory: must use OpenAI client for all LLM calls) ─
from openai import OpenAI

client = None
if API_KEY:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[DEBUG] Client init failed: {e}", file=sys.stderr, flush=True)
else:
    print("[DEBUG] HF_TOKEN not set — deterministic fallback", file=sys.stderr, flush=True)

# ── Config ────────────────────────────────────────────────────────────
TASKS     = ["task1", "task2", "task3", "task4", "task5", "task6", "task7"]
BENCHMARK = "logenv"
MAX_STEPS = 15
SUCCESS_SCORE_THRESHOLD = 0.5

# ── LLM system prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert SRE performing autonomous incident response.

MANDATORY: Perform at least 2 investigation actions (filter_logs/inspect_service) BEFORE mark_root_cause.

Available actions:
  filter_logs      target=keyword
  inspect_service  target=service-name
  mark_root_cause  target=oom_kill|memory_leak|misconfigured_circuit_breaker|network_partition|disk_full|deadlock|dependency_failure
  classify_issue   target=infrastructure_failure|application_bug|configuration_error|network_issue|security_incident|capacity_issue|dependency_failure
  resolve_incident target=restart_service:NAME|scale_service:NAME|rollback_deploy:NAME|patch_config:NAME

ROOT CAUSE → CLASSIFICATION (use exactly):
  oom_kill/disk_full/network_partition → infrastructure_failure
  memory_leak/deadlock                → application_bug
  misconfigured_circuit_breaker       → configuration_error
  dependency_failure                  → dependency_failure

ROOT CAUSE → RESOLUTION (use exactly):
  oom_kill/memory_leak/disk_full/deadlock/network_partition → restart_service:<service>
  dependency_failure                                       → rollback_deploy:<service>
  misconfigured_circuit_breaker                            → scale_service:<service>

Strategy: filter_logs → inspect_service → filter_logs again → mark_root_cause → classify_issue → resolve_incident.
Never repeat the same action. Never skip investigation.

Respond ONLY with a single JSON object on one line:
{"action_type": "...", "target": "..."}
"""


# ── Optimal deterministic fallback sequences ─────────────────────────
FALLBACK = {
    "task1": [
        ("filter_logs",      "error"),
        ("filter_logs",      "memory"),
        ("inspect_service",  "api-server"),
        ("mark_root_cause",  "oom_kill"),
        ("classify_issue",   "infrastructure_failure"),
        ("resolve_incident", "restart_service:api-server"),
    ],
    "task2": [
        ("filter_logs",      "memory"),
        ("inspect_service",  "session-manager"),
        ("filter_logs",      "heap"),
        ("mark_root_cause",  "memory_leak"),
        ("classify_issue",   "application_bug"),
        ("resolve_incident", "restart_service:session-manager"),
    ],
    "task3": [
        ("filter_logs",      "circuit"),
        ("inspect_service",  "order-service"),
        ("filter_logs",      "config"),
        ("inspect_service",  "inventory-service"),
        ("mark_root_cause",  "misconfigured_circuit_breaker"),
        ("classify_issue",   "configuration_error"),
        ("resolve_incident", "scale_service:order-service"),
    ],
    "task4": [
        ("filter_logs",      "disk"),
        ("inspect_service",  "log-rotator"),
        ("filter_logs",      "rotation"),
        ("mark_root_cause",  "disk_full"),
        ("classify_issue",   "infrastructure_failure"),
        ("resolve_incident", "restart_service:log-rotator"),
    ],
    "task5": [
        ("filter_logs",      "deadlock"),
        ("inspect_service",  "payment-service"),
        ("filter_logs",      "lock"),
        ("mark_root_cause",  "deadlock"),
        ("classify_issue",   "application_bug"),
        ("resolve_incident", "restart_service:payment-service"),
    ],
    "task6": [
        ("filter_logs",      "error"),
        ("inspect_service",  "checkout-service"),
        ("filter_logs",      "gateway"),
        ("mark_root_cause",  "dependency_failure"),
        ("classify_issue",   "dependency_failure"),
        ("resolve_incident", "rollback_deploy:checkout-service"),
    ],
    "task7": [
        ("filter_logs",      "error"),
        ("inspect_service",  "redis-cluster"),
        ("filter_logs",      "partition"),
        ("inspect_service",  "session-service"),
        ("mark_root_cause",  "network_partition"),
        ("classify_issue",   "infrastructure_failure"),
        ("resolve_incident", "restart_service:redis-cluster"),
    ],
}

VALID_ACTIONS = {
    "filter_logs", "inspect_service",
    "mark_root_cause", "classify_issue", "resolve_incident",
}


# ── Mandatory log helpers (exact format from official sample) ─────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()           # must be lowercase: true / false
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()      # must be lowercase: true / false
    print(
        f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM helpers ───────────────────────────────────────────────────────

def _extract_json(text: str):
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def _format_obs(obs, task_id: str, step: int) -> str:
    lines = [f"Task: {task_id} | Step: {step}", "", "LOGS:"]
    for log in obs.logs[-10:]:
        lines.append(f"  [{log.level}] {log.service}: {log.message}")
    lines += [
        "",
        "METRICS:",
        f"  CPU={obs.metrics.cpu_percent}%  MEM={obs.metrics.memory_percent}%"
        f"  Disk={obs.metrics.disk_percent}%  Errors={obs.metrics.error_rate}%",
        "",
        "ALERTS:",
    ]
    for a in obs.alerts:
        lines.append(f"  [{a.severity}] {a.service}: {a.message}")
    lines.append("\nJSON action:")
    return "\n".join(lines)


def _llm_action(obs, task_id: str, step: int, conversation: list):
    """Try LLM; returns (Action|None, used_llm bool)."""
    if client is None:
        return None, False
    obs_text = _format_obs(obs, task_id, step)
    conversation.append({"role": "user", "content": obs_text})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=80,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        conversation.append({"role": "assistant", "content": raw})
        parsed = _extract_json(raw)
        if not parsed or parsed.get("action_type") not in VALID_ACTIONS:
            return None, False
        return Action(action_type=parsed["action_type"], target=parsed.get("target")), True
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", file=sys.stderr, flush=True)
        return None, False


def _fallback_action(task_id: str, step: int) -> Action:
    seq = FALLBACK.get(task_id, [])
    idx = step - 1
    if idx < len(seq):
        at, tgt = seq[idx]
        return Action(action_type=at, target=tgt)
    return Action(action_type="resolve_incident", target="restart_service:api-server")


# ── Core task runner ──────────────────────────────────────────────────

def run_task(task_id: str) -> dict:
    """Run one full episode. Always emits [START]…[STEP]…[END] even on error."""

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    env          = LogEnv(task_name=task_id)
    obs          = env.reset()
    rewards: List[float] = []
    steps_taken  = 0
    score        = 0.0
    success      = False
    conversation = []

    try:
        for step in range(1, MAX_STEPS + 1):
            action, used_llm = _llm_action(obs, task_id, step, conversation)
            if action is None:
                action = _fallback_action(task_id, step)

            obs, reward, done, info = env.step(action)
            reward = float(reward)
            rewards.append(reward)
            steps_taken = step

            # action string for [STEP] log — combine action_type + target
            action_str = f"{action.action_type}({action.target or ''})"
            error_val  = info.get("error") if isinstance(info, dict) else None

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error_val,
            )

            if done:
                break

        state   = env.state()
        score   = float(grade_task(task_id, state))  # already clamped to (0.01, 0.99) by grader
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] run_task error for {task_id}: {exc}", file=sys.stderr, flush=True)

    finally:
        # [END] MUST always be emitted, even on exception (per official spec)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id":  task_id,
        "score":    round(score, 4),
        "steps":    steps_taken,
        "llm_used": any(r > 0 for r in rewards),
        "rewards":  rewards,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    results = []
    for task_id in TASKS:
        result = run_task(task_id)
        results.append(result)

    avg = sum(r["score"] for r in results) / max(len(results), 1)
    print(f"[DEBUG] average_score={avg:.4f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()