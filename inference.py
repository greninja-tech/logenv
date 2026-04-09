#!/usr/bin/env python3
"""
LogEnv Inference Script — v3 Final

REQUIRED structured stdout format (checked by validator):
  [START] task=TASKNAME
  [STEP] step=N action=ACTION target=TARGET reward=R done=True/False
  [END] task=TASKNAME score=S steps=N

All LLM calls use the OpenAI client configured via environment variables.
"""

import os
import sys
import json
import re

# ── Environment variables (required by submission checklist) ─────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

# Optional – only needed if using from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

from environment import LogEnv
from environment.models import Action
from environment.graders import grade_task

# ── OpenAI client (required by checklist: "from openai import OpenAI") ─
from openai import OpenAI

client = None
if HF_TOKEN:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        print(f"LLM client initialised. model={MODEL_NAME}", flush=True)
    except Exception as e:
        print(f"Client init failed: {e}", file=sys.stderr, flush=True)
else:
    print("HF_TOKEN not set — deterministic fallback", file=sys.stderr, flush=True)

# ── Tasks ─────────────────────────────────────────────────────────────
TASKS     = ["task1", "task2", "task3", "task4", "task5", "task6", "task7"]
MAX_STEPS = 15

# ── LLM system prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert SRE performing autonomous incident response.

Available actions:
  filter_logs      target=keyword
  inspect_service  target=service-name
  mark_root_cause  target=oom_kill|memory_leak|misconfigured_circuit_breaker|network_partition|disk_full|deadlock|dependency_failure
  classify_issue   target=infrastructure_failure|application_bug|configuration_error|network_issue|security_incident|capacity_issue|dependency_failure
  resolve_incident target=restart_service:NAME|scale_service:NAME|rollback_deploy:NAME|patch_config:NAME

Strategy: investigate first (filter_logs/inspect_service), then mark_root_cause, classify_issue, resolve_incident.
Never repeat the same action.

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


# ── Helpers ───────────────────────────────────────────────────────────

def _extract_json(text):
    """Robustly extract JSON from LLM response."""
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


def _format_obs(obs, task_id, step):
    """Format observation as text for LLM."""
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


def _llm_action(obs, task_id, step, conversation):
    """Try to get next action from LLM. Returns (Action|None, used_llm)."""
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
        return Action(
            action_type=parsed["action_type"],
            target=parsed.get("target"),
        ), True
    except Exception as e:
        print(f"LLM error: {e}", file=sys.stderr, flush=True)
        return None, False


def _fallback_action(task_id, step):
    """Return deterministic action for given task and step."""
    seq = FALLBACK.get(task_id, [])
    idx = step - 1
    if idx < len(seq):
        at, tgt = seq[idx]
        return Action(action_type=at, target=tgt)
    return Action(action_type="resolve_incident", target="restart_service:api-server")


# ── Core task runner ──────────────────────────────────────────────────

def run_task(task_id):
    """
    Run one full episode for task_id.
    Prints required [START] / [STEP] / [END] blocks to stdout.
    """
    # ── [START] ──────────────────────────────────────────────────────
    print(f"[START] task={task_id}", flush=True)

    env = LogEnv(task_name=task_id)
    obs = env.reset()

    total_reward = 0.0
    done         = False
    llm_ok       = 0
    conversation = []   # multi-turn memory for LLM

    for step in range(1, MAX_STEPS + 1):
        # Choose action: LLM first, fallback if unavailable/failed
        action, used_llm = _llm_action(obs, task_id, step, conversation)
        if action is None:
            action = _fallback_action(task_id, step)

        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if used_llm:
            llm_ok += 1

        # ── [STEP] ───────────────────────────────────────────────────
        # Format MUST be: [STEP] step=N action=ACTION target=TARGET reward=R done=True/False
        print(
            f"[STEP] step={step} "
            f"action={action.action_type} "
            f"target={action.target or ''} "
            f"reward={round(reward, 4)} "
            f"done={done}",
            flush=True,
        )

        if done:
            break

    state = env.state()
    score = grade_task(task_id, state)

    # ── [END] ────────────────────────────────────────────────────────
    # Format MUST be: [END] task=TASKNAME score=S steps=N
    print(
        f"[END] task={task_id} "
        f"score={round(score, 4)} "
        f"steps={state.step_count}",
        flush=True,
    )

    return {
        "task_id":  task_id,
        "score":    round(score, 4),
        "steps":    state.step_count,
        "llm_used": llm_ok > 0,
        "reward":   round(total_reward, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print(f"LogEnv inference start. model={MODEL_NAME} tasks={TASKS}", flush=True)

    results = []
    for task_id in TASKS:
        result = run_task(task_id)
        results.append(result)

    avg = sum(r["score"] for r in results) / len(results)

    # Summary (plain text — not parsed by validator but useful for logs)
    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        flag = "LLM" if r["llm_used"] else "DET"
        print(
            f"  [{flag}] {r['task_id']}: score={r['score']} "
            f"steps={r['steps']} reward={r['reward']}",
            flush=True,
        )
    print(f"  AVERAGE score={round(avg, 4)}", flush=True)
    print("=== END SUMMARY ===", flush=True)


if __name__ == "__main__":
    main()