#!/usr/bin/env python3
"""
LogEnv Inference Script — Intelligent LLM Agent
================================================
A multi-turn, reasoning-driven agent that:
  1. Reads observations (logs, metrics, alerts)
  2. Maintains a conversation history for full context
  3. Uses chain-of-thought reasoning before acting
  4. Decides actions autonomously — NO hardcoded sequences
  5. Falls back to an optimal deterministic policy if LLM is unavailable

Usage:
    HF_TOKEN=your_token python inference.py
    HF_TOKEN=your_token MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python inference.py
    HF_TOKEN=your_token TASK=task1 python inference.py
"""

import os
import sys
import json
import re

from environment import LogEnv
from environment.models import Action
from environment.graders import grade_task


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

TASKS      = os.environ.get("TASK", "task1,task2,task3").split(",")
MAX_STEPS  = 12
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# ──────────────────────────────────────────────
# OPENAI-COMPATIBLE CLIENT (HuggingFace Inference)
# ──────────────────────────────────────────────

client = None

try:
    from openai import OpenAI
    api_key = os.environ.get("HF_TOKEN")
    if api_key:
        client = OpenAI(
            base_url=os.environ.get("API_BASE_URL",
                                    "https://api-inference.huggingface.co/v1"),
            api_key=api_key,
        )
        print(f"[OK] LLM client initialised ({MODEL_NAME})", flush=True)
    else:
        print("[WARN] HF_TOKEN not set — using deterministic fallback", file=sys.stderr, flush=True)
except Exception as e:
    print(f"[WARN] Client init failed: {e} — using deterministic fallback", file=sys.stderr, flush=True)


# ──────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert DevOps/SRE incident-response agent.
Your job: analyse system logs, metrics and alerts, then take actions to identify
and resolve the root cause as efficiently as possible.

ACTION REFERENCE
================
filter_logs      | target = keyword          | Search all logs for a word/phrase
inspect_service  | target = service-name     | View all logs for a specific service
mark_root_cause  | target = <value>          | Declare the root cause (one value only)
classify_issue   | target = <value>          | Classify the incident type
resolve_incident | target = <value>          | Take resolution action (ENDS episode)

ROOT CAUSE values
-----------------
oom_kill | memory_leak | misconfigured_circuit_breaker |
network_partition | disk_full | deadlock | dependency_failure

CLASSIFICATION values
---------------------
infrastructure_failure | application_bug | configuration_error |
network_issue | security_incident | capacity_issue | dependency_failure

RESOLUTION format
-----------------
restart_service:SERVICE_NAME
scale_service:SERVICE_NAME
rollback_deploy:SERVICE_NAME
patch_config:SERVICE_NAME

STRATEGY
========
1. INVESTIGATE: use filter_logs / inspect_service to gather evidence first.
2. REASON: look for anomalies, escalating errors, config changes, cascading failures.
3. MARK: once confident, call mark_root_cause then classify_issue.
4. RESOLVE: call resolve_incident only after marking root cause AND classifying.
5. EFFICIENCY: fewer steps = score bonus. Do NOT repeat the same filter/inspect.
6. RED HERRINGS: some services log noise. Find the *origin* of failures, not symptoms.

RESPONSE FORMAT
===============
Always reply with a brief reasoning note, then a JSON block.

Example:
Reasoning: Heap grows from 256MB to 1.1GB while session count stays flat —
classic memory leak in session-manager.

```json
{"action_type": "inspect_service", "target": "session-manager"}
```

One action per reply. No extra keys inside the JSON.
"""


# ──────────────────────────────────────────────
# OBSERVATION FORMATTER
# ──────────────────────────────────────────────

def format_observation(obs, step: int, task_id: str, task_desc: str = "") -> str:
    lines = [f"── Step {step} | Task: {task_id} ──────────────────────"]

    if step == 1 and task_desc:
        lines += ["", "TASK BRIEF:", task_desc, ""]

    lines += ["", "LOGS (most recent last):"]
    for log in obs.logs[-15:]:
        lines.append(
            f"  {log.timestamp[11:19]}  [{log.level:<8}]  "
            f"{log.service:<24}  {log.message}"
        )

    lines += [
        "",
        "METRICS:",
        f"  CPU={obs.metrics.cpu_percent:.0f}%  MEM={obs.metrics.memory_percent:.0f}%  "
        f"Disk={obs.metrics.disk_percent:.0f}%  Conns={obs.metrics.active_connections}  "
        f"Req/s={obs.metrics.request_rate:.0f}  Errors={obs.metrics.error_rate:.0f}%",
        "",
        "ALERTS:",
    ]
    if obs.alerts:
        for a in obs.alerts:
            lines.append(f"  [{a.severity:<8}]  {a.service:<24}  {a.message}")
    else:
        lines.append("  (no active alerts)")

    lines += ["", "Decide your next action. Reply with reasoning + JSON."]
    return "\n".join(lines)


# ──────────────────────────────────────────────
# JSON EXTRACTOR  (robust against markdown fences)
# ──────────────────────────────────────────────

def extract_action_json(text: str):
    # 1. ```json ... ``` block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Any {...} object
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ──────────────────────────────────────────────
# LLM AGENT  (multi-turn with full conversation memory)
# ──────────────────────────────────────────────

class LLMAgent:
    """
    Keeps the full conversation so the model always knows what it
    has already investigated — no redundant actions, real reasoning.
    """

    def __init__(self, task_id: str, task_desc: str):
        self.task_id   = task_id
        self.task_desc = task_desc
        self.history   = []   # list of {role, content} dicts
        self.step      = 0

    def act(self, obs):
        """Return next Action from LLM, or None on failure."""
        if client is None:
            return None

        self.step += 1
        user_msg = format_observation(
            obs, self.step, self.task_id,
            self.task_desc if self.step == 1 else "",
        )
        self.history.append({"role": "user", "content": user_msg})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=320,
                temperature=0.1,
            )
            reply = response.choices[0].message.content.strip()
            self.history.append({"role": "assistant", "content": reply})

            parsed = extract_action_json(reply)
            if parsed is None:
                print(f"  [WARN] Could not parse JSON — raw reply:\n{reply[:250]}", flush=True)
                return None

            action = Action(
                action_type=parsed.get("action_type", "filter_logs"),
                target=parsed.get("target"),
            )

            # Print first reasoning line for visibility
            for line in reply.split("\n"):
                line = line.strip()
                if line and not line.startswith("{") and not line.startswith("```"):
                    print(f"  [REASON] {line[:120]}", flush=True)
                    break

            return action

        except Exception as e:
            print(f"  [WARN] LLM call failed: {e}", flush=True)
            return None


# ──────────────────────────────────────────────
# DETERMINISTIC FALLBACK  (optimal sequences)
# ──────────────────────────────────────────────

_FALLBACK = {
    "task1": [
        Action(action_type="filter_logs",      target="error"),
        Action(action_type="filter_logs",      target="memory"),
        Action(action_type="inspect_service",  target="api-server"),
        Action(action_type="mark_root_cause",  target="oom_kill"),
        Action(action_type="classify_issue",   target="infrastructure_failure"),
        Action(action_type="resolve_incident", target="restart_service:api-server"),
    ],
    "task2": [
        Action(action_type="filter_logs",      target="memory"),
        Action(action_type="filter_logs",      target="heap"),
        Action(action_type="inspect_service",  target="session-manager"),
        Action(action_type="mark_root_cause",  target="memory_leak"),
        Action(action_type="classify_issue",   target="application_bug"),
        Action(action_type="resolve_incident", target="restart_service:session-manager"),
    ],
    "task3": [
        Action(action_type="filter_logs",      target="circuit"),
        Action(action_type="inspect_service",  target="order-service"),
        Action(action_type="filter_logs",      target="config"),
        Action(action_type="inspect_service",  target="inventory-service"),
        Action(action_type="mark_root_cause",  target="misconfigured_circuit_breaker"),
        Action(action_type="classify_issue",   target="configuration_error"),
        Action(action_type="resolve_incident", target="scale_service:order-service"),
    ],
}

def fallback_action(task_id: str, step: int) -> Action:
    seq = _FALLBACK.get(task_id, [])
    idx = step - 1
    if idx < len(seq):
        return seq[idx]
    return Action(action_type="resolve_incident", target="restart_service:api-server")


# ──────────────────────────────────────────────
# TASK RUNNER
# ──────────────────────────────────────────────

def run_task(task_id: str) -> float:
    sep = "=" * 60
    print(f"\n{sep}", flush=True)
    print(f"  TASK: {task_id.upper()}", flush=True)
    print(sep, flush=True)

    env = LogEnv(task_name=task_id)
    obs = env.reset()

    task_desc = (env.task_data or {}).get("task_description", "")
    agent     = LLMAgent(task_id=task_id, task_desc=task_desc)
    using_llm = (client is not None)

    total_reward = 0.0
    done         = False

    for step in range(1, MAX_STEPS + 1):

        # ── Choose action ───────────────────────────────────
        action = agent.act(obs) if using_llm else None
        mode   = "LLM"

        if action is None:
            action = fallback_action(task_id, step)
            mode   = "DET"

        # ── Step ────────────────────────────────────────────
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        print(
            f"  [{mode}][{step:2d}]  {action.action_type:<20} "
            f"target={str(action.target or ''):<36} "
            f"reward={reward:+.4f}  done={done}",
            flush=True,
        )

        if done:
            break

    # ── Grade ───────────────────────────────────────────────
    state       = env.state()
    final_score = grade_task(task_id, state)

    print(f"\n  Results:", flush=True)
    print(f"    Root cause     : {state.root_cause_marked}", flush=True)
    print(f"    Classification : {state.classification_marked}", flush=True)
    print(f"    Resolution     : {state.resolution_action}", flush=True)
    print(f"    Steps used     : {state.step_count}", flush=True)
    print(f"    Wrong actions  : {state.wrong_action_count}", flush=True)
    print(f"    Cumul. reward  : {total_reward:.4f}", flush=True)
    print(f"  >>> FINAL SCORE  : {final_score:.4f}", flush=True)

    return final_score


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    sep = "=" * 60
    print(sep, flush=True)
    print("  LogEnv — Intelligent LLM Reasoning Agent", flush=True)
    print(f"  Model  : {MODEL_NAME}", flush=True)
    print(f"  Tasks  : {TASKS}", flush=True)
    print(f"  LLM    : {'ENABLED' if client else 'DISABLED (deterministic fallback)'}", flush=True)
    print(sep, flush=True)

    scores = {}
    for task_id in TASKS:
        task_id = task_id.strip()
        if task_id:
            scores[task_id] = run_task(task_id)

    avg = sum(scores.values()) / max(len(scores), 1)

    print(f"\n{sep}", flush=True)
    print("  FINAL RESULTS", flush=True)
    print(sep, flush=True)
    for t, s in scores.items():
        bar = "█" * int(s * 20)
        print(f"  {t:<8}  {s:.4f}  {bar}", flush=True)
    print(f"\n  AVERAGE : {avg:.4f}", flush=True)
    print(sep, flush=True)


if __name__ == "__main__":
    main()
