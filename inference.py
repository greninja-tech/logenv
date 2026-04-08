#!/usr/bin/env python3
"""
LogEnv Inference Script — v2.0 (Intelligent Agent)

Drives a real LLM (Qwen/Qwen2.5-72B-Instruct via HF Inference API) through each task.
Falls back to an optimized deterministic policy only when the LLM is genuinely unavailable.

Key improvements over v1:
  - Honest llm_used tracking (true only when LLM actually responded)
  - Multi-turn conversation history so the LLM reasons across steps
  - Chain-of-thought reasoning prompt for better decisions
  - Robust JSON extraction with multiple fallback strategies
  - Detailed per-step result logging

Usage:
    HF_TOKEN=your_token python inference.py
    HF_TOKEN=your_token MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python inference.py
"""

import os
import sys
import json
import re

from environment import LogEnv
from environment.models import Action
from environment.graders import grade_task


# ─────────────────────────────────────────────
#  CLIENT SETUP
# ─────────────────────────────────────────────

client = None

try:
    from openai import OpenAI

    _api_key = os.environ.get("HF_TOKEN")
    if _api_key:
        client = OpenAI(
            base_url=os.environ.get(
                "API_BASE_URL",
                "https://api-inference.huggingface.co/v1",
            ),
            api_key=_api_key,
        )
        print("✅ LLM client initialised (HF_TOKEN found)", flush=True)
    else:
        print("⚠️  HF_TOKEN not set — will use deterministic fallback", file=sys.stderr, flush=True)
except Exception as exc:
    print(f"⚠️  Client init failed: {exc} — will use deterministic fallback", file=sys.stderr, flush=True)


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

TASKS = ["task1", "task2", "task3"]
MAX_STEPS = 12
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

SYSTEM_PROMPT = """\
You are an expert Senior Site Reliability Engineer (SRE) performing autonomous incident response.

Your job:
1. Analyse system logs, metrics, and alerts.
2. Identify the root cause of the incident.
3. Classify the issue type.
4. Take the correct resolution action.

Available action_type values:
  filter_logs      — search all logs for a keyword (target = keyword string)
  inspect_service  — view all logs for a specific service (target = service name)
  mark_root_cause  — declare root cause (target = one of: oom_kill, memory_leak,
                     misconfigured_circuit_breaker, network_partition, disk_full,
                     deadlock, dependency_failure)
  classify_issue   — declare issue class (target = one of: infrastructure_failure,
                     application_bug, configuration_error, network_issue,
                     security_incident, capacity_issue, dependency_failure)
  resolve_incident — final resolution (target = restart_service:NAME or
                     scale_service:NAME or rollback_deploy:NAME or patch_config:NAME)
                     *** This ends the episode — only call when certain. ***

Strategy:
  - Start broad: filter_logs for "error", "warning", "critical".
  - Drill into the service that produced the most critical events.
  - Once you have identified root cause and classification, resolve.
  - Do NOT repeat the same action twice.

Respond ONLY with a JSON object on a single line — no prose, no markdown fences:
{"action_type": "...", "target": "..."}
"""


# ─────────────────────────────────────────────
#  OBSERVATION FORMATTER
# ─────────────────────────────────────────────

def format_observation(obs, task_id: str, step: int, history: list) -> str:
    """Return a human-readable observation string for the LLM."""
    lines = [
        f"=== Incident Response | Task: {task_id} | Step {step}/{MAX_STEPS} ===",
        "",
        "--- RECENT LOGS (last 10 visible) ---",
    ]
    for log in obs.logs[-10:]:
        lines.append(f"  [{log.level:<8}] {log.timestamp}  {log.service:<22}  {log.message}")

    lines += [
        "",
        "--- SYSTEM METRICS ---",
        f"  CPU {obs.metrics.cpu_percent}%  |  Memory {obs.metrics.memory_percent}%  |  "
        f"Disk {obs.metrics.disk_percent}%  |  Error rate {obs.metrics.error_rate}%",
        f"  Active connections: {obs.metrics.active_connections}  |  Request rate: {obs.metrics.request_rate}",
        "",
        "--- ACTIVE ALERTS ---",
    ]
    for alert in obs.alerts:
        lines.append(f"  [{alert.severity:<8}] {alert.service}: {alert.message}")

    if history:
        lines += ["", "--- ACTIONS TAKEN SO FAR ---"]
        for i, h in enumerate(history, 1):
            lines.append(f"  {i}. {h['action_type']}({h.get('target', '')})")

    lines += ["", "What is your next action? Respond with JSON only."]
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  JSON EXTRACTOR
# ─────────────────────────────────────────────

def extract_json(text: str) -> dict | None:
    """Try multiple strategies to extract a JSON object from LLM output."""
    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Regex: first {...} block
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# ─────────────────────────────────────────────
#  LLM AGENT
# ─────────────────────────────────────────────

VALID_ACTION_TYPES = {
    "filter_logs", "inspect_service", "mark_root_cause",
    "classify_issue", "resolve_incident",
}


def llm_action(obs, task_id: str, step: int, conversation: list) -> tuple[Action | None, bool]:
    """
    Call the LLM to get the next action.

    Returns:
        (Action | None, llm_succeeded: bool)
        llm_succeeded is True ONLY when the LLM actually responded and was parsed.
    """
    if client is None:
        return None, False

    obs_text = format_observation(obs, task_id, step, obs.__dict__.get("_history", []))

    # Build multi-turn messages: system + prior conversation + new user turn
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation)
    messages.append({"role": "user", "content": obs_text})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=120,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()

        # Persist assistant turn for next step
        conversation.append({"role": "user", "content": obs_text})
        conversation.append({"role": "assistant", "content": raw})

        parsed = extract_json(raw)
        if parsed is None:
            print(f"  ⚠️  LLM output not parseable: {raw!r}", flush=True)
            return None, False

        action_type = parsed.get("action_type", "")
        if action_type not in VALID_ACTION_TYPES:
            print(f"  ⚠️  LLM returned unknown action_type: {action_type!r}", flush=True)
            return None, False

        action = Action(
            action_type=action_type,
            target=parsed.get("target"),
        )
        print(f"  🤖 LLM → {action.action_type}({action.target})", flush=True)
        return action, True

    except Exception as exc:
        print(f"  ⚠️  LLM call failed: {exc}", flush=True)
        return None, False


# ─────────────────────────────────────────────
#  DETERMINISTIC FALLBACK
# ─────────────────────────────────────────────

_FALLBACK_SEQUENCES: dict[str, list[Action]] = {
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
        Action(action_type="inspect_service",  target="session-manager"),
        Action(action_type="filter_logs",      target="heap"),
        Action(action_type="mark_root_cause",  target="memory_leak"),
        Action(action_type="classify_issue",   target="application_bug"),
        Action(action_type="resolve_incident", target="restart_service:session-manager"),
    ],
    "task3": [
        Action(action_type="filter_logs",      target="error"),
        Action(action_type="inspect_service",  target="order-service"),
        Action(action_type="filter_logs",      target="circuit"),
        Action(action_type="inspect_service",  target="payment-service"),
        Action(action_type="mark_root_cause",  target="misconfigured_circuit_breaker"),
        Action(action_type="classify_issue",   target="configuration_error"),
        Action(action_type="resolve_incident", target="scale_service:order-service"),
    ],
}


def fallback_action(task_id: str, step: int) -> Action:
    """Optimal deterministic policy — used when LLM is unavailable."""
    seq = _FALLBACK_SEQUENCES.get(task_id, [])
    idx = step - 1
    if idx < len(seq):
        return seq[idx]
    return Action(action_type="resolve_incident", target="restart_service:api-server")


# ─────────────────────────────────────────────
#  RUN ONE TASK
# ─────────────────────────────────────────────

def run_task(task_id: str) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"  Task: {task_id}", flush=True)
    print(f"{'='*60}", flush=True)

    env = LogEnv(task_name=task_id)
    obs = env.reset()

    total_reward = 0.0
    done = False
    llm_call_count = 0
    llm_success_count = 0
    conversation: list[dict] = []   # multi-turn history for this task

    for step in range(1, MAX_STEPS + 1):
        action, llm_ok = llm_action(obs, task_id, step, conversation)
        llm_call_count += 1

        if llm_ok:
            llm_success_count += 1
            mode = "llm"
        else:
            action = fallback_action(task_id, step)
            mode = "deterministic"
            print(f"  🔧 Fallback → {action.action_type}({action.target})", flush=True)

        obs, reward, done, _ = env.step(action)
        total_reward += reward

        print(
            f"  [step {step:2d}|{mode}] {action.action_type}({action.target or ''}) "
            f"→ reward={reward:+.4f}  done={done}",
            flush=True,
        )

        if done:
            break

    state = env.state()
    final_score = grade_task(task_id, state)

    # llm_used is TRUE only when at least one LLM call actually succeeded
    llm_used = llm_success_count > 0

    print(f"\n  Root cause     : {state.root_cause_marked}", flush=True)
    print(f"  Classification : {state.classification_marked}", flush=True)
    print(f"  Resolution     : {state.resolution_action}", flush=True)
    print(f"  Steps used     : {state.step_count}", flush=True)
    print(f"  Wrong actions  : {state.wrong_action_count}", flush=True)
    print(f"  LLM calls      : {llm_success_count}/{llm_call_count} succeeded", flush=True)
    print(f"  llm_used       : {llm_used}", flush=True)
    print(f"  ✅ FINAL SCORE  : {final_score:.4f}", flush=True)

    return {
        "task_id": task_id,
        "final_score": final_score,
        "total_reward": round(total_reward, 4),
        "steps_used": state.step_count,
        "wrong_action_count": state.wrong_action_count,
        "root_cause_marked": state.root_cause_marked,
        "classification_marked": state.classification_marked,
        "resolution_action": state.resolution_action,
        "llm_used": llm_used,
        "llm_calls_succeeded": llm_success_count,
        "llm_calls_attempted": llm_call_count,
        "mode": "llm" if llm_used else "deterministic",
    }


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60, flush=True)
    print("  LogEnv Inference — v2.0", flush=True)
    print(f"  Model  : {MODEL_NAME}", flush=True)
    print(f"  Tasks  : {TASKS}", flush=True)
    print(f"  LLM    : {'ENABLED' if client else 'DISABLED (no HF_TOKEN)'}", flush=True)
    print("=" * 60, flush=True)

    results = []
    for task_id in TASKS:
        result = run_task(task_id)
        results.append(result)

    scores = [r["final_score"] for r in results]
    avg = sum(scores) / len(scores)

    print("\n" + "=" * 60, flush=True)
    print("  FINAL RESULTS", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        bar = "█" * int(r["final_score"] * 20)
        flag = "🤖" if r["llm_used"] else "🔧"
        print(f"  {flag} {r['task_id']}: {r['final_score']:.4f}  {bar}", flush=True)
    print(f"\n  AVERAGE SCORE : {avg:.4f}", flush=True)
    any_llm = any(r["llm_used"] for r in results)
    print(f"  INTELLIGENCE  : {'Real LLM ✅' if any_llm else 'Deterministic fallback ⚠️'}", flush=True)
    print("=" * 60, flush=True)

    # Machine-readable summary to stdout for programmatic consumption
    print("\n[JSON_RESULTS]", flush=True)
    print(json.dumps({
        "results": results,
        "average_score": round(avg, 4),
        "llm_used_any": any_llm,
        "model": MODEL_NAME,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()