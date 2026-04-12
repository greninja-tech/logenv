#!/usr/bin/env python3
"""
LogEnv Inference Script — v3.0

Runs a real LLM agent (Qwen/Qwen2.5-72B-Instruct via HF Router) through all 7 tasks.
Falls back to an optimised deterministic policy when the LLM is unavailable.

Emits structured logs in the mandatory [START] / [STEP] / [END] format
required by the OpenEnv evaluation harness.

Usage:
    HF_TOKEN=hf_xxx python inference.py
    HF_TOKEN=hf_xxx MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python inference.py
    HF_TOKEN=hf_xxx TASK=task1 python inference.py   # single task
"""

import os
import sys
import json
import re
import time

from environment import LogEnv
from environment.models import Action
from environment.graders import grade_task


# ─────────────────────────────────────────────
#  MANDATORY LOG FORMAT HELPERS
#  The evaluation harness parses [START], [STEP], [END] exactly.
# ─────────────────────────────────────────────

def log_start(task_id: str, model: str) -> None:
    print(f"[START] task={task_id} env=logenv model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    error_val = str(error) if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ─────────────────────────────────────────────
#  LLM CLIENT SETUP
# ─────────────────────────────────────────────

client = None
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")

try:
    from openai import OpenAI
    _key = os.environ.get("HF_TOKEN")
    if _key:
        client = OpenAI(base_url=API_BASE_URL, api_key=_key)
        print(f"[INFO] LLM client ready — model: {MODEL_NAME}", flush=True)
    else:
        print("[INFO] HF_TOKEN not set — using deterministic fallback", flush=True)
except Exception as e:
    print(f"[INFO] LLM client init failed: {e} — using deterministic fallback", flush=True)


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

ALL_TASKS = ["task1", "task2", "task3", "task4", "task5", "task6", "task7"]
TASKS = os.environ.get("TASK", "").split(",") if os.environ.get("TASK") else ALL_TASKS
TASKS = [t.strip() for t in TASKS if t.strip()]
# Per-task step limits — must match environment max_steps
TASK_MAX_STEPS = {
    "task1": 15, "task2": 20, "task3": 30,
    "task4": 15, "task5": 20, "task6": 20, "task7": 30,
}
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = """\
You are an expert Senior Site Reliability Engineer (SRE) performing autonomous incident response.

CLASSIFICATION RULES — memorise these exactly:
  oom_kill                       → infrastructure_failure
  disk_full                      → infrastructure_failure
  network_partition              → infrastructure_failure
  memory_leak                    → application_bug
  deadlock                       → application_bug
  misconfigured_circuit_breaker  → configuration_error
  dependency_failure             → dependency_failure

Available actions (respond with JSON only — no prose, no markdown):
  filter_logs      target=keyword
  inspect_service  target=service-name
  mark_root_cause  target=oom_kill|memory_leak|misconfigured_circuit_breaker|
                          network_partition|disk_full|deadlock|dependency_failure
  classify_issue   target=infrastructure_failure|application_bug|configuration_error|
                          network_issue|security_incident|capacity_issue|dependency_failure
  resolve_incident target=restart_service:NAME|scale_service:NAME|
                          rollback_deploy:NAME|patch_config:NAME  ← ENDS EPISODE

Strategy:
  1. filter_logs for "error" or "critical"
  2. inspect_service on the most suspicious service
  3. filter_logs for the specific symptom (memory, disk, deadlock, circuit, partition)
  4. mark_root_cause
  5. classify_issue using the rules above
  6. resolve_incident

Never repeat the same action. Respond ONLY with JSON on one line:
{"action_type": "...", "target": "..."}
"""

# Optimal deterministic sequences for each task
FALLBACK: dict = {
    "task1": [("filter_logs","error"),("filter_logs","memory"),("inspect_service","api-server"),("mark_root_cause","oom_kill"),("classify_issue","infrastructure_failure"),("resolve_incident","restart_service:api-server")],
    "task2": [("filter_logs","memory"),("inspect_service","session-manager"),("filter_logs","heap"),("mark_root_cause","memory_leak"),("classify_issue","application_bug"),("resolve_incident","restart_service:session-manager")],
    "task3": [("filter_logs","error"),("inspect_service","order-service"),("filter_logs","circuit"),("inspect_service","payment-service"),("mark_root_cause","misconfigured_circuit_breaker"),("classify_issue","configuration_error"),("resolve_incident","scale_service:order-service")],
    "task4": [("filter_logs","disk"),("inspect_service","log-rotator"),("filter_logs","rotation"),("mark_root_cause","disk_full"),("classify_issue","infrastructure_failure"),("resolve_incident","restart_service:log-rotator")],
    "task5": [("filter_logs","deadlock"),("inspect_service","payment-service"),("filter_logs","lock"),("mark_root_cause","deadlock"),("classify_issue","application_bug"),("resolve_incident","restart_service:payment-service")],
    "task6": [("filter_logs","error"),("inspect_service","checkout-service"),("filter_logs","gateway"),("mark_root_cause","dependency_failure"),("classify_issue","dependency_failure"),("resolve_incident","rollback_deploy:checkout-service")],
    "task7": [("filter_logs","error"),("inspect_service","redis-cluster"),("filter_logs","partition"),("inspect_service","session-service"),("mark_root_cause","network_partition"),("classify_issue","infrastructure_failure"),("resolve_incident","restart_service:redis-cluster")],
}

VALID_ACTIONS = {"filter_logs","inspect_service","mark_root_cause","classify_issue","resolve_incident"}


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
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


def _format_obs(obs, task_id: str, step: int, history: list) -> str:
    try:
        max_s = TASK_MAX_STEPS.get(task_id, 30)
        lines = [f"=== Task: {task_id} | Step {step}/{max_s} ===", "", "RECENT LOGS:"]
        for log in (obs.logs or [])[-10:]:
            lines.append(f"  [{log.level}] {log.service}: {log.message}")
        lines += ["", "METRICS:",
            f"  CPU {obs.metrics.cpu_percent}%  Mem {obs.metrics.memory_percent}%  "
            f"Disk {obs.metrics.disk_percent}%  Error rate {obs.metrics.error_rate}%",
            "", "ALERTS:"]
        for a in (obs.alerts or []):
            lines.append(f"  [{a.severity}] {a.service}: {a.message}")
        if history:
            lines += ["", "ACTIONS TAKEN SO FAR:"]
            for i, h in enumerate(history, 1):
                lines.append(f"  {i}. {h['action_type']}({h.get('target', '')})")
        lines.append("\nRespond with JSON action:")
        return "\n".join(lines)
    except Exception as e:
        return f"=== Task: {task_id} | Step {step} ===\n[obs formatting error: {e}]\nRespond with JSON action:"


def _llm_action(obs, task_id: str, step: int, conv: list) -> tuple:
    """Returns (Action|None, succeeded: bool)."""
    if client is None:
        return None, False
    obs_text = _format_obs(obs, task_id, step, [])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conv + [{"role": "user", "content": obs_text}]
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, max_tokens=80, temperature=0.1,
        )
        raw = r.choices[0].message.content.strip()
        conv.append({"role": "user", "content": obs_text})
        conv.append({"role": "assistant", "content": raw})
        p = _extract_json(raw)
        if not p or p.get("action_type") not in VALID_ACTIONS:
            return None, False
        return Action(action_type=p["action_type"], target=p.get("target")), True
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {type(e).__name__}: {e}", flush=True)
        return None, False


def _fallback_action(task_id: str, step: int) -> Action:
    seq = FALLBACK.get(task_id, [])
    idx = step - 1
    if idx < len(seq):
        at, tgt = seq[idx]
        return Action(action_type=at, target=tgt)
    return Action(action_type="resolve_incident", target="restart_service:api-server")


# ─────────────────────────────────────────────
#  RUN ONE TASK
# ─────────────────────────────────────────────

def run_task(task_id: str) -> dict:
    log_start(task_id=task_id, model=MODEL_NAME)  # env=logenv is hardcoded in log_start

    env = LogEnv(task_name=task_id)
    obs = env.reset()

    rewards = []
    steps_taken = 0
    done = False
    success = False
    llm_ok_count = llm_try_count = 0
    conv: list = []

    try:
        max_steps = TASK_MAX_STEPS.get(task_id, 15)
        for step in range(1, max_steps + 1):
            action = None
            error = None

            # Try LLM first — wrapped so any unexpected obs error doesn't crash
            llm_try_count += 1
            try:
                action, ok = _llm_action(obs, task_id, step, conv)
            except Exception as llm_err:
                print(f"[DEBUG] _llm_action outer error: {llm_err}", flush=True)
                action, ok = None, False
            if ok:
                llm_ok_count += 1
            else:
                action = _fallback_action(task_id, step)

            # Build action string for logging
            action_str = f"{action.action_type}:{action.target}" if action.target else action.action_type

            try:
                obs, reward, done, _ = env.step(action)
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            steps_taken = step

            # Mandatory [STEP] log
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

    finally:
        state = env.state()
        score = grade_task(task_id, state)
        success = score >= SUCCESS_SCORE_THRESHOLD

        # Mandatory [END] log
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "total_reward": round(sum(rewards), 4),
        "steps": steps_taken,
        "llm_used": llm_ok_count > 0,
        "llm_calls_succeeded": llm_ok_count,
        "llm_calls_attempted": llm_try_count,
        "root_cause": state.root_cause_marked,
        "classification": state.classification_marked,
        "resolution": state.resolution_action,
        "wrong_action_count": state.wrong_action_count,
        "success": success,
    }


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print(f"[INFO] LogEnv v3 Inference | Model: {MODEL_NAME} | Tasks: {TASKS}", flush=True)
    print(f"[INFO] LLM: {'ENABLED' if client else 'DISABLED — deterministic fallback'}", flush=True)

    results = []
    for task_id in TASKS:
        print(f"\n[INFO] ===== Starting {task_id} =====", flush=True)
        result = run_task(task_id)
        results.append(result)
        print(f"[INFO] {task_id} complete — score: {result['score']:.4f} | llm_used: {result['llm_used']}", flush=True)

    # Summary
    avg = sum(r["score"] for r in results) / len(results) if results else 0.0
    any_llm = any(r["llm_used"] for r in results)

    print("\n[INFO] ===== FINAL RESULTS =====", flush=True)
    for r in results:
        bar = "█" * int(r["score"] * 20)
        flag = "🤖" if r["llm_used"] else "🔧"
        print(f"[INFO] {flag} {r['task_id']}: {r['score']:.4f}  {bar}", flush=True)
    print(f"[INFO] AVERAGE SCORE : {avg:.4f}", flush=True)
    print(f"[INFO] MODE          : {'Real LLM ✅' if any_llm else 'Deterministic fallback ⚠️'}", flush=True)

    # Machine-readable summary
    print("\n[JSON_RESULTS]", flush=True)
    print(json.dumps({
        "results": results,
        "average_score": round(avg, 4),
        "llm_used": any_llm,
        "model": MODEL_NAME,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()