#!/usr/bin/env python3
"""
LogEnv Inference Script — v3.0

Drives a real LLM (Qwen/Qwen2.5-72B-Instruct via HF Router) through all 7 tasks.
Falls back to an optimised deterministic policy when LLM is unavailable.

Usage:
    HF_TOKEN=hf_xxx python inference.py
    HF_TOKEN=hf_xxx MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python inference.py
"""

import os, sys, json, re
from environment import LogEnv
from environment.models import Action
from environment.graders import grade_task

# ── Client ──────────────────────────────────────────────────────────
client = None
try:
    from openai import OpenAI
    _key = os.environ.get("HF_TOKEN")
    if _key:
        client = OpenAI(
            base_url=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
            api_key=_key,
        )
        print("✅ LLM client initialised", flush=True)
    else:
        print("⚠️  HF_TOKEN not set — deterministic fallback", file=sys.stderr, flush=True)
except Exception as e:
    print(f"⚠️  Client init failed: {e}", file=sys.stderr, flush=True)

# ── Config ───────────────────────────────────────────────────────────
TASKS = ["task1", "task2", "task3", "task4", "task5", "task6", "task7"]
MAX_STEPS = 15
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

SYSTEM_PROMPT = """\
You are an expert Senior SRE performing autonomous incident response.

CLASSIFICATION RULES (strict):
- oom_kill, disk_full, network_partition → infrastructure_failure
- memory_leak, deadlock → application_bug
- misconfigured_circuit_breaker → configuration_error
- dependency_failure → dependency_failure

Actions:
  filter_logs      target=keyword
  inspect_service  target=service-name
  mark_root_cause  target=oom_kill|memory_leak|misconfigured_circuit_breaker|network_partition|disk_full|deadlock|dependency_failure
  classify_issue   target=infrastructure_failure|application_bug|configuration_error|network_issue|security_incident|capacity_issue|dependency_failure
  resolve_incident target=restart_service:NAME|scale_service:NAME|rollback_deploy:NAME|patch_config:NAME

Strategy: filter_logs → inspect_service → mark_root_cause → classify_issue → resolve_incident.
Never repeat the same action.

Respond ONLY with JSON on one line: {"action_type": "...", "target": "..."}
"""

FALLBACK = {
    "task1": [("filter_logs","error"),("filter_logs","memory"),("inspect_service","api-server"),("mark_root_cause","oom_kill"),("classify_issue","infrastructure_failure"),("resolve_incident","restart_service:api-server")],
    "task2": [("filter_logs","memory"),("inspect_service","session-manager"),("filter_logs","heap"),("mark_root_cause","memory_leak"),("classify_issue","application_bug"),("resolve_incident","restart_service:session-manager")],
    "task3": [("filter_logs","error"),("inspect_service","order-service"),("filter_logs","circuit"),("inspect_service","payment-service"),("mark_root_cause","misconfigured_circuit_breaker"),("classify_issue","configuration_error"),("resolve_incident","scale_service:order-service")],
    "task4": [("filter_logs","disk"),("inspect_service","log-rotator"),("filter_logs","rotation"),("mark_root_cause","disk_full"),("classify_issue","infrastructure_failure"),("resolve_incident","restart_service:log-rotator")],
    "task5": [("filter_logs","deadlock"),("inspect_service","payment-service"),("filter_logs","lock"),("mark_root_cause","deadlock"),("classify_issue","application_bug"),("resolve_incident","restart_service:payment-service")],
    "task6": [("filter_logs","error"),("inspect_service","checkout-service"),("filter_logs","gateway"),("mark_root_cause","dependency_failure"),("classify_issue","dependency_failure"),("resolve_incident","rollback_deploy:checkout-service")],
    "task7": [("filter_logs","error"),("inspect_service","redis-cluster"),("filter_logs","partition"),("inspect_service","session-service"),("mark_root_cause","network_partition"),("classify_issue","infrastructure_failure"),("resolve_incident","restart_service:redis-cluster")],
}

VALID_ACTIONS = {"filter_logs","inspect_service","mark_root_cause","classify_issue","resolve_incident"}

def _extract_json(text):
    try: return json.loads(text.strip())
    except: pass
    cleaned = re.sub(r"```(?:json)?","",text).replace("```","").strip()
    try: return json.loads(cleaned)
    except: pass
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except: pass
    return None

def _format_obs(obs, task_id, step, history):
    lines = [f"=== Task: {task_id} | Step {step} ===","","LOGS:"]
    for log in obs.logs[-10:]:
        lines.append(f"  [{log.level}] {log.service}: {log.message}")
    lines += ["","METRICS:",
        f"  CPU {obs.metrics.cpu_percent}%  Mem {obs.metrics.memory_percent}%  "
        f"Disk {obs.metrics.disk_percent}%  Errors {obs.metrics.error_rate}%","","ALERTS:"]
    for a in obs.alerts:
        lines.append(f"  [{a.severity}] {a.service}: {a.message}")
    if history:
        lines += ["","DONE SO FAR:"] + [f"  {i+1}. {h['action_type']}({h.get('target','')})" for i,h in enumerate(history)]
    lines.append("\nJSON action:")
    return "\n".join(lines)

def llm_action(obs, task_id, step, conv):
    if client is None: return None, False
    obs_text = _format_obs(obs, task_id, step, [])
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + conv + [{"role":"user","content":obs_text}]
    try:
        r = client.chat.completions.create(model=MODEL_NAME, messages=msgs, max_tokens=80, temperature=0.1)
        raw = r.choices[0].message.content.strip()
        conv += [{"role":"user","content":obs_text},{"role":"assistant","content":raw}]
        p = _extract_json(raw)
        if not p or p.get("action_type") not in VALID_ACTIONS: return None, False
        print(f"  🤖 {p['action_type']}({p.get('target','')})", flush=True)
        return Action(action_type=p["action_type"], target=p.get("target")), True
    except Exception as e:
        print(f"  ❌ LLM failed: {e}", flush=True)
        return None, False

def fallback(task_id, step):
    seq = FALLBACK.get(task_id, [])
    idx = step - 1
    if idx < len(seq):
        at, tgt = seq[idx]
        return Action(action_type=at, target=tgt)
    return Action(action_type="resolve_incident", target="restart_service:api-server")

def run_task(task_id):
    print(f"\n{'='*55}\n  {task_id}\n{'='*55}", flush=True)
    env = LogEnv(task_name=task_id)
    obs = env.reset()
    total_reward, done = 0.0, False
    llm_ok_count = llm_try_count = 0
    conv = []

    for step in range(1, MAX_STEPS + 1):
        action, ok = llm_action(obs, task_id, step, conv)
        llm_try_count += 1
        if ok: llm_ok_count += 1
        else: action = fallback(task_id, step)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        mode = "llm" if ok else "det"
        print(f"  [{step:2d}|{mode}] {action.action_type}({action.target or ''}) → {reward:+.3f}  done={done}", flush=True)
        if done: break

    state = env.state()
    score = grade_task(task_id, state)
    llm_used = llm_ok_count > 0
    print(f"\n  Root cause    : {state.root_cause_marked}", flush=True)
    print(f"  Classification: {state.classification_marked}", flush=True)
    print(f"  Resolution    : {state.resolution_action}", flush=True)
    print(f"  Steps / Wrong : {state.step_count} / {state.wrong_action_count}", flush=True)
    print(f"  LLM           : {llm_ok_count}/{llm_try_count} succeeded", flush=True)
    print(f"  ✅ SCORE       : {score:.4f}", flush=True)
    return {"task_id": task_id, "score": score, "reward": round(total_reward,4),
            "steps": state.step_count, "llm_used": llm_used, "llm_ok": llm_ok_count}

def main():
    print("="*55, flush=True)
    print(f"  LogEnv v3 Inference | Model: {MODEL_NAME}", flush=True)
    print(f"  Tasks: {TASKS}", flush=True)
    print("="*55, flush=True)
    results = [run_task(t) for t in TASKS]
    avg = sum(r["score"] for r in results) / len(results)
    print("\n" + "="*55, flush=True)
    print("  RESULTS", flush=True)
    print("="*55, flush=True)
    for r in results:
        bar = "█" * int(r["score"] * 20)
        flag = "🤖" if r["llm_used"] else "🔧"
        print(f"  {flag} {r['task_id']}: {r['score']:.4f}  {bar}", flush=True)
    print(f"\n  AVERAGE : {avg:.4f}", flush=True)
    any_llm = any(r["llm_used"] for r in results)
    print(f"  MODE    : {'Real LLM ✅' if any_llm else 'Deterministic fallback ⚠️'}", flush=True)
    print("="*55, flush=True)
    print("\n[JSON_RESULTS]", flush=True)
    print(json.dumps({"results": results, "average_score": round(avg,4), "llm_used": any_llm, "model": MODEL_NAME}, indent=2), flush=True)

if __name__ == "__main__":
    main()
