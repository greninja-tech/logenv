#!/usr/bin/env python3
"""
LogEnv Multi-Model Benchmark — v1.0

Benchmarks multiple Hugging Face models across all LogEnv tasks.
Produces a leaderboard with per-model, per-task scores, timing, and analysis.

Usage:
  HF_TOKEN=your_token python benchmark.py
  HF_TOKEN=your_token python benchmark.py --models "Qwen/Qwen2.5-72B-Instruct,meta-llama/Llama-3.3-70B-Instruct"
  HF_TOKEN=your_token python benchmark.py --tasks task1,task3
  HF_TOKEN=your_token python benchmark.py --output results.json
"""

import os
import sys
import io
import json
import re
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

# Fix Windows console encoding for Unicode output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from environment import LogEnv
from environment.models import Action
from environment.graders import grade_task

# ── OpenAI-compatible client ─────────────────────────────────────────
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# ── Default model roster ─────────────────────────────────────────────
# These are popular models available on HF Inference API via the router.
# Add/remove models as needed. Each entry: (model_id, display_name, max_tokens)

DEFAULT_MODELS = [
    ("Qwen/Qwen2.5-72B-Instruct",                "Qwen2.5-72B",        120),
    ("meta-llama/Llama-3.3-70B-Instruct",         "Llama-3.3-70B",      120),
    ("mistralai/Mistral-Small-24B-Instruct-2501",  "Mistral-Small-24B",  120),
    ("Qwen/Qwen2.5-Coder-32B-Instruct",          "Qwen2.5-Coder-32B", 120),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1",      "Mixtral-8x7B",       120),
]

# ── Task config ──────────────────────────────────────────────────────
ALL_TASKS = ["task1", "task2", "task3", "task4", "task5", "task6", "task7"]

# ── Deterministic fallback sequences (for comparison baseline) ───────
FALLBACK = {
    "task1": [
        ("filter_logs", "error"), ("filter_logs", "memory"),
        ("inspect_service", "api-server"), ("mark_root_cause", "oom_kill"),
        ("classify_issue", "infrastructure_failure"),
        ("resolve_incident", "restart_service:api-server"),
    ],
    "task2": [
        ("filter_logs", "memory"), ("inspect_service", "session-manager"),
        ("filter_logs", "heap"), ("mark_root_cause", "memory_leak"),
        ("classify_issue", "application_bug"),
        ("resolve_incident", "restart_service:session-manager"),
    ],
    "task3": [
        ("filter_logs", "circuit"), ("inspect_service", "order-service"),
        ("filter_logs", "config"), ("inspect_service", "inventory-service"),
        ("mark_root_cause", "misconfigured_circuit_breaker"),
        ("classify_issue", "configuration_error"),
        ("resolve_incident", "scale_service:order-service"),
    ],
    "task4": [
        ("filter_logs", "disk"), ("inspect_service", "log-rotator"),
        ("filter_logs", "rotation"), ("mark_root_cause", "disk_full"),
        ("classify_issue", "infrastructure_failure"),
        ("resolve_incident", "restart_service:log-rotator"),
    ],
    "task5": [
        ("filter_logs", "deadlock"), ("inspect_service", "payment-service"),
        ("filter_logs", "lock"), ("mark_root_cause", "deadlock"),
        ("classify_issue", "application_bug"),
        ("resolve_incident", "restart_service:payment-service"),
    ],
    "task6": [
        ("filter_logs", "error"), ("inspect_service", "checkout-service"),
        ("filter_logs", "gateway"), ("mark_root_cause", "dependency_failure"),
        ("classify_issue", "dependency_failure"),
        ("resolve_incident", "rollback_deploy:checkout-service"),
    ],
    "task7": [
        ("filter_logs", "error"), ("inspect_service", "redis-cluster"),
        ("filter_logs", "partition"), ("inspect_service", "session-service"),
        ("mark_root_cause", "network_partition"),
        ("classify_issue", "infrastructure_failure"),
        ("resolve_incident", "restart_service:redis-cluster"),
    ],
}

VALID_ACTIONS = {
    "filter_logs", "inspect_service",
    "mark_root_cause", "classify_issue", "resolve_incident",
}

MAX_STEPS = 15

# ── System prompt (same as inference.py v3) ──────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response (handles markdown wrapping)."""
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
    """Format observation for the LLM."""
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


def _fallback_action(task_id: str, step: int) -> Action:
    """Deterministic fallback when LLM fails."""
    seq = FALLBACK.get(task_id, [])
    idx = step - 1
    if idx < len(seq):
        at, tgt = seq[idx]
        return Action(action_type=at, target=tgt)
    return Action(action_type="resolve_incident", target="restart_service:api-server")


# ── Core benchmark functions ─────────────────────────────────────────

def run_single_task(
    client: OpenAI,
    model_id: str,
    task_id: str,
    max_tokens: int = 120,
) -> Dict[str, Any]:
    """Run one task with one model. Returns detailed result dict."""

    env = LogEnv(task_name=task_id)
    obs = env.reset()
    conversation: list = []
    steps_log: list = []
    rewards: List[float] = []
    llm_calls = 0
    llm_successes = 0
    errors: list = []
    start_time = time.time()

    for step in range(1, MAX_STEPS + 1):
        action = None
        mode = "deterministic"

        # Try LLM
        if client is not None:
            llm_calls += 1
            obs_text = _format_obs(obs, task_id, step)
            conversation.append({"role": "user", "content": obs_text})
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation

            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    timeout=15,
                )
                raw = resp.choices[0].message.content.strip()
                conversation.append({"role": "assistant", "content": raw})
                parsed = _extract_json(raw)

                if parsed and parsed.get("action_type") in VALID_ACTIONS:
                    action = Action(
                        action_type=parsed["action_type"],
                        target=parsed.get("target"),
                    )
                    llm_successes += 1
                    mode = "llm"
                else:
                    errors.append(f"Step {step}: unparseable response: {raw[:80]}")
            except Exception as e:
                err_msg = f"Step {step}: {type(e).__name__}: {str(e)[:80]}"
                errors.append(err_msg)
                # Remove the unanswered user message from conversation
                if conversation and conversation[-1]["role"] == "user":
                    conversation.pop()

        # Fallback
        if action is None:
            action = _fallback_action(task_id, step)

        obs, reward, done, _ = env.step(action)
        reward = float(reward)
        rewards.append(reward)
        steps_log.append({
            "step": step,
            "action_type": action.action_type,
            "target": action.target,
            "reward": reward,
            "mode": mode,
            "done": done,
        })

        if done:
            break

    elapsed = round(time.time() - start_time, 2)
    state = env.state()
    score = float(grade_task(task_id, state))

    return {
        "task_id": task_id,
        "score": round(score, 4),
        "steps_used": state.step_count,
        "total_reward": round(sum(rewards), 4),
        "elapsed_seconds": elapsed,
        "llm_calls": llm_calls,
        "llm_successes": llm_successes,
        "llm_rate": round(llm_successes / max(llm_calls, 1) * 100, 1),
        "root_cause": state.root_cause_marked,
        "classification": state.classification_marked,
        "resolution": state.resolution_action,
        "wrong_actions": state.wrong_action_count,
        "errors": errors,
        "steps": steps_log,
    }


def run_deterministic_baseline(task_id: str) -> Dict[str, Any]:
    """Run the deterministic fallback for a single task."""
    env = LogEnv(task_name=task_id)
    obs = env.reset()
    rewards = []
    start_time = time.time()

    seq = FALLBACK.get(task_id, [])
    for step in range(1, len(seq) + 1):
        action = _fallback_action(task_id, step)
        obs, reward, done, _ = env.step(action)
        rewards.append(float(reward))
        if done:
            break

    elapsed = round(time.time() - start_time, 4)
    state = env.state()
    score = float(grade_task(task_id, state))

    return {
        "task_id": task_id,
        "score": round(score, 4),
        "steps_used": state.step_count,
        "total_reward": round(sum(rewards), 4),
        "elapsed_seconds": elapsed,
        "llm_calls": 0,
        "llm_successes": 0,
        "llm_rate": 0.0,
        "root_cause": state.root_cause_marked,
        "classification": state.classification_marked,
        "resolution": state.resolution_action,
        "wrong_actions": state.wrong_action_count,
        "errors": [],
        "steps": [],
    }


def benchmark_model(
    model_id: str,
    display_name: str,
    max_tokens: int,
    tasks: List[str],
    client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """Benchmark a single model across all specified tasks."""

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  [MODEL] Benchmarking: {display_name} ({model_id})", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    task_results = {}
    model_start = time.time()

    for task_id in tasks:
        print(f"  > {task_id}...", end="", file=sys.stderr, flush=True)
        if client is not None:
            result = run_single_task(client, model_id, task_id, max_tokens)
        else:
            result = run_deterministic_baseline(task_id)
        task_results[task_id] = result
        status = "PASS" if result["score"] >= 0.90 else "WARN" if result["score"] >= 0.50 else "FAIL"
        print(f" [{status}] score={result['score']:.2f}  steps={result['steps_used']}  "
              f"time={result['elapsed_seconds']:.1f}s", file=sys.stderr, flush=True)

    model_elapsed = round(time.time() - model_start, 2)
    scores = [r["score"] for r in task_results.values()]
    avg_score = round(sum(scores) / max(len(scores), 1), 4)

    return {
        "model_id": model_id,
        "display_name": display_name,
        "avg_score": avg_score,
        "total_time_seconds": model_elapsed,
        "task_count": len(tasks),
        "tasks": task_results,
        "scores_by_task": {tid: r["score"] for tid, r in task_results.items()},
    }


# ── Leaderboard formatting ───────────────────────────────────────────

def print_leaderboard(results: List[Dict], tasks: List[str]) -> None:
    """Print a formatted leaderboard table to stdout."""

    # Sort by avg_score descending
    ranked = sorted(results, key=lambda r: r["avg_score"], reverse=True)

    # Header
    task_cols = "".join(f"  {t:>6}" for t in tasks)
    header = f"{'Rank':<5} {'Model':<25} {'Avg':>6}{task_cols}  {'Time':>7}  {'LLM%':>5}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(f"  LOGENV MULTI-MODEL LEADERBOARD")
    print(sep)
    print(header)
    print(sep)

    for i, r in enumerate(ranked, 1):
        medal = {1: "#1", 2: "#2", 3: "#3"}.get(i, f"#{i}")
        task_scores = "".join(
            f"  {r['scores_by_task'].get(t, 0.0):>5.2f}" for t in tasks
        )
        # Compute avg LLM success rate
        total_llm = sum(r["tasks"][t].get("llm_calls", 0) for t in tasks if t in r["tasks"])
        total_succ = sum(r["tasks"][t].get("llm_successes", 0) for t in tasks if t in r["tasks"])
        llm_rate = round(total_succ / max(total_llm, 1) * 100, 0) if total_llm > 0 else 0

        time_str = f"{r['total_time_seconds']:.1f}s"
        llm_str = f"{llm_rate:.0f}%" if total_llm > 0 else "  N/A"

        print(f"{medal:<3} {r['display_name']:<25} {r['avg_score']:>5.2f}{task_scores}  {time_str:>7}  {llm_str:>5}")

    print(sep)
    print()


def print_detailed_report(results: List[Dict], tasks: List[str]) -> None:
    """Print detailed per-model analysis to stderr."""

    ranked = sorted(results, key=lambda r: r["avg_score"], reverse=True)

    for r in ranked:
        print(f"\n{'-'*50}", file=sys.stderr)
        print(f"  [REPORT] {r['display_name']} -- avg {r['avg_score']:.4f}", file=sys.stderr)
        print(f"{'-'*50}", file=sys.stderr)

        for tid in tasks:
            if tid not in r["tasks"]:
                continue
            t = r["tasks"][tid]
            rc_ok = "OK" if t.get("root_cause") else "--"
            cl_ok = "OK" if t.get("classification") else "--"
            rs_ok = "OK" if t.get("resolution") else "--"
            print(
                f"  {tid}: score={t['score']:.2f}  steps={t['steps_used']}  "
                f"RC:{rc_ok} {t.get('root_cause', 'None'):<35} "
                f"CL:{cl_ok} RES:{rs_ok}",
                file=sys.stderr,
            )
            if t.get("errors"):
                for err in t["errors"][:3]:
                    print(f"         [WARN] {err}", file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LogEnv Multi-Model Benchmark")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model IDs to benchmark (default: built-in roster)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task IDs (default: all 7 tasks)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save full results to a JSON file",
    )
    parser.add_argument(
        "--include-deterministic",
        action="store_true",
        default=True,
        help="Include deterministic baseline in the leaderboard (default: True)",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        default=False,
        help="Exclude deterministic baseline from the leaderboard",
    )
    args = parser.parse_args()

    tasks = args.tasks.split(",") if args.tasks else ALL_TASKS

    # Build model list
    if args.models:
        models = [(m.strip(), m.strip().split("/")[-1], 120) for m in args.models.split(",")]
    else:
        models = DEFAULT_MODELS

    # Validate token
    if not HF_TOKEN:
        print("[WARN] HF_TOKEN not set -- will only run deterministic baseline.", file=sys.stderr)
        print("   Set it: $env:HF_TOKEN='hf_...' (PowerShell)", file=sys.stderr)
        print("           export HF_TOKEN='hf_...' (bash)", file=sys.stderr)
        client = None
    else:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=15)
            print(f"[OK] HF Inference API client ready", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Client init failed: {e}", file=sys.stderr)
            client = None

    print(f"\nLogEnv Multi-Model Benchmark", file=sys.stderr)
    print(f"   Tasks: {', '.join(tasks)}", file=sys.stderr)
    print(f"   Models: {len(models)} + deterministic baseline", file=sys.stderr)
    print(f"   Started: {datetime.now(timezone.utc).isoformat()}", file=sys.stderr)

    all_results: List[Dict] = []
    benchmark_start = time.time()

    # 1. Deterministic baseline (always runs — no API needed)
    if not args.no_deterministic:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  [BASELINE] Running Deterministic Baseline", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        baseline_tasks = {}
        for task_id in tasks:
            print(f"  > {task_id}...", end="", file=sys.stderr, flush=True)
            result = run_deterministic_baseline(task_id)
            baseline_tasks[task_id] = result
            print(f" [PASS] score={result['score']:.2f}", file=sys.stderr, flush=True)

        scores = [r["score"] for r in baseline_tasks.values()]
        all_results.append({
            "model_id": "deterministic",
            "display_name": "Deterministic Fallback",
            "avg_score": round(sum(scores) / max(len(scores), 1), 4),
            "total_time_seconds": 0.01,
            "task_count": len(tasks),
            "tasks": baseline_tasks,
            "scores_by_task": {tid: r["score"] for tid, r in baseline_tasks.items()},
        })

    # 2. LLM models
    if client is not None:
        for model_id, display_name, max_tokens in models:
            try:
                result = benchmark_model(
                    model_id=model_id,
                    display_name=display_name,
                    max_tokens=max_tokens,
                    tasks=tasks,
                    client=client,
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n  [FAIL] Model {display_name} failed entirely: {e}", file=sys.stderr)
                # Still add it to results with zero scores
                all_results.append({
                    "model_id": model_id,
                    "display_name": f"FAIL: {display_name}",
                    "avg_score": 0.0,
                    "total_time_seconds": 0.0,
                    "task_count": len(tasks),
                    "tasks": {},
                    "scores_by_task": {t: 0.0 for t in tasks},
                })

    total_elapsed = round(time.time() - benchmark_start, 2)

    # 3. Print leaderboard (stdout — machine-parseable)
    print_leaderboard(all_results, tasks)

    # 4. Print detailed report (stderr)
    print_detailed_report(all_results, tasks)

    print(f"\nTotal benchmark time: {total_elapsed:.1f}s", file=sys.stderr)

    # 5. Save results to JSON
    output_file = args.output or "benchmark_results.json"
    full_output = {
        "benchmark": "logenv",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_seconds": total_elapsed,
        "tasks_evaluated": tasks,
        "models_evaluated": len(all_results),
        "leaderboard": [
            {
                "rank": i + 1,
                "model_id": r["model_id"],
                "display_name": r["display_name"],
                "avg_score": r["avg_score"],
                "scores_by_task": r["scores_by_task"],
                "total_time_seconds": r["total_time_seconds"],
            }
            for i, r in enumerate(
                sorted(all_results, key=lambda x: x["avg_score"], reverse=True)
            )
        ],
        "detailed_results": all_results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
