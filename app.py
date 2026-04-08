import os
import json
import re
import sys
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from environment import LogEnv
from environment.models import Action
from environment.graders import grade_task


# ──────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────

app = FastAPI(
    title="LogEnv — Log Analysis & Incident Response",
    description=(
        "OpenEnv-compliant environment for autonomous log analysis and incident response.\n\n"
        "**Quick start:**\n"
        "1. `POST /reset` with `{\"task_id\": \"task1\"}` to start\n"
        "2. `POST /step` with action to interact\n"
        "3. `GET /state` to inspect state\n"
        "4. `GET /grade/{task_id}` to get score\n"
        "5. `POST /run_agent` to run the full LLM agent on a task"
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_envs: Dict[str, LogEnv] = {}
_current_task_id: str = "task1"


def get_env(task_id: str) -> LogEnv:
    if task_id not in _envs:
        _envs[task_id] = LogEnv(task_name=task_id)
    return _envs[task_id]


# ──────────────────────────────────────────────
# LLM AGENT  (inline — same logic as inference.py)
# ──────────────────────────────────────────────

_llm_client = None

try:
    from openai import OpenAI
    api_key = os.environ.get("HF_TOKEN")
    if api_key:
        _llm_client = OpenAI(
            base_url=os.environ.get("API_BASE_URL",
                                    "https://api-inference.huggingface.co/v1"),
            api_key=api_key,
        )
except Exception:
    pass

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

SYSTEM_PROMPT = """You are an expert DevOps/SRE incident-response agent.
Analyse logs, metrics and alerts; take actions to find and resolve the root cause.

ACTIONS
-------
filter_logs      target=keyword         Search all logs for a word
inspect_service  target=service-name    View logs for one service
mark_root_cause  target=<value>         Declare root cause
classify_issue   target=<value>         Classify incident type
resolve_incident target=<value>         Resolve (ends episode)

ROOT CAUSE: oom_kill | memory_leak | misconfigured_circuit_breaker |
            network_partition | disk_full | deadlock | dependency_failure
CLASSIFICATION: infrastructure_failure | application_bug | configuration_error |
                network_issue | security_incident | capacity_issue | dependency_failure
RESOLUTION: restart_service:NAME | scale_service:NAME | rollback_deploy:NAME | patch_config:NAME

RULES
-----
- Investigate first. Mark root cause + classify before resolving.
- Do not repeat identical actions.
- Beware red herrings — trace to the origin, not symptoms.

REPLY FORMAT  (always):
Reasoning: <1-2 sentences>
```json
{"action_type": "...", "target": "..."}
```"""

def _extract_json(text: str):
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None

def _format_obs(obs, step: int, task_id: str, task_desc: str = "") -> str:
    lines = [f"Step {step} | Task {task_id}"]
    if step == 1 and task_desc:
        lines += ["", "TASK:", task_desc]
    lines += ["", "LOGS:"]
    for log in obs.logs[-15:]:
        lines.append(f"  {log.timestamp[11:19]} [{log.level}] {log.service}: {log.message}")
    lines += [
        "",
        f"METRICS: CPU={obs.metrics.cpu_percent:.0f}% MEM={obs.metrics.memory_percent:.0f}%"
        f" Errors={obs.metrics.error_rate:.0f}%",
        "",
        "ALERTS:",
    ]
    for a in obs.alerts:
        lines.append(f"  [{a.severity}] {a.service}: {a.message}")
    lines.append("\nNext action?")
    return "\n".join(lines)

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

def _fallback(task_id: str, step: int) -> Action:
    seq = _FALLBACK.get(task_id, [])
    idx = step - 1
    return seq[idx] if idx < len(seq) else Action(action_type="resolve_incident", target="restart_service:api-server")


# ──────────────────────────────────────────────
# REQUEST MODELS
# ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task1"

class StepRequest(BaseModel):
    task_id: str = "task1"
    action_type: str
    parameters: Dict[str, Any] = {}

class AgentRunRequest(BaseModel):
    task_id: str = "task1"
    max_steps: int = 12


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html><head><title>LogEnv v2</title>
    <style>body{font-family:sans-serif;max-width:860px;margin:40px auto;padding:20px}
    code{background:#f4f4f4;padding:2px 6px;border-radius:3px}
    .badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.8em;font-weight:bold}
    .easy{background:#d4edda;color:#155724}.medium{background:#fff3cd;color:#856404}.hard{background:#f8d7da;color:#721c24}
    </style></head>
    <body>
    <h1>🚀 LogEnv v2 — Intelligent Incident Response Agent</h1>
    <p>OpenEnv-compliant environment + LLM reasoning agent for autonomous DevOps/SOC tasks.</p>
    <h3>Tasks</h3>
    <ul>
      <li><b>task1</b> <span class="badge easy">Easy</span> — OOM Server Crash</li>
      <li><b>task2</b> <span class="badge medium">Medium</span> — Memory Leak in Microservices</li>
      <li><b>task3</b> <span class="badge hard">Hard</span> — Cascading Circuit Breaker Failure</li>
    </ul>
    <h3>API</h3>
    <ul>
      <li><code>POST /reset</code> — Start / restart a task</li>
      <li><code>POST /step</code> — Take a manual action</li>
      <li><code>GET /state</code> — Full episode state</li>
      <li><code>GET /grade/{task_id}</code> — Current score</li>
      <li><code>POST /run_agent</code> — <b>Run the LLM reasoning agent end-to-end</b></li>
    </ul>
    <p><a href="/docs">📖 Swagger UI</a></p>
    </body></html>
    """


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "llm_available": _llm_client is not None,
        "model": MODEL_NAME,
    }


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {"task_id": "task1", "name": "Simple Server Crash",           "difficulty": "easy",   "max_steps": 15},
            {"task_id": "task2", "name": "Memory Leak in Microservices",  "difficulty": "medium", "max_steps": 20},
            {"task_id": "task3", "name": "Distributed Cascading Failure", "difficulty": "hard",   "max_steps": 30},
        ]
    }


# ── Core OpenEnv endpoints ───────────────────

@app.post("/reset")
async def reset(req: ResetRequest):
    global _current_task_id
    _current_task_id = req.task_id
    _envs[req.task_id] = LogEnv(task_name=req.task_id)
    obs = _envs[req.task_id].reset()
    return obs.model_dump()


@app.post("/step")
async def step(req: StepRequest):
    env = get_env(req.task_id)
    if env.state_data is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first.")
    action = Action(action_type=req.action_type, target=req.parameters.get("target"))
    obs, reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}


@app.get("/state")
async def get_state():
    env = get_env(_current_task_id)
    if env.state_data is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first.")
    return env.state().model_dump()


@app.get("/state/{task_id}")
async def get_state_by_task(task_id: str):
    env = get_env(task_id)
    if env.state_data is None:
        raise HTTPException(status_code=400, detail=f"No session for {task_id}. Reset first.")
    return env.state().model_dump()


@app.get("/grade/{task_id}")
async def get_grade(task_id: str):
    env = get_env(task_id)
    if env.state_data is None:
        raise HTTPException(status_code=400, detail=f"No session for {task_id}. Reset first.")
    state = env.state()
    score = grade_task(task_id, state)
    return {
        "task_id": task_id,
        "score": score,
        "step_count": state.step_count,
        "root_cause_marked": state.root_cause_marked,
        "classification_marked": state.classification_marked,
        "resolution_action": state.resolution_action,
        "wrong_action_count": state.wrong_action_count,
    }


# ── Agent endpoint ───────────────────────────

@app.post("/run_agent")
async def run_agent(req: AgentRunRequest):
    """
    Run the full LLM reasoning agent on a task end-to-end.
    Returns the complete trajectory, all reasoning, and the final score.
    Falls back to optimal deterministic policy if HF_TOKEN is not set.
    """
    task_id   = req.task_id
    max_steps = req.max_steps

    # Fresh environment
    env = LogEnv(task_name=task_id)
    obs = env.reset()
    task_desc = (env.task_data or {}).get("task_description", "")

    history    = []
    trajectory = []
    done       = False
    using_llm  = (_llm_client is not None)

    for step in range(1, max_steps + 1):
        action   = None
        reasoning = ""
        mode     = "deterministic"

        if using_llm:
            # Build messages with full conversation history
            user_msg = _format_obs(obs, step, task_id, task_desc if step == 1 else "")
            history.append({"role": "user", "content": user_msg})
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

            try:
                response = _llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=320,
                    temperature=0.1,
                )
                reply = response.choices[0].message.content.strip()
                history.append({"role": "assistant", "content": reply})

                parsed = _extract_json(reply)
                if parsed:
                    action = Action(
                        action_type=parsed.get("action_type", "filter_logs"),
                        target=parsed.get("target"),
                    )
                    mode = "llm"
                    # Extract first reasoning line
                    for line in reply.split("\n"):
                        line = line.strip()
                        if line and not line.startswith("{") and not line.startswith("```"):
                            reasoning = line[:200]
                            break
            except Exception as e:
                reasoning = f"LLM error: {e}"

        if action is None:
            action = _fallback(task_id, step)
            mode   = "deterministic"

        obs, reward, done, _ = env.step(action)

        trajectory.append({
            "step":        step,
            "mode":        mode,
            "reasoning":   reasoning,
            "action_type": action.action_type,
            "target":      action.target,
            "reward":      reward,
            "done":        done,
        })

        if done:
            break

    state       = env.state()
    final_score = grade_task(task_id, state)

    return {
        "task_id":              task_id,
        "llm_used":             using_llm,
        "model":                MODEL_NAME if using_llm else "deterministic",
        "final_score":          final_score,
        "steps_used":           state.step_count,
        "root_cause_marked":    state.root_cause_marked,
        "classification_marked":state.classification_marked,
        "resolution_action":    state.resolution_action,
        "wrong_action_count":   state.wrong_action_count,
        "trajectory":           trajectory,
    }


# ──────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
