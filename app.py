import os
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from environment import LogEnv
from environment.models import Action
from environment.graders import grade_task


# ---------------- APP SETUP ----------------

app = FastAPI(
    title="LogEnv — Log Analysis & Incident Response",
    description=(
        "OpenEnv-compliant environment for autonomous log analysis and incident response. "
        "Simulates real-world DevOps/SOC scenarios.\n\n"
        "**Quick start:**\n"
        "1. `POST /reset` with `{\"task_id\": \"task1\"}` to start\n"
        "2. `POST /step` with action to interact\n"
        "3. `GET /state` to inspect state\n"
        "4. `GET /grade/{task_id}` to get current score"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store environments per task
_envs: Dict[str, LogEnv] = {}

# Track current active task (for /state)
_current_task_id: str = "task1"


def get_env(task_id: str) -> LogEnv:
    if task_id not in _envs:
        _envs[task_id] = LogEnv(task_name=task_id)
    return _envs[task_id]


# ---------------- REQUEST MODELS ----------------

class ResetRequest(BaseModel):
    task_id: str = "task1"


class StepRequest(BaseModel):
    task_id: str = "task1"
    action_type: str
    parameters: Dict[str, Any] = {}


# ---------------- ROUTES ----------------

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html><head><title>LogEnv</title></head>
    <body style="font-family:sans-serif;max-width:800px;margin:40px auto;padding:20px">
    <h1>🚀 LogEnv — Log Analysis & Incident Response</h1>
    <p>OpenEnv-compliant environment for autonomous DevOps/SOC agents.</p>
    <h3>Tasks</h3>
    <ul>
      <li><b>task1</b> (Easy) — Simple Server OOM Crash</li>
      <li><b>task2</b> (Medium) — Memory Leak in Microservices</li>
      <li><b>task3</b> (Hard) — Distributed Cascading Failure</li>
    </ul>
    <h3>Actions</h3>
    <ul>
      <li><code>filter_logs</code> — Search logs by keyword</li>
      <li><code>inspect_service</code> — View logs for a service</li>
      <li><code>mark_root_cause</code> — Identify root cause</li>
      <li><code>classify_issue</code> — Classify the incident</li>
      <li><code>resolve_incident</code> — Take resolution action (ends episode)</li>
    </ul>
    <p><a href="/docs">📖 API Docs (Swagger UI)</a></p>
    </body></html>
    """


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "task_id": "task1",
                "name": "Simple Server Crash",
                "difficulty": "easy",
                "max_steps": 15,
                "description": "A web server crashes due to OOM kill. Investigate and resolve."
            },
            {
                "task_id": "task2",
                "name": "Memory Leak in Microservices",
                "difficulty": "medium",
                "max_steps": 20,
                "description": "Memory leak in session-manager causes gradual degradation."
            },
            {
                "task_id": "task3",
                "name": "Distributed Cascading Failure",
                "difficulty": "hard",
                "max_steps": 30,
                "description": "Misconfigured circuit breaker causes cascading failure. Multiple red herrings."
            },
        ]
    }


# ---------------- CORE OPENENV ENDPOINTS ----------------

@app.post("/reset")
async def reset(req: ResetRequest):
    """
    Reset the environment for a given task.
    Returns initial observation (first 5 log entries, metrics, alerts).
    """
    global _current_task_id
    _current_task_id = req.task_id

    # Force fresh environment on reset
    _envs[req.task_id] = LogEnv(task_name=req.task_id)
    env = _envs[req.task_id]
    obs = env.reset()

    return obs.model_dump()


@app.post("/step")
async def step(req: StepRequest):
    """
    Take an action in the environment.

    Available action_types:
    - filter_logs: {"target": "keyword"} — filter logs by keyword
    - inspect_service: {"target": "service-name"} — view service logs
    - mark_root_cause: {"target": "oom_kill|memory_leak|misconfigured_circuit_breaker|..."}
    - classify_issue: {"target": "infrastructure_failure|application_bug|configuration_error|..."}
    - resolve_incident: {"target": "restart_service:NAME|scale_service:NAME"} — ends episode
    """
    env = get_env(req.task_id)

    if env.state_data is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first to initialize the environment.")

    action = Action(
        action_type=req.action_type,
        target=req.parameters.get("target")
    )

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
async def get_state():
    """
    Get the full current state of the active task environment.
    The active task is set by the most recent /reset call.
    """
    env = get_env(_current_task_id)

    if env.state_data is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first.")

    return env.state().model_dump()


@app.get("/state/{task_id}")
async def get_state_by_task(task_id: str):
    """Get the full current state for a specific task."""
    env = get_env(task_id)

    if env.state_data is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active session for {task_id}. Call POST /reset with task_id={task_id} first."
        )

    return env.state().model_dump()


@app.get("/grade/{task_id}")
async def get_grade(task_id: str):
    """
    Get the current grade for a task based on the actions taken so far.
    Score is 0.0-1.0. For best results, call after resolve_incident.
    """
    env = get_env(task_id)

    if env.state_data is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active session for {task_id}. Call POST /reset first."
        )

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


# ---------------- RUN ----------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)