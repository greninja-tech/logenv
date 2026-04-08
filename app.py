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
    title="Log Analysis & Incident Response Environment",
    description="OpenEnv-compliant environment for autonomous log analysis.",
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

# Track current active task (IMPORTANT for /state)
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
    <h1>🚀 Log Analysis Environment</h1>
    <p>Use /docs to interact with the API</p>
    """


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/tasks")
async def tasks():
    return {
        "tasks": [
            {"task_id": "task1", "difficulty": "easy"},
            {"task_id": "task2", "difficulty": "medium"},
            {"task_id": "task3", "difficulty": "hard"},
        ]
    }


# ---------------- CORE OPENENV ENDPOINTS ----------------

@app.post("/reset")
async def reset(req: ResetRequest):
    global _current_task_id

    _current_task_id = req.task_id  # track active task

    env = get_env(req.task_id)
    obs = env.reset()

    return obs.model_dump()


@app.post("/step")
async def step(req: StepRequest):
    env = get_env(req.task_id)

    if env.state_data is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

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


# 🔥 FIXED: OpenEnv expects /state (NOT /state/{task_id})
@app.get("/state")
async def get_state():
    env = get_env(_current_task_id)

    if env.state_data is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    return env.state().model_dump()


# ---------------- OPTIONAL (NOT REQUIRED BUT GOOD) ----------------

@app.get("/grade/{task_id}")
async def get_grade(task_id: str):
    env = get_env(task_id)

    if env.state_data is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    state = env.state()
    score = grade_task(task_id, state)

    return {
        "task_id": task_id,
        "score": score,
        "step_count": state.step_count,
    }


# ---------------- RUN ----------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)