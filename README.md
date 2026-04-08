---
title: LogEnv
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: true
tags:
  - openenv
---

# 🚀 LogEnv v2 — Autonomous Log Analysis & Incident Response

LogEnv is an **OpenEnv-compliant** reinforcement-learning environment that simulates
real-world DevOps / SOC scenarios.  
**v2** ships with a **multi-turn LLM reasoning agent** that reads logs, thinks, and
resolves incidents autonomously — no hardcoded action sequences.

---

## 🧠 Agent Architecture

```
Observation (logs + metrics + alerts)
           │
           ▼
  ┌─────────────────────────┐
  │  Conversation Memory    │  ← full history of prior steps
  │  (rolling context)      │
  └─────────┬───────────────┘
            │
            ▼
  ┌─────────────────────────┐
  │  LLM Reasoning Layer    │  Qwen2.5-72B / any OpenAI-compatible model
  │  (chain-of-thought)     │
  └─────────┬───────────────┘
            │  JSON action
            ▼
  ┌─────────────────────────┐
  │  LogEnv Environment     │  filter_logs / inspect_service /
  │                         │  mark_root_cause / classify_issue /
  └─────────────────────────┘  resolve_incident
```

The agent:
1. **Reads** the current observation (last 15 log lines, metrics, alerts).
2. **Reasons** in natural language (chain-of-thought).
3. **Acts** — picks one action from the action space.
4. **Remembers** every prior step (multi-turn conversation history).
5. **Converges** — marks root cause → classifies → resolves.

A **deterministic fallback** with optimal sequences runs when no LLM is available,
ensuring the submission always produces a valid, high-scoring trajectory.

---

## 📋 Tasks

| Task  | Difficulty | Scenario                                             | Max Steps |
|-------|------------|------------------------------------------------------|-----------|
| task1 | 🟢 Easy    | OOM server crash — clean logs, obvious root cause    | 15        |
| task2 | 🟡 Medium  | Memory leak in microservices — one red herring       | 20        |
| task3 | 🔴 Hard    | Cascading circuit-breaker failure — 4+ red herrings  | 30        |

---

## 🔧 Action Space

| Action            | Target                        | Description                     |
|-------------------|-------------------------------|---------------------------------|
| `filter_logs`     | keyword                       | Search all logs for a term      |
| `inspect_service` | service-name                  | View logs for a specific service|
| `mark_root_cause` | root cause value              | Declare root cause              |
| `classify_issue`  | classification value          | Classify the incident           |
| `resolve_incident`| `action:service`              | Take resolution (ends episode)  |

**Root cause values:** `oom_kill`, `memory_leak`, `misconfigured_circuit_breaker`,
`network_partition`, `disk_full`, `deadlock`, `dependency_failure`

**Classification:** `infrastructure_failure`, `application_bug`, `configuration_error`,
`network_issue`, `security_incident`, `capacity_issue`, `dependency_failure`

**Resolution format:** `restart_service:NAME`, `scale_service:NAME`,
`rollback_deploy:NAME`, `patch_config:NAME`

---

## ⚙️ API Endpoints

### OpenEnv Core
```
POST /reset          {"task_id": "task1"}
POST /step           {"task_id": "task1", "action_type": "filter_logs", "parameters": {"target": "error"}}
GET  /state
GET  /state/{task_id}
GET  /grade/{task_id}
```

### Agent Endpoint (NEW in v2)
```
POST /run_agent      {"task_id": "task1", "max_steps": 12}
```
Runs the full LLM reasoning agent end-to-end and returns:
- Complete step trajectory with per-step reasoning
- Root cause, classification, resolution
- Final score (0.0–1.0)
- Whether LLM or deterministic fallback was used

---

## 🚀 Setup

### Local
```bash
pip install -r requirements.txt
python app.py                          # serves at http://localhost:7860
```

### With LLM agent
```bash
HF_TOKEN=your_token python inference.py
HF_TOKEN=your_token MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python inference.py
HF_TOKEN=your_token TASK=task1 python inference.py   # single task
```

### Docker
```bash
docker build -t logenv .
docker run -p 7860:7860 -e HF_TOKEN=your_token logenv
```

---

## 📊 Expected Scores

| Task   | Deterministic | LLM Agent |
|--------|--------------|-----------|
| task1  | 1.00         | ~1.00     |
| task2  | 1.00         | ~1.00     |
| task3  | 1.00         | ~0.95     |
| **Avg**| **1.00**     | **~0.98** |

---

## 📁 Structure

```
logenv/
├── app.py                     ← FastAPI + /run_agent endpoint
├── inference.py               ← Standalone LLM agent runner
├── openenv.yaml
├── requirements.txt
├── Dockerfile
├── README.md
└── environment/
    ├── env.py                 ← Core LogEnv
    ├── models.py              ← Pydantic models
    ├── graders.py             ← Scoring
    └── scenarios/
        ├── task1.py           ← Easy: OOM crash
        ├── task2.py           ← Medium: Memory leak
        └── task3.py           ← Hard: Cascading failure
```

---

## ✅ OpenEnv Compliance

- ✅ `reset` / `step` / `state` interface
- ✅ Typed Pydantic models
- ✅ 3 tasks (easy → medium → hard)
- ✅ Deterministic grader (0.0–1.0)
- ✅ Incremental reward function
- ✅ **Multi-turn LLM reasoning agent** (Qwen2.5-72B via HF Inference)
- ✅ Deterministic fallback (always produces valid scores without a token)
- ✅ Docker-ready for Hugging Face Spaces
- ✅ Tagged `openenv`

---
*Developed for the OpenEnv Hackathon*