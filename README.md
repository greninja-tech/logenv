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

# 🚀 LogEnv — Autonomous Log Analysis & Incident Response

LogEnv is an **OpenEnv-compliant** reinforcement-learning environment simulating
real-world DevOps / SOC incident response scenarios.

An LLM agent reads system logs, metrics, and alerts — then takes sequential actions
to investigate, identify root causes, classify incidents, and apply fixes.

---

## 🧠 Agent Architecture

```
Observation (logs + metrics + alerts)
           │
           ▼
  ┌─────────────────────────┐
  │  Conversation Memory    │  ← full history of prior steps
  └─────────┬───────────────┘
            │
            ▼
  ┌─────────────────────────┐
  │  LLM Reasoning Layer    │  Qwen2.5-72B via HF Inference
  └─────────┬───────────────┘
            │  JSON action
            ▼
  ┌─────────────────────────┐
  │  LogEnv Environment     │  filter_logs / inspect_service /
  │                         │  mark_root_cause / classify_issue /
  └─────────────────────────┘  resolve_incident
```

---

## 📋 Tasks

| Task  | Difficulty    | Scenario                                              | Max Steps |
|-------|---------------|-------------------------------------------------------| ----------|
| task1 | 🟢 Easy       | OOM server crash — clean logs, obvious root cause     | 15        |
| task2 | 🟡 Medium     | Memory leak in microservices — one red herring        | 20        |
| task3 | 🔴 Hard       | Cascading circuit-breaker failure — 4+ red herrings   | 30        |
| task4 | 🟡 Easy-Med   | Disk full — log rotation daemon fails silently        | 15        |
| task5 | 🟡 Medium     | Payment service deadlock — red herring network blip   | 20        |
| task6 | 🟠 Med-Hard   | Third-party dependency failure — bad deploy           | 20        |
| task7 | 🔴 Hard       | Network partition / split brain in Redis cluster      | 30        |

---

## 🔧 Action Space

| Action            | Target                        | Description                      |
|-------------------|-------------------------------|----------------------------------|
| `filter_logs`     | keyword                       | Search all logs for a term       |
| `inspect_service` | service-name                  | View logs for a specific service |
| `mark_root_cause` | root cause value              | Declare root cause               |
| `classify_issue`  | classification value          | Classify the incident            |
| `resolve_incident`| `action:service`              | Take resolution (ends episode)   |

**Root cause values:** `oom_kill`, `memory_leak`, `misconfigured_circuit_breaker`,
`network_partition`, `disk_full`, `deadlock`, `dependency_failure`

**Classification:** `infrastructure_failure`, `application_bug`, `configuration_error`,
`network_issue`, `security_incident`, `capacity_issue`, `dependency_failure`

**Resolution format:** `restart_service:NAME`, `scale_service:NAME`,
`rollback_deploy:NAME`, `patch_config:NAME`

---

## ⚙️ API Endpoints

```
POST /reset          {"task_id": "task1"}
POST /step           {"task_id": "task1", "action_type": "filter_logs", "parameters": {"target": "error"}}
GET  /state
GET  /state/{task_id}
GET  /grade/{task_id}
POST /run_agent      {"task_id": "task1", "max_steps": 12}
GET  /health
GET  /tasks
```

---

## 🚀 Setup

### Local
```bash
pip install -r requirements.txt
python app.py   # serves at http://localhost:7860
```

### With LLM agent
```bash
HF_TOKEN=your_token python inference.py
HF_TOKEN=your_token MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python inference.py
```

### Docker
```bash
docker build -t logenv .
docker run -p 7860:7860 -e HF_TOKEN=your_token logenv
```

---

## 📊 Baseline Scores

| Task   | Score  |
|--------|--------|
| task1  | 0.99   |
| task2  | 0.99   |
| task3  | 0.99   |
| task4  | 0.99   |
| task5  | 0.99   |
| task6  | 0.99   |
| task7  | 0.99   |
| **Avg**| **0.99** |

---

## 📁 Structure

```
logenv/
├── app.py                     ← FastAPI server
├── inference.py               ← LLM agent runner (OpenEnv sample format)
├── openenv.yaml               ← OpenEnv metadata
├── requirements.txt
├── Dockerfile
├── README.md
└── environment/
    ├── env.py                 ← Core LogEnv
    ├── models.py              ← Pydantic models
    ├── graders.py             ← Scoring (strictly 0.01–0.99)
    └── scenarios/
        ├── task1.py           ← Easy: OOM crash
        ├── task2.py           ← Medium: Memory leak
        ├── task3.py           ← Hard: Cascading failure
        ├── task4.py           ← Easy-Med: Disk full
        ├── task5.py           ← Medium: Deadlock
        ├── task6.py           ← Med-Hard: Dependency failure
        └── task7.py           ← Hard: Network partition
```

---

## ✅ OpenEnv Compliance

- ✅ `reset` / `step` / `state` interface
- ✅ Typed Pydantic models
- ✅ 7 tasks (easy → hard)
- ✅ Deterministic grader (scores strictly in (0, 1))
- ✅ Incremental reward function
- ✅ Multi-turn LLM reasoning agent (Qwen2.5-72B via HF Inference)
- ✅ Deterministic fallback (always produces valid scores without HF_TOKEN)
- ✅ Docker-ready for Hugging Face Spaces
- ✅ Tagged `openenv`

---
*Developed for the OpenEnv Hackathon*