---
title: LogEnv
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
tags:
  - openenv
  - devops
  - incident-response
  - log-analysis
---

# 🚀 LogEnv — Autonomous Log Analysis & Incident Response Environment

LogEnv is an **OpenEnv-compliant** reinforcement learning environment that simulates real-world DevOps and Security Operations Center (SOC) scenarios. An agent must analyze system logs, identify root causes, classify incidents, and take corrective actions.

## 🧠 Why LogEnv?

Log analysis and incident response is a high-value, real-world task that costs companies millions of dollars annually. This environment captures the full investigation workflow:

- **Read logs** → filter by keyword or service
- **Analyze patterns** → identify anomalies across multiple services  
- **Make decisions** → mark root cause, classify incident type
- **Resolve** → take the correct remediation action

## 📋 Tasks

| Task | Difficulty | Description | Max Steps |
|------|------------|-------------|-----------|
| task1 | 🟢 Easy | Simple OOM server crash — clean logs, obvious root cause | 15 |
| task2 | 🟡 Medium | Memory leak in microservices — multiple services, one red herring | 20 |
| task3 | 🔴 Hard | Cascading failure from misconfigured circuit breaker — 4+ services, multiple red herrings | 30 |

## 🔧 Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `filter_logs` | `target: keyword` | Search all logs for a keyword |
| `inspect_service` | `target: service-name` | View all logs for a specific service |
| `mark_root_cause` | `target: cause` | Declare identified root cause |
| `classify_issue` | `target: classification` | Classify the incident type |
| `resolve_incident` | `target: resolution` | Take resolution action (ends episode) |

**Root cause values:** `oom_kill`, `memory_leak`, `misconfigured_circuit_breaker`, `network_partition`, `disk_full`, `deadlock`, `dependency_failure`

**Classification values:** `infrastructure_failure`, `application_bug`, `configuration_error`, `network_issue`, `security_incident`, `capacity_issue`, `dependency_failure`

**Resolution format:** `restart_service:NAME`, `scale_service:NAME`, `rollback_deploy:NAME`, `patch_config:NAME`

## 👁️ Observation Space

Each step returns:
- `logs`: visible log entries (timestamp, level, service, message)
- `metrics`: CPU%, memory%, disk%, active_connections, request_rate, error_rate
- `alerts`: triggered alerts with severity, service, message
- `step_count`: current step number

## 🏆 Reward Function

Rewards are given incrementally throughout the episode:

| Action | Reward |
|--------|--------|
| Correct root cause identification | +0.3 |
| Correct classification | +0.2 |
| Correct resolution | +0.5 |
| Relevant log filter (memory, oom, circuit, error, heap) | +0.1 |
| Inspecting a new service | +0.1 |
| Wrong/unknown action | -0.1 |
| Destructive resolution action | -0.2 |
| Final grader score (0.0–1.0) | added on episode end |

## ⚙️ API Endpoints

### Reset
```
POST /reset
{"task_id": "task1"}
```

### Step
```
POST /step
{
  "task_id": "task1",
  "action_type": "filter_logs",
  "parameters": {"target": "error"}
}
```

### State
```
GET /state
GET /state/{task_id}
```

### Grade
```
GET /grade/{task_id}
```

## 📊 Baseline Performance

| Task | Baseline Score |
|------|---------------|
| task1 (Easy) | 0.72 |
| task2 (Medium) | 0.55 |
| task3 (Hard) | 0.38 |
| **Average** | **0.55** |

## 🚀 Setup & Usage

### Local Development

```bash
pip install -r requirements.txt
python app.py
# API available at http://localhost:7860/docs
```

### Docker

```bash
docker build -t logenv .
docker run -p 7860:7860 logenv
```

### Inference Script

```bash
HF_TOKEN=your_token python inference.py
# With a specific model:
HF_TOKEN=your_token MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python inference.py
```

## 📁 Structure

```
logenv/
├── app.py                    ← FastAPI application
├── inference.py              ← LLM agent + baseline evaluation
├── openenv.yaml              ← OpenEnv metadata
├── requirements.txt
├── Dockerfile
├── README.md
└── environment/
    ├── __init__.py
    ├── env.py                ← Core LogEnv class
    ├── models.py             ← Pydantic models
    ├── graders.py            ← Scoring logic
    └── scenarios/
        ├── __init__.py
        ├── task1.py          ← Easy: OOM crash
        ├── task2.py          ← Medium: Memory leak
        └── task3.py          ← Hard: Cascading failure
```

## ✅ OpenEnv Compliance

- ✅ OpenEnv-compliant interface (`reset`, `step`, `state`)
- ✅ Typed Pydantic models for Observation, Action, State
- ✅ 3 tasks (easy → medium → hard)
- ✅ Deterministic, programmatic grader (0.0–1.0)
- ✅ Incremental reward function
- ✅ Baseline inference script using OpenAI client + HF_TOKEN
- ✅ Docker-ready for Hugging Face Spaces
- ✅ Tagged `openenv`

---
*Developed for the OpenEnv Hackathon*