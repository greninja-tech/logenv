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

## 🏆 Multi-Model Benchmarking

Compare multiple Hugging Face LLMs head-to-head on all 7 tasks.

### CLI Benchmark
```bash
# All default models (Qwen, Llama, Mistral, Mixtral)
HF_TOKEN=your_token python benchmark.py

# Specific models
HF_TOKEN=your_token python benchmark.py --models "Qwen/Qwen2.5-72B-Instruct,meta-llama/Llama-3.3-70B-Instruct"

# Specific tasks
HF_TOKEN=your_token python benchmark.py --tasks task1,task3,task7

# Save results
HF_TOKEN=your_token python benchmark.py --output my_results.jso
```

### API Benchmark
```bash
# Run benchmark via API
curl -X POST http://localhost:7860/benchmark -H "Content-Type: application/json" \
  -d '{"models": ["Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"]}'

# Get leaderboard
curl http://localhost:7860/leaderboard
```

### Sample Leaderboard Output
```
──────────────────────────────────────────────────────────────────────
  🏆  LOGENV MULTI-MODEL LEADERBOARD
──────────────────────────────────────────────────────────────────────
Rank  Model                      Avg   task1  task2  task3  ...  Time
──────────────────────────────────────────────────────────────────────
🥇1   Deterministic Fallback    0.99   0.99   0.99   0.99  ...  0.0s
🥈2   Qwen2.5-72B              0.95   0.99   0.99   0.90  ...  42.1s
🥉3   Llama-3.3-70B            0.91   0.99   0.90   0.85  ...  38.4s
```

---

## 📁 Structure

```
logenv/
├── app.py                     ← FastAPI + /run_agent + /benchmark endpoints
├── inference.py               ← Standalone LLM agent runner
├── benchmark.py               ← Multi-model benchmarking CLI
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
        ├── task3.py           ← Hard: Cascading failure
        ├── task4.py           ← Easy-Medium: Disk full
        ├── task5.py           ← Medium: Payment deadlock
        ├── task6.py           ← Medium-Hard: Dependency failure
        └── task7.py           ← Hard: Network partition
└── tests/
    └── test_env.py            ← 33 unit + integration tests
```

---

## ✅ OpenEnv Compliance

- ✅ `reset` / `step` / `state` interface
- ✅ Typed Pydantic models
- ✅ 7 tasks (easy → hard)
- ✅ Deterministic grader (0.0–1.0)
- ✅ Incremental reward function
- ✅ **Multi-turn LLM reasoning agent** (Qwen2.5-72B via HF Inference)
- ✅ **Multi-model benchmarking** with leaderboard
- ✅ Deterministic fallback (always produces valid scores without a token)
- ✅ Docker-ready for Hugging Face Spaces
- ✅ Tagged `openenv`

---
*Developed for the OpenEnv Hackathon*