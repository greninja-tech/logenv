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

An **OpenEnv-compliant** reinforcement-learning environment where AI agents diagnose
and resolve real-world production incidents — the same task that costs engineering
teams millions of dollars annually.

Agents observe system logs, metrics, and alerts, then take sequential actions to
investigate, identify root causes, classify incidents, and apply fixes — exactly
like a real on-call SRE engineer.

---

## Why this environment?

Log analysis and incident response is one of the highest-value unsolved problems
in software engineering. Every large company has dedicated SRE/NOC teams spending
hours per week on incidents that a trained agent could resolve in seconds.

This environment captures the full investigation workflow:
- **Partial observability** — agent sees only a window of logs, must filter to find signals
- **Red herrings** — deliberate misleading signals in medium/hard tasks
- **Causal chains** — root cause may be 2-3 hops from the visible symptom
- **Episode variation** — noise logs shuffled each `reset()` so agents must reason, not memorise

---

## Tasks

| Task  | Difficulty  | Root Cause                      | Key Challenge                              | Max Steps |
|-------|-------------|----------------------------------|--------------------------------------------|-----------|
| task1 | 🟢 Easy     | OOM kill                         | Clean signals, fast resolution expected    | 15        |
| task2 | 🟡 Medium   | Memory leak                      | Postgres red herring, service correlation  | 20        |
| task3 | 🔴 Hard     | Misconfigured circuit breaker    | 3 red herrings, 4+ service correlation     | 30        |
| task4 | 🟡 Easy-Med | Disk full (log rotation failure) | Trace causal chain postgres → log-rotator  | 15        |
| task5 | 🟡 Medium   | Database deadlock                | Network blip red herring                   | 20        |
| task6 | 🟠 Med-Hard | Third-party dependency failure   | Distinguish external vs internal failure   | 20        |
| task7 | 🔴 Hard     | Network partition / split brain  | 5+ service correlation, deploy red herring | 30        |

---

## Action Space

| Action             | Target                   | Reward Signal                          |
|--------------------|--------------------------|----------------------------------------|
| `filter_logs`      | keyword                  | +0.05 to +0.10 (diminishing on repeat) |
| `inspect_service`  | service-name             | +0.08 to +0.15 (first visit)           |
| `mark_root_cause`  | root cause enum          | +0.30–0.35 if correct, -0.10 if wrong  |
| `classify_issue`   | classification enum      | +0.20 if correct, -0.10 if wrong       |
| `resolve_incident` | `action_type:service`    | +0.50 if correct, partial for right svc|

**Root cause values:** `oom_kill` · `memory_leak` · `misconfigured_circuit_breaker` ·
`network_partition` · `disk_full` · `deadlock` · `dependency_failure`

**Classification values:** `infrastructure_failure` · `application_bug` · `configuration_error` ·
`network_issue` · `security_incident` · `capacity_issue` · `dependency_failure`

**Resolution format:** `restart_service:NAME` · `scale_service:NAME` ·
`rollback_deploy:NAME` · `patch_config:NAME`

---

## Observation Space

Each step returns:
- `logs` — sliding window of log entries visible to agent (timestamp, level, service, message)
- `metrics` — system metrics (CPU%, memory%, disk%, connections, request rate, error rate)
- `alerts` — triggered alerts with severity, service, message
- `step_count` — current step number

---

## Reward Function

Rewards provide **dense signal** throughout the episode:

- **Investigation quality** — correct service inspection and keyword filtering give immediate reward
- **Diminishing returns** — repeating the same keyword filter gives less reward each time
- **Investigation bonus** — inspecting the affected service before marking root cause gives +0.05 bonus
- **Red herring penalties** — task-specific penalties for chasing misleading signals
- **Efficiency bonus** — solving easy tasks quickly gives small bonus
- **Final grader score** — added to reward at episode end (0.01–0.99, never exactly 0 or 1)

---

## Episode Variation

Each `reset()` call **shuffles noise log positions** while preserving the chronological
order of WARNING/ERROR/CRITICAL logs. This means:

- The causal chain is always intact (agents can reason correctly)
- But the exact log positions vary per episode (agents cannot memorise positions)
- `reset(seed=42)` gives reproducible episodes for evaluation

---

## API Endpoints

```
POST /reset              {"task_id": "task1"}          — start episode (optional seed param)
POST /step               {"task_id": "task1", "action_type": "filter_logs", "parameters": {"target": "error"}}
GET  /state
GET  /state/{task_id}
GET  /grade/{task_id}
POST /run_agent          {"task_id": "task1", "max_steps": 15}
GET  /tasks
GET  /health
```

---

## Setup

### Local
```bash
pip install -r requirements.txt
python app.py   # http://localhost:7860/docs
```

### With LLM agent
```bash
HF_TOKEN=hf_xxx python inference.py
HF_TOKEN=hf_xxx MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python inference.py
```

### Docker
```bash
docker build -t logenv .
docker run -p 7860:7860 -e HF_TOKEN=hf_xxx logenv
```

### Run tests
```bash
pip install pytest
pytest tests/ -v
```

---

## Baseline Scores (deterministic agent)

| Task   | Score |
|--------|-------|
| task1  | 0.99  |
| task2  | 0.99  |
| task3  | 0.98  |
| task4  | 0.99  |
| task5  | 0.99  |
| task6  | 0.99  |
| task7  | 0.97  |
| **Avg**| **0.99** |

---

## Project Structure

```
logenv/
├── app.py                      ← FastAPI server + /run_agent endpoint
├── inference.py                ← LLM agent (OpenEnv stdout format)
├── openenv.yaml                ← OpenEnv metadata
├── requirements.txt
├── Dockerfile
├── README.md
├── tests/
│   └── test_env.py             ← 25+ tests covering all 7 tasks
└── environment/
    ├── env.py                  ← LogEnv (log shuffling, dense rewards)
    ├── models.py               ← Typed Pydantic models
    ├── graders.py              ← Central grading (strictly 0.01–0.99)
    └── scenarios/
        ├── task1.py            ← Easy: OOM crash
        ├── task2.py            ← Medium: Memory leak
        ├── task3.py            ← Hard: Cascading circuit breaker
        ├── task4.py            ← Easy-Med: Disk full
        ├── task5.py            ← Medium: Deadlock
        ├── task6.py            ← Med-Hard: Dependency failure
        └── task7.py            ← Hard: Network partition
```

---

## OpenEnv Compliance

- ✅ `reset()` / `step()` / `state()` interface
- ✅ Typed Pydantic models (Observation, Action, EpisodeState)
- ✅ 7 tasks ranging easy → hard
- ✅ Deterministic graders (0.01–0.99, strictly open interval)
- ✅ Dense reward function with partial progress signals
- ✅ Red herring penalties — task-specific
- ✅ Episode variation via log shuffling (reproducible with seed)
- ✅ Multi-turn LLM reasoning agent (Qwen2.5-72B via HF Router)
- ✅ Deterministic fallback policy (valid scores without HF_TOKEN)
- ✅ Docker-ready for Hugging Face Spaces
- ✅ 25+ unit and integration tests

---
*Developed for the OpenEnv Hackathon — Meta PyTorch × Scaler School of Technology*
