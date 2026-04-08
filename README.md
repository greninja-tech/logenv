---
title: LogEnv
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# 🚀 LogEnv — OpenEnv Log Analysis Environment

LogEnv is an **OpenEnv-compliant environment** designed for autonomous log analysis and incident response.

## 🧠 Features

* 3 tasks (easy, medium, hard)
* Structured observations (logs, metrics, alerts)
* Action-based interaction system
* Reward-driven evaluation
* FastAPI-based environment
* Deterministic + optional LLM policy

## ⚙️ API Endpoints

### 🔄 Reset

POST `/reset`

```json
{
  "task_id": "task1"
}
```

### ▶️ Step

POST `/step`

```json
{
  "task_id": "task1",
  "action_type": "filter_logs",
  "parameters": {
    "target": "error"
  }
}
```

### 📊 State

GET `/state/{task_id}`

### 🧮 Grade

GET `/grade/{task_id}`

## 🤖 Inference

Run locally:

```bash
python inference.py
```

Outputs:

* Structured logs
* Step-by-step actions
* Final scores

## 🐳 Deployment

* Docker-based deployment
* Compatible with Hugging Face Spaces
* Runs on port `7860`

## 📁 Structure

```
logenv/
│
├── app.py
├── inference.py
├── openenv.yaml
├── requirements.txt
├── Dockerfile
│
├── environment/
└── tests/
```

## 📌 Notes

* Works without API key (fallback policy)
* Designed for reproducibility
* Meets OpenEnv evaluation constraints

## 🎯 Status

* OpenEnv compliant ✅
* Docker ready ✅
* HF Space ready ✅
* Baseline reproducible ✅

## 🚀 Author

Developed for OpenEnv Hackathon