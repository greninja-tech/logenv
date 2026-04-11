"""
Task 1 (Easy): Simple Server Crash
A web server crashes due to Out-of-Memory (OOM). 
The logs are clean and the root cause is obvious.
Ground truth: OOM kill on the 'api-server' service.
"""

from ..models import LogEntry, SystemMetrics, Alert, EpisodeState
from typing import Dict, Any


GROUND_TRUTH = {
    "root_cause": "oom_kill",
    "classification": "infrastructure_failure",
    "resolution": "restart_service:api-server",
    "affected_service": "api-server",
}

TASK_DESCRIPTION = (
    "TASK 1 (Easy) — Server Crash Investigation\n"
    "Your production API server has gone down. Users are reporting 502 errors. "
    "Investigate the logs to identify the root cause, classify the incident, "
    "and take the correct resolution action.\n"
    "Services: api-server, nginx, postgres, redis"
)

ALL_LOGS = [
    LogEntry(timestamp="2024-01-15T10:00:01Z", level="INFO",    service="nginx",      message="Starting nginx/1.24.0"),
    LogEntry(timestamp="2024-01-15T10:00:02Z", level="INFO",    service="api-server", message="Application server started on port 8080"),
    LogEntry(timestamp="2024-01-15T10:00:03Z", level="INFO",    service="postgres",   message="Database connection pool initialized (size=20)"),
    LogEntry(timestamp="2024-01-15T10:00:04Z", level="INFO",    service="redis",      message="Redis cache connected at localhost:6379"),
    LogEntry(timestamp="2024-01-15T10:01:00Z", level="INFO",    service="api-server", message="Handling 120 req/s — all healthy"),
    LogEntry(timestamp="2024-01-15T10:02:00Z", level="INFO",    service="api-server", message="Handling 135 req/s — memory usage: 512MB"),
    LogEntry(timestamp="2024-01-15T10:03:00Z", level="INFO",    service="api-server", message="Handling 140 req/s — memory usage: 1.1GB"),
    LogEntry(timestamp="2024-01-15T10:04:00Z", level="WARNING", service="api-server", message="Memory usage elevated: 1.8GB / 2.0GB limit"),
    LogEntry(timestamp="2024-01-15T10:04:10Z", level="WARNING", service="api-server", message="GC pressure increasing — full GC triggered"),
    LogEntry(timestamp="2024-01-15T10:04:20Z", level="WARNING", service="api-server", message="Memory usage: 1.95GB — approaching limit"),
    LogEntry(timestamp="2024-01-15T10:04:30Z", level="ERROR",   service="api-server", message="java.lang.OutOfMemoryError: Java heap space"),
    LogEntry(timestamp="2024-01-15T10:04:31Z", level="CRITICAL",service="api-server", message="OOM Kill signal received — process terminated (PID 4821)"),
    LogEntry(timestamp="2024-01-15T10:04:31Z", level="ERROR",   service="nginx",      message="upstream api-server unavailable — returning 502"),
    LogEntry(timestamp="2024-01-15T10:04:32Z", level="CRITICAL",service="nginx",      message="502 Bad Gateway — all upstream hosts down"),
    LogEntry(timestamp="2024-01-15T10:04:33Z", level="INFO",    service="postgres",   message="Idle — awaiting connections"),
    LogEntry(timestamp="2024-01-15T10:04:33Z", level="INFO",    service="redis",      message="Connected — 0 active clients"),
    LogEntry(timestamp="2024-01-15T10:04:45Z", level="ERROR",   service="nginx",      message="Health check failed for api-server (timeout 5s)"),
    LogEntry(timestamp="2024-01-15T10:04:50Z", level="ERROR",   service="nginx",      message="Health check failed for api-server (timeout 5s)"),
    LogEntry(timestamp="2024-01-15T10:05:00Z", level="CRITICAL",service="nginx",      message="Service api-server marked DOWN after 3 consecutive failures"),
]

METRICS = SystemMetrics(
    cpu_percent=12.0,
    memory_percent=98.5,
    disk_percent=45.0,
    active_connections=0,
    request_rate=0.0,
    error_rate=100.0,
)

ALERTS = [
    Alert(alert_id="ALT-001", severity="CRITICAL", service="api-server",
          message="Process OOM killed — PID 4821 terminated", triggered_at="2024-01-15T10:04:31Z"),
    Alert(alert_id="ALT-002", severity="HIGH",     service="nginx",
          message="All upstream hosts unavailable — 502 rate 100%", triggered_at="2024-01-15T10:04:32Z"),
]


def get_scenario() -> Dict[str, Any]:
    return {
        "task_id": "task1",
        "scenario_name": "simple_server_crash",
        "task_description": TASK_DESCRIPTION,
        "all_logs": ALL_LOGS,
        "metrics": METRICS,
        "alerts": ALERTS,
        "ground_truth": GROUND_TRUTH,
        "max_steps": 15,
        "difficulty": "easy",
    }



def grade(state: EpisodeState) -> float:
    """Task 1 — Easy: OOM Crash. Clean signals, fast resolution expected."""
    score = 0.0
    gt = GROUND_TRUTH

    if state.root_cause_marked == gt["root_cause"]:
        score += 0.30
    if state.classification_marked == gt["classification"]:
        score += 0.20
    if state.resolution_action == gt["resolution"]:
        score += 0.40
    elif state.resolution_action and "api-server" in state.resolution_action:
        score += 0.15  # right service, wrong action type

    # Efficiency bonus: this is an easy task — slow resolution is penalised
    if state.step_count <= 5 and score >= 0.90:
        score += 0.10  # very fast
    elif state.step_count <= 8 and score >= 0.70:
        score += 0.05

    score -= 0.05 * state.wrong_action_count
    score -= 0.10 * state.destructive_action_count

    return round(max(0.01, min(0.99, score)), 4)
