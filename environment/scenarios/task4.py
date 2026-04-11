"""
Task 4 (Easy-Medium): Disk Full — Log Rotation Failure
The logging daemon fills the disk, causing the database to refuse new writes.
Logs are clear but the causal chain spans two services.
Ground truth: disk_full on postgres; fix = restart_service:log-rotator
"""

from ..models import LogEntry, SystemMetrics, Alert, EpisodeState
from typing import Dict, Any


GROUND_TRUTH = {
    "root_cause": "disk_full",
    "classification": "infrastructure_failure",
    "resolution": "restart_service:log-rotator",
    "affected_service": "log-rotator",
}

TASK_DESCRIPTION = (
    "TASK 4 (Easy-Medium) — Disk Full Investigation\n"
    "The database has started refusing writes. Storage alerts are firing. "
    "Trace the cause — something is consuming disk space abnormally fast. "
    "Find the culprit service, classify the failure, and resolve it.\n"
    "Services: postgres, log-rotator, api-server, nginx, redis"
)

ALL_LOGS = [
    LogEntry(timestamp="2024-02-10T08:00:00Z", level="INFO",    service="api-server",   message="Server started — all systems nominal"),
    LogEntry(timestamp="2024-02-10T08:00:01Z", level="INFO",    service="postgres",     message="Database ready — accepting connections"),
    LogEntry(timestamp="2024-02-10T08:00:02Z", level="INFO",    service="log-rotator",  message="Log rotation daemon started — interval: 1h"),
    LogEntry(timestamp="2024-02-10T08:30:00Z", level="WARNING", service="log-rotator",  message="Rotation failed: /var/log/app — file locked by api-server"),
    LogEntry(timestamp="2024-02-10T09:00:00Z", level="WARNING", service="log-rotator",  message="Rotation failed again: skipping — will retry in 1h"),
    LogEntry(timestamp="2024-02-10T09:30:00Z", level="WARNING", service="log-rotator",  message="Rotation failed: /var/log/app — file locked by api-server"),
    LogEntry(timestamp="2024-02-10T10:00:00Z", level="ERROR",   service="log-rotator",  message="CRITICAL: Log rotation has failed 3 consecutive times — disk usage growing"),
    LogEntry(timestamp="2024-02-10T10:01:00Z", level="WARNING", service="postgres",     message="Disk usage at 85% — approaching threshold"),
    LogEntry(timestamp="2024-02-10T10:15:00Z", level="WARNING", service="postgres",     message="Disk usage at 92% — write performance degrading"),
    LogEntry(timestamp="2024-02-10T10:28:00Z", level="ERROR",   service="log-rotator",  message="Cannot rotate: disk at 97% — /var/log/app is 45GB"),
    LogEntry(timestamp="2024-02-10T10:30:00Z", level="CRITICAL",service="postgres",     message="Disk full — refusing all write operations (ENOSPC)"),
    LogEntry(timestamp="2024-02-10T10:30:01Z", level="ERROR",   service="api-server",   message="Database write failed: ENOSPC — no space left on device"),
    LogEntry(timestamp="2024-02-10T10:30:02Z", level="ERROR",   service="api-server",   message="Transaction rollback — 503 responses starting"),
    LogEntry(timestamp="2024-02-10T10:30:05Z", level="CRITICAL",service="nginx",        message="Upstream api-server returning 503 — health check failed"),
    LogEntry(timestamp="2024-02-10T10:30:10Z", level="INFO",    service="redis",        message="Cache hit rate 100% — serving stale data"),
    LogEntry(timestamp="2024-02-10T10:31:00Z", level="ERROR",   service="postgres",     message="Cannot write WAL — disk full, replication lag growing"),
    LogEntry(timestamp="2024-02-10T10:32:00Z", level="CRITICAL",service="postgres",     message="Standby replica disconnected — replication stopped"),
]

METRICS = SystemMetrics(
    cpu_percent=18.0,
    memory_percent=62.0,
    disk_percent=99.1,
    active_connections=0,
    request_rate=0.0,
    error_rate=100.0,
)

ALERTS = [
    Alert(alert_id="ALT-401", severity="CRITICAL", service="postgres",
          message="Disk full — all writes rejected (ENOSPC)", triggered_at="2024-02-10T10:30:00Z"),
    Alert(alert_id="ALT-402", severity="CRITICAL", service="log-rotator",
          message="Log rotation failed 4 consecutive times — disk at 99%", triggered_at="2024-02-10T10:28:00Z"),
    Alert(alert_id="ALT-403", severity="HIGH",     service="api-server",
          message="503 error rate 100% — database writes failing", triggered_at="2024-02-10T10:30:02Z"),
]


def get_scenario() -> Dict[str, Any]:
    return {
        "task_id": "task4",
        "scenario_name": "disk_full_log_rotation",
        "task_description": TASK_DESCRIPTION,
        "all_logs": ALL_LOGS,
        "metrics": METRICS,
        "alerts": ALERTS,
        "ground_truth": GROUND_TRUTH,
        "max_steps": 15,
        "difficulty": "easy-medium",
    }



def grade(state: EpisodeState) -> float:
    """Task 4 — Easy-Medium: Disk full. Rewards tracing the causal chain."""
    score = 0.0
    gt = GROUND_TRUTH

    if state.root_cause_marked == gt["root_cause"]:
        score += 0.30
    if state.classification_marked == gt["classification"]:
        score += 0.20
    if state.resolution_action == gt["resolution"]:
        score += 0.40
    elif state.resolution_action and "log-rotator" in state.resolution_action:
        score += 0.15

    # Bonus: did agent trace the chain postgres → log-rotator?
    if "log-rotator" in state.services_inspected:
        score += 0.05
    if "postgres" in state.services_inspected:
        score += 0.03  # checked the symptom service too
    kw = " ".join(state.keywords_filtered).lower()
    if "disk" in kw or "rotation" in kw:
        score += 0.02

    # Penalty: restarting postgres directly (treats symptom not cause)
    if any("restart_service" in str(a) and "postgres" in str(a)
           for a in state.actions_history):
        score -= 0.15

    # Efficiency bonus
    if state.step_count <= 6 and score >= 0.88:
        score += 0.05

    score -= 0.05 * state.wrong_action_count
    score -= 0.10 * state.destructive_action_count

    return round(max(0.01, min(0.99, score)), 4)
