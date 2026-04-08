"""
Task 2 (Medium): Memory Leak in Microservices
A memory leak in the 'session-manager' service causes gradual degradation.
Logs are noisier; the agent must filter and correlate across services.
Ground truth: memory_leak in session-manager; fix = restart + scale.
"""

from ..models import LogEntry, SystemMetrics, Alert, EpisodeState
from typing import Dict, Any


GROUND_TRUTH = {
    "root_cause": "memory_leak",
    "classification": "application_bug",
    "resolution": "restart_service:session-manager",
    "affected_service": "session-manager",
}

TASK_DESCRIPTION = (
    "TASK 2 (Medium) — Memory Leak Investigation\n"
    "Over the past hour, response times have been degrading. No hard crash yet, "
    "but users report slow logins and occasional 503s. Multiple services are involved. "
    "Identify which service has a memory leak, classify the issue, and resolve it.\n"
    "Services: api-gateway, session-manager, user-service, payment-service, postgres"
)

ALL_LOGS = [
    # Hour 1 — baseline
    LogEntry(timestamp="2024-01-15T09:00:00Z", level="INFO",    service="api-gateway",     message="Request routing normal — p99 latency: 45ms"),
    LogEntry(timestamp="2024-01-15T09:00:01Z", level="INFO",    service="session-manager", message="Session cache initialized — heap: 256MB"),
    LogEntry(timestamp="2024-01-15T09:00:02Z", level="INFO",    service="user-service",    message="User service healthy — 200 req/min"),
    LogEntry(timestamp="2024-01-15T09:00:03Z", level="INFO",    service="payment-service", message="Payment processor connected"),
    LogEntry(timestamp="2024-01-15T09:05:00Z", level="INFO",    service="session-manager", message="Active sessions: 1,204 — heap: 278MB"),
    LogEntry(timestamp="2024-01-15T09:10:00Z", level="INFO",    service="session-manager", message="Active sessions: 1,198 — heap: 312MB"),
    LogEntry(timestamp="2024-01-15T09:15:00Z", level="INFO",    service="user-service",    message="Auth requests: 180/min — all OK"),
    LogEntry(timestamp="2024-01-15T09:15:00Z", level="INFO",    service="session-manager", message="Active sessions: 1,201 — heap: 358MB"),
    LogEntry(timestamp="2024-01-15T09:20:00Z", level="INFO",    service="api-gateway",     message="p99 latency: 62ms — slight increase"),
    LogEntry(timestamp="2024-01-15T09:20:00Z", level="INFO",    service="session-manager", message="Active sessions: 1,199 — heap: 412MB"),
    LogEntry(timestamp="2024-01-15T09:25:00Z", level="WARNING", service="session-manager", message="Heap growing abnormally — sessions stable but heap: 478MB"),
    LogEntry(timestamp="2024-01-15T09:25:01Z", level="INFO",    service="postgres",        message="Query performance nominal — avg 8ms"),
    LogEntry(timestamp="2024-01-15T09:30:00Z", level="WARNING", service="session-manager", message="GC unable to reclaim memory — heap: 556MB (sessions: 1,202)"),
    LogEntry(timestamp="2024-01-15T09:30:00Z", level="INFO",    service="api-gateway",     message="p99 latency: 110ms — degrading"),
    LogEntry(timestamp="2024-01-15T09:35:00Z", level="WARNING", service="api-gateway",     message="Upstream session-manager slow — avg response: 280ms"),
    LogEntry(timestamp="2024-01-15T09:35:00Z", level="INFO",    service="user-service",    message="Auth latency increasing — p99: 310ms"),
    LogEntry(timestamp="2024-01-15T09:35:01Z", level="INFO",    service="session-manager", message="Heap: 634MB — GC running continuously"),
    LogEntry(timestamp="2024-01-15T09:40:00Z", level="ERROR",   service="session-manager", message="GC overhead limit exceeded — application threads paused"),
    LogEntry(timestamp="2024-01-15T09:40:01Z", level="WARNING", service="api-gateway",     message="Session-manager timeouts increasing — 503 rate: 2%"),
    LogEntry(timestamp="2024-01-15T09:40:02Z", level="INFO",    service="payment-service", message="No issues — operating normally"),
    LogEntry(timestamp="2024-01-15T09:45:00Z", level="ERROR",   service="session-manager", message="Heap: 891MB — suspected memory leak in SessionCache.cleanup()"),
    LogEntry(timestamp="2024-01-15T09:45:01Z", level="WARNING", service="user-service",    message="Login timeouts 8% — upstream session-manager unresponsive"),
    LogEntry(timestamp="2024-01-15T09:45:02Z", level="INFO",    service="postgres",        message="Connection pool healthy — 12/20 connections used"),
    LogEntry(timestamp="2024-01-15T09:50:00Z", level="CRITICAL",service="session-manager", message="Heap: 1.1GB / 1.2GB limit — imminent OOM risk"),
    LogEntry(timestamp="2024-01-15T09:50:01Z", level="ERROR",   service="api-gateway",     message="503 rate: 18% — session-manager failing"),
    LogEntry(timestamp="2024-01-15T09:50:02Z", level="WARNING", service="user-service",    message="Degraded — login success rate 82%"),
    # Red herring: postgres has a slow query log but is NOT the cause
    LogEntry(timestamp="2024-01-15T09:52:00Z", level="WARNING", service="postgres",        message="Slow query detected: SELECT * FROM sessions (2.1s) — needs index"),
    LogEntry(timestamp="2024-01-15T09:52:01Z", level="INFO",    service="payment-service", message="Payment success rate 100% — unaffected"),
    LogEntry(timestamp="2024-01-15T09:55:00Z", level="CRITICAL",service="session-manager", message="Heap: 1.19GB — crash imminent, manual intervention required"),
    LogEntry(timestamp="2024-01-15T09:55:01Z", level="ERROR",   service="api-gateway",     message="503 rate: 35% — circuit breaker triggered for session-manager"),
]

METRICS = SystemMetrics(
    cpu_percent=85.0,
    memory_percent=91.0,
    disk_percent=48.0,
    active_connections=847,
    request_rate=142.0,
    error_rate=35.0,
)

ALERTS = [
    Alert(alert_id="ALT-010", severity="CRITICAL", service="session-manager",
          message="Heap usage 99% — memory leak suspected", triggered_at="2024-01-15T09:55:00Z"),
    Alert(alert_id="ALT-011", severity="HIGH",     service="api-gateway",
          message="503 error rate 35% — circuit breaker open", triggered_at="2024-01-15T09:55:01Z"),
    Alert(alert_id="ALT-012", severity="MEDIUM",   service="user-service",
          message="Login success rate below 85% threshold", triggered_at="2024-01-15T09:50:02Z"),
    Alert(alert_id="ALT-013", severity="LOW",      service="postgres",
          message="Slow query detected — non-critical", triggered_at="2024-01-15T09:52:00Z"),
]


def get_scenario() -> Dict[str, Any]:
    return {
        "task_id": "task2",
        "scenario_name": "memory_leak",
        "task_description": TASK_DESCRIPTION,
        "all_logs": ALL_LOGS,
        "metrics": METRICS,
        "alerts": ALERTS,
        "ground_truth": GROUND_TRUTH,
        "max_steps": 20,
        "difficulty": "medium",
    }


def grade(state: EpisodeState) -> float:
    """Deterministic grader for Task 2."""
    score = 0.0
    gt = GROUND_TRUTH

    # Root cause identification (0.30)
    if state.root_cause_marked == gt["root_cause"]:
        score += 0.30

    # Classification (0.20)
    if state.classification_marked == gt["classification"]:
        score += 0.20

    # Resolution action (0.40)
    if state.resolution_action == gt["resolution"]:
        score += 0.40
    elif state.resolution_action and "session-manager" in state.resolution_action:
        score += 0.15  # partial: right service, wrong action

    # Investigation quality bonus: inspected the right service (0.10)
    if "session-manager" in state.services_inspected:
        score += 0.05
    if "memory" in " ".join(state.keywords_filtered).lower() or "heap" in " ".join(state.keywords_filtered).lower():
        score += 0.05

    # Penalties
    score -= 0.05 * state.wrong_action_count
    score -= 0.10 * state.destructive_action_count
    # Penalty for restarting the wrong service (postgres)
    if "restart_service:postgres" in state.actions_history or \
       any("restart_service" in str(a) and "postgres" in str(a) for a in state.actions_history):
        score -= 0.10

    return round(max(0.0, min(1.0, score)), 4)