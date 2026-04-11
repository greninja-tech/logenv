"""
Task 5 (Medium): Database Deadlock in Payment Service
Concurrent transactions in the payment service cause a deadlock.
Logs show lock wait timeouts. Red herring: slow network latency spike at same time.
Ground truth: deadlock in payment-service; fix = restart_service:payment-service
"""

from ..models import LogEntry, SystemMetrics, Alert, EpisodeState
from typing import Dict, Any


GROUND_TRUTH = {
    "root_cause": "deadlock",
    "classification": "application_bug",
    "resolution": "restart_service:payment-service",
    "affected_service": "payment-service",
}

TASK_DESCRIPTION = (
    "TASK 5 (Medium) — Payment Service Deadlock\n"
    "Payments are failing intermittently. Users report checkout errors. "
    "The issue started 20 minutes ago and is worsening. "
    "There was a network blip at the same time — be careful, that may be a red herring. "
    "Identify the true root cause and resolve it.\n"
    "Services: payment-service, order-service, postgres, api-gateway, redis"
)

ALL_LOGS = [
    # Baseline — healthy
    LogEntry(timestamp="2024-03-05T14:00:00Z", level="INFO",    service="api-gateway",     message="All services healthy — p99 latency 38ms"),
    LogEntry(timestamp="2024-03-05T14:00:01Z", level="INFO",    service="payment-service", message="Payment processor ready — 12 worker threads"),
    LogEntry(timestamp="2024-03-05T14:00:02Z", level="INFO",    service="postgres",        message="Connection pool healthy — 45/100 connections used"),
    # Red herring — network blip
    LogEntry(timestamp="2024-03-05T14:10:00Z", level="WARNING", service="api-gateway",     message="Network latency spike detected — p99 jumped to 180ms"),
    LogEntry(timestamp="2024-03-05T14:10:05Z", level="INFO",    service="api-gateway",     message="Latency normalised — p99 back to 42ms"),
    # Deadlock begins
    LogEntry(timestamp="2024-03-05T14:12:00Z", level="WARNING", service="payment-service", message="Lock wait timeout on txn #8821 — retrying"),
    LogEntry(timestamp="2024-03-05T14:12:01Z", level="WARNING", service="payment-service", message="Lock wait timeout on txn #8822 — retrying"),
    LogEntry(timestamp="2024-03-05T14:12:10Z", level="ERROR",   service="postgres",        message="Deadlock detected between txn #8821 and #8822 — rolling back #8822"),
    LogEntry(timestamp="2024-03-05T14:12:11Z", level="ERROR",   service="payment-service", message="Transaction rolled back — payment #PAY-9921 failed"),
    LogEntry(timestamp="2024-03-05T14:13:00Z", level="ERROR",   service="postgres",        message="Deadlock detected between txn #8830 and #8831 — rolling back"),
    LogEntry(timestamp="2024-03-05T14:13:01Z", level="ERROR",   service="payment-service", message="Transaction rolled back — payment #PAY-9930 failed"),
    LogEntry(timestamp="2024-03-05T14:14:00Z", level="ERROR",   service="postgres",        message="Deadlock rate: 12/min — lock contention on payments table"),
    LogEntry(timestamp="2024-03-05T14:14:05Z", level="ERROR",   service="payment-service", message="Worker threads blocked — 8/12 threads waiting on locks"),
    LogEntry(timestamp="2024-03-05T14:15:00Z", level="CRITICAL",service="payment-service", message="All worker threads deadlocked — payment processing halted"),
    LogEntry(timestamp="2024-03-05T14:15:01Z", level="ERROR",   service="order-service",   message="Payment gateway timeout — orders stuck in pending state"),
    LogEntry(timestamp="2024-03-05T14:15:05Z", level="CRITICAL",service="api-gateway",     message="Payment endpoint returning 502 — downstream timeout"),
    LogEntry(timestamp="2024-03-05T14:15:10Z", level="INFO",    service="redis",           message="Session cache normal — no anomalies detected"),
    LogEntry(timestamp="2024-03-05T14:16:00Z", level="ERROR",   service="postgres",        message="Connection pool exhausted — payment-service holding 98/100 connections"),
]

METRICS = SystemMetrics(
    cpu_percent=89.0,
    memory_percent=74.0,
    disk_percent=41.0,
    active_connections=98,
    request_rate=120.0,
    error_rate=87.0,
)

ALERTS = [
    Alert(alert_id="ALT-501", severity="CRITICAL", service="payment-service",
          message="All payment workers deadlocked — 0 transactions/min", triggered_at="2024-03-05T14:15:00Z"),
    Alert(alert_id="ALT-502", severity="CRITICAL", service="postgres",
          message="Deadlock rate 12/min — lock contention critical", triggered_at="2024-03-05T14:14:00Z"),
    Alert(alert_id="ALT-503", severity="HIGH",     service="order-service",
          message="Payment timeout — 340 orders stuck in pending", triggered_at="2024-03-05T14:15:01Z"),
]


def get_scenario() -> Dict[str, Any]:
    return {
        "task_id": "task5",
        "scenario_name": "payment_deadlock",
        "task_description": TASK_DESCRIPTION,
        "all_logs": ALL_LOGS,
        "metrics": METRICS,
        "alerts": ALERTS,
        "ground_truth": GROUND_TRUTH,
        "max_steps": 20,
        "difficulty": "medium",
    }



def grade(state: EpisodeState) -> float:
    """Task 5 — Medium: Deadlock. Rewards ignoring the network red herring."""
    score = 0.0
    gt = GROUND_TRUTH

    if state.root_cause_marked == gt["root_cause"]:
        score += 0.30
    if state.classification_marked == gt["classification"]:
        score += 0.20
    if state.resolution_action == gt["resolution"]:
        score += 0.40
    elif state.resolution_action and "payment" in state.resolution_action:
        score += 0.15

    # Investigation quality
    if "payment-service" in state.services_inspected:
        score += 0.05
    kw = " ".join(state.keywords_filtered).lower()
    if "deadlock" in kw or "lock" in kw:
        score += 0.05

    # Red herring penalty: if agent blamed the network blip
    if state.root_cause_marked and "network" in state.root_cause_marked:
        score -= 0.20
    # Penalty: restarting api-gateway (treating symptom)
    if any("restart_service" in str(a) and "gateway" in str(a)
           for a in state.actions_history):
        score -= 0.10

    score -= 0.05 * state.wrong_action_count
    score -= 0.10 * state.destructive_action_count

    return round(max(0.01, min(0.99, score)), 4)
