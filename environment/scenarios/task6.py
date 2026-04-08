"""
Task 6 (Medium-Hard): Third-Party Dependency Failure
An external payment gateway goes down, causing checkout failures.
Internal services are healthy. Agent must distinguish external dependency
failure from internal bugs. Red herrings: slow DB query, high CPU on worker.
Ground truth: dependency_failure on checkout-service; fix = rollback_deploy:checkout-service
"""

from ..models import LogEntry, SystemMetrics, Alert, EpisodeState
from typing import Dict, Any


GROUND_TRUTH = {
    "root_cause": "dependency_failure",
    "classification": "dependency_failure",
    "resolution": "rollback_deploy:checkout-service",
    "affected_service": "checkout-service",
}

TASK_DESCRIPTION = (
    "TASK 6 (Medium-Hard) — External Dependency Failure\n"
    "Checkout is failing for all users. A new version of checkout-service was deployed "
    "30 minutes ago. Revenue impact is confirmed — $0 processed in last 25 minutes. "
    "Internal services appear healthy. Investigate whether this is internal or external. "
    "Beware of misleading signals.\n"
    "Services: checkout-service, payment-gateway, inventory-service, user-service, postgres, redis"
)

ALL_LOGS = [
    # Pre-deploy baseline
    LogEntry(timestamp="2024-04-01T09:00:00Z", level="INFO",    service="checkout-service",  message="v2.3.1 running — checkout success rate 99.2%"),
    LogEntry(timestamp="2024-04-01T09:28:00Z", level="INFO",    service="checkout-service",  message="Deploying v2.4.0 — new payment gateway SDK"),
    LogEntry(timestamp="2024-04-01T09:30:00Z", level="INFO",    service="checkout-service",  message="v2.4.0 deployed successfully"),
    # Red herring 1 — slow DB query
    LogEntry(timestamp="2024-04-01T09:30:30Z", level="WARNING", service="postgres",          message="Slow query detected: SELECT * FROM inventory (2.3s) — non-critical"),
    # Dependency failures begin
    LogEntry(timestamp="2024-04-01T09:31:00Z", level="ERROR",   service="checkout-service",  message="Payment gateway API timeout after 30s — endpoint: api.stripe-mock.io/charge"),
    LogEntry(timestamp="2024-04-01T09:31:01Z", level="ERROR",   service="checkout-service",  message="Checkout failed for user #U-4421 — payment gateway unreachable"),
    LogEntry(timestamp="2024-04-01T09:31:30Z", level="ERROR",   service="checkout-service",  message="Payment gateway API timeout after 30s — 3rd consecutive failure"),
    LogEntry(timestamp="2024-04-01T09:32:00Z", level="ERROR",   service="checkout-service",  message="Circuit breaker OPEN — payment gateway marked DOWN"),
    LogEntry(timestamp="2024-04-01T09:32:01Z", level="CRITICAL",service="checkout-service",  message="All checkouts failing — payment gateway dependency unavailable"),
    # Red herring 2 — high CPU on worker (unrelated)
    LogEntry(timestamp="2024-04-01T09:32:10Z", level="WARNING", service="inventory-service", message="CPU spike 78% — batch inventory sync running (scheduled)"),
    # More gateway failures
    LogEntry(timestamp="2024-04-01T09:33:00Z", level="ERROR",   service="checkout-service",  message="payment-gateway DNS resolution: api.stripe-mock.io NXDOMAIN — host not found"),
    LogEntry(timestamp="2024-04-01T09:33:01Z", level="ERROR",   service="checkout-service",  message="v2.4.0 SDK uses wrong gateway endpoint — regression from config change"),
    LogEntry(timestamp="2024-04-01T09:34:00Z", level="INFO",    service="user-service",      message="User sessions healthy — 8,200 active users"),
    LogEntry(timestamp="2024-04-01T09:34:01Z", level="INFO",    service="redis",             message="Cache healthy — hit rate 94%"),
    LogEntry(timestamp="2024-04-01T09:35:00Z", level="CRITICAL",service="checkout-service",  message="Revenue impact: 0 successful payments in last 5 minutes"),
    LogEntry(timestamp="2024-04-01T09:35:10Z", level="INFO",    service="inventory-service", message="CPU normalised — batch sync complete"),
    LogEntry(timestamp="2024-04-01T09:36:00Z", level="ERROR",   service="checkout-service",  message="Retrying payment gateway — all attempts failing (wrong SDK endpoint in v2.4.0)"),
]

METRICS = SystemMetrics(
    cpu_percent=34.0,
    memory_percent=58.0,
    disk_percent=38.0,
    active_connections=8200,
    request_rate=340.0,
    error_rate=100.0,
)

ALERTS = [
    Alert(alert_id="ALT-601", severity="CRITICAL", service="checkout-service",
          message="Payment gateway unreachable — 0 successful checkouts", triggered_at="2024-04-01T09:32:01Z"),
    Alert(alert_id="ALT-602", severity="CRITICAL", service="checkout-service",
          message="Revenue impact confirmed — $0 processed in 5 minutes", triggered_at="2024-04-01T09:35:00Z"),
    Alert(alert_id="ALT-603", severity="HIGH",     service="checkout-service",
          message="v2.4.0 deployed 5min before failures — rollback candidate", triggered_at="2024-04-01T09:33:01Z"),
]


def get_scenario() -> Dict[str, Any]:
    return {
        "task_id": "task6",
        "scenario_name": "dependency_failure_bad_deploy",
        "task_description": TASK_DESCRIPTION,
        "all_logs": ALL_LOGS,
        "metrics": METRICS,
        "alerts": ALERTS,
        "ground_truth": GROUND_TRUTH,
        "max_steps": 20,
        "difficulty": "medium-hard",
    }


def grade(state: EpisodeState) -> float:
    score = 0.0
    gt = GROUND_TRUTH

    if state.root_cause_marked == gt["root_cause"]:
        score += 0.30
    if state.classification_marked == gt["classification"]:
        score += 0.20
    if state.resolution_action == gt["resolution"]:
        score += 0.40
    elif state.resolution_action and "checkout" in state.resolution_action:
        score += 0.15

    if state.step_count <= 8 and score >= 0.90:
        score += 0.10
    elif state.step_count <= 14 and score >= 0.70:
        score += 0.05

    score -= 0.05 * state.wrong_action_count
    score -= 0.10 * state.destructive_action_count

    return round(max(0.0, min(1.0, score)), 4)