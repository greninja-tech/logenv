"""
Task 3 (Hard): Distributed System Cascading Failure
A misconfigured circuit breaker in the 'order-service' causes a cascading failure
across a microservices mesh. Multiple red herrings (network blip, slow DB query,
noisy metrics). Agent must cross-reference 4+ services.
Ground truth: misconfigured_circuit_breaker in order-service.
"""

from ..models import LogEntry, SystemMetrics, Alert, EpisodeState
from typing import Dict, Any


GROUND_TRUTH = {
    "root_cause": "misconfigured_circuit_breaker",
    "classification": "configuration_error",
    "resolution": "scale_service:order-service",
    "affected_service": "order-service",
}

TASK_DESCRIPTION = (
    "TASK 3 (Hard) — Cascading Distributed System Failure\n"
    "A major incident is in progress. Multiple services are reporting errors simultaneously. "
    "The situation started ~20 minutes ago and is worsening. Revenue impact is confirmed. "
    "You must trace the root cause through the noise — beware of misleading signals. "
    "There are red herrings. Cross-reference services carefully.\n"
    "Services: api-gateway, order-service, inventory-service, payment-service, "
    "notification-service, user-service, postgres-primary, postgres-replica, kafka, redis-cluster"
)

ALL_LOGS = [
    # Background noise — pre-incident (healthy)
    LogEntry(timestamp="2024-01-15T14:00:00Z", level="INFO", service="api-gateway",         message="All services healthy. Routing 850 req/s"),
    LogEntry(timestamp="2024-01-15T14:00:01Z", level="INFO", service="order-service",       message="Order processing nominal — 120 orders/min"),
    LogEntry(timestamp="2024-01-15T14:00:02Z", level="INFO", service="inventory-service",   message="Stock checks: 340/min — all OK"),
    LogEntry(timestamp="2024-01-15T14:00:03Z", level="INFO", service="payment-service",     message="Payment gateway connected — success rate 99.8%"),
    LogEntry(timestamp="2024-01-15T14:00:04Z", level="INFO", service="kafka",               message="Consumer lag: 0ms — all partitions healthy"),
    LogEntry(timestamp="2024-01-15T14:00:05Z", level="INFO", service="redis-cluster",       message="Cache hit rate: 94% — 3 nodes healthy"),
    # RED HERRING 1: network blip (looks scary, is harmless)
    LogEntry(timestamp="2024-01-15T14:02:00Z", level="WARNING", service="api-gateway",      message="Network jitter detected — packet loss 0.3% (transient)"),
    LogEntry(timestamp="2024-01-15T14:02:01Z", level="WARNING", service="api-gateway",      message="Packet loss resolved — normal operations resumed"),
    LogEntry(timestamp="2024-01-15T14:02:05Z", level="INFO",    service="order-service",    message="Order processing normal post-jitter — no impact"),
    # Circuit breaker config change (THE ROOT CAUSE trigger — subtle)
    LogEntry(timestamp="2024-01-15T14:05:00Z", level="INFO",    service="order-service",    message="Config reload: circuit_breaker.threshold updated to 5 (was 50)"),
    LogEntry(timestamp="2024-01-15T14:05:01Z", level="INFO",    service="order-service",    message="Circuit breaker config applied — monitoring"),
    # Incident begins: CB fires at 5 failures instead of 50
    LogEntry(timestamp="2024-01-15T14:06:00Z", level="WARNING", service="order-service",    message="Circuit breaker OPEN for inventory-service (5 failures in 10s — threshold: 5)"),
    LogEntry(timestamp="2024-01-15T14:06:01Z", level="ERROR",   service="order-service",    message="Inventory checks failing — circuit breaker preventing requests to inventory-service"),
    LogEntry(timestamp="2024-01-15T14:06:02Z", level="ERROR",   service="inventory-service",message="Receiving no requests — idle (circuit breaker upstream blocking us)"),
    LogEntry(timestamp="2024-01-15T14:06:10Z", level="ERROR",   service="order-service",    message="Cannot validate stock — failing orders with 503"),
    LogEntry(timestamp="2024-01-15T14:06:11Z", level="ERROR",   service="api-gateway",      message="order-service returning 503 — upstream error rate 12%"),
    # RED HERRING 2: postgres replica lag (coincidental, not the cause)
    LogEntry(timestamp="2024-01-15T14:07:00Z", level="WARNING", service="postgres-replica", message="Replication lag: 4.2s — elevated (cause: write burst from order rollbacks)"),
    LogEntry(timestamp="2024-01-15T14:07:01Z", level="INFO",    service="postgres-primary", message="Accepting writes normally — 1,240 TPS"),
    # Cascading: payment fails because orders fail upstream
    LogEntry(timestamp="2024-01-15T14:08:00Z", level="ERROR",   service="payment-service",  message="Order confirmations not arriving — holding transactions in pending state"),
    LogEntry(timestamp="2024-01-15T14:08:01Z", level="WARNING", service="payment-service",  message="Pending transaction queue: 842 items — growing"),
    # Kafka consumer lag starts building
    LogEntry(timestamp="2024-01-15T14:09:00Z", level="WARNING", service="kafka",            message="Consumer lag increasing — order-events topic: 12,400 messages"),
    LogEntry(timestamp="2024-01-15T14:09:01Z", level="INFO",    service="notification-service", message="Waiting for order events — no activity (upstream blocked)"),
    # RED HERRING 3: user-service reports memory spike (red herring — unrelated batch job)
    LogEntry(timestamp="2024-01-15T14:10:00Z", level="WARNING", service="user-service",     message="Memory spike: 78% — batch analytics job running (scheduled, expected)"),
    LogEntry(timestamp="2024-01-15T14:10:01Z", level="INFO",    service="user-service",     message="Auth requests unaffected — login success 99.9%"),
    # Cascading worsens
    LogEntry(timestamp="2024-01-15T14:12:00Z", level="ERROR",   service="order-service",    message="Circuit breaker OPEN for payment-service (5 failures — same CB config issue)"),
    LogEntry(timestamp="2024-01-15T14:12:01Z", level="CRITICAL",service="order-service",    message="All downstream circuit breakers open — order processing completely halted"),
    LogEntry(timestamp="2024-01-15T14:12:02Z", level="CRITICAL",service="api-gateway",      message="order-service 100% error rate — removing from load balancer"),
    LogEntry(timestamp="2024-01-15T14:12:03Z", level="ERROR",   service="payment-service",  message="Pending queue: 4,210 transactions — timeout risk"),
    LogEntry(timestamp="2024-01-15T14:12:04Z", level="WARNING", service="kafka",            message="Consumer lag: 45,000 messages — order-events partition overflow risk"),
    # Redis cache — fine, another red herring opportunity
    LogEntry(timestamp="2024-01-15T14:13:00Z", level="INFO",    service="redis-cluster",    message="Cache hit rate dropped to 71% (less order traffic to cache)"),
    # More cascade
    LogEntry(timestamp="2024-01-15T14:14:00Z", level="CRITICAL",service="api-gateway",      message="Revenue impact confirmed — 0 orders processed in last 120s"),
    LogEntry(timestamp="2024-01-15T14:14:01Z", level="ERROR",   service="inventory-service" ,message="Receiving zero requests — service idle, circuit breaker upstream"),
    LogEntry(timestamp="2024-01-15T14:15:00Z", level="WARNING", service="postgres-replica", message="Replication lag: 8.1s — increasing as rollback writes surge"),
    LogEntry(timestamp="2024-01-15T14:15:01Z", level="INFO",    service="postgres-primary", message="Write volume elevated due to order rollbacks — 2,100 TPS"),
    LogEntry(timestamp="2024-01-15T14:16:00Z", level="CRITICAL",service="order-service",    message="Config audit: circuit_breaker.threshold=5 (INCORRECT — should be 50). Deploy #4821"),
    LogEntry(timestamp="2024-01-15T14:16:01Z", level="CRITICAL",service="order-service",    message="Root cause confirmed internally — misconfigured CB threshold blocking all downstream"),
]

METRICS = SystemMetrics(
    cpu_percent=22.0,
    memory_percent=74.0,
    disk_percent=51.0,
    active_connections=124,
    request_rate=0.0,
    error_rate=100.0,
)

ALERTS = [
    Alert(alert_id="ALT-100", severity="CRITICAL", service="order-service",
          message="100% error rate — removed from load balancer", triggered_at="2024-01-15T14:12:02Z"),
    Alert(alert_id="ALT-101", severity="CRITICAL", service="api-gateway",
          message="Revenue impact — 0 orders/min for 120s", triggered_at="2024-01-15T14:14:00Z"),
    Alert(alert_id="ALT-102", severity="HIGH",     service="payment-service",
          message="4,210 pending transactions — approaching timeout", triggered_at="2024-01-15T14:12:03Z"),
    Alert(alert_id="ALT-103", severity="HIGH",     service="kafka",
          message="Consumer lag 45,000 messages — overflow risk", triggered_at="2024-01-15T14:12:04Z"),
    Alert(alert_id="ALT-104", severity="MEDIUM",   service="postgres-replica",
          message="Replication lag 8.1s — elevated", triggered_at="2024-01-15T14:15:00Z"),
    Alert(alert_id="ALT-105", severity="LOW",      service="user-service",
          message="Memory elevated — scheduled batch job", triggered_at="2024-01-15T14:10:00Z"),
]


def get_scenario() -> Dict[str, Any]:
    return {
        "task_id": "task3",
        "scenario_name": "distributed_cascade_failure",
        "task_description": TASK_DESCRIPTION,
        "all_logs": ALL_LOGS,
        "metrics": METRICS,
        "alerts": ALERTS,
        "ground_truth": GROUND_TRUTH,
        "max_steps": 30,
        "difficulty": "hard",
    }



def grade(state: EpisodeState) -> float:
    """Task 3 — Hard: Cascading failure. Requires deep multi-service investigation."""
    score = 0.0
    gt = GROUND_TRUTH

    # Root cause worth more — it is the hardest to identify
    if state.root_cause_marked == gt["root_cause"]:
        score += 0.35
    if state.classification_marked == gt["classification"]:
        score += 0.20
    if state.resolution_action == gt["resolution"]:
        score += 0.30
    elif state.resolution_action and "order-service" in state.resolution_action:
        score += 0.12

    # Investigation quality: hard task requires cross-referencing multiple services
    services_bonus = 0.0
    if "order-service" in state.services_inspected:
        services_bonus += 0.05
    if "inventory-service" in state.services_inspected:
        services_bonus += 0.03
    if len(state.services_inspected) >= 3:
        services_bonus += 0.02  # reward for thorough investigation
    score += min(services_bonus, 0.10)

    # Keyword investigation bonus
    kw = " ".join(state.keywords_filtered).lower()
    if "circuit" in kw or "config" in kw:
        score += 0.05

    # Red herring penalties (hard task — falling for traps is costly)
    if any("postgres" in str(a) and "restart" in str(a) for a in state.actions_history):
        score -= 0.15
    if state.root_cause_marked and "user" in state.root_cause_marked:
        score -= 0.15
    if state.root_cause_marked and "network" in state.root_cause_marked:
        score -= 0.10  # network blip was a red herring

    score -= 0.05 * state.wrong_action_count
    score -= 0.10 * state.destructive_action_count

    return round(max(0.01, min(0.99, score)), 4)
