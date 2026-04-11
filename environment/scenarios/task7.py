"""
Task 7 (Hard): Network Partition — Split Brain in Distributed Cache
A network partition splits the Redis cluster into two halves.
Services on each side see different data. Writes succeed on one side,
fail on the other. Multiple red herrings including a deploy and slow DB.
Agent must correlate cross-service evidence.
Ground truth: network_partition on redis-cluster; fix = restart_service:redis-cluster
"""

from ..models import LogEntry, SystemMetrics, Alert, EpisodeState
from typing import Dict, Any


GROUND_TRUTH = {
    "root_cause": "network_partition",
    "classification": "infrastructure_failure",
    "resolution": "restart_service:redis-cluster",
    "affected_service": "redis-cluster",
}

TASK_DESCRIPTION = (
    "TASK 7 (Hard) — Network Partition / Split Brain\n"
    "A major data consistency incident is in progress. Some users see stale data, "
    "others get errors. The pattern is geographic — users in region-A are fine, "
    "region-B users are failing. A deploy happened 45 minutes ago (probably unrelated). "
    "There are multiple false signals. You need to cross-reference 5+ services carefully.\n"
    "Services: redis-cluster, api-server, session-service, user-service, "
    "postgres-primary, postgres-replica, load-balancer"
)

ALL_LOGS = [
    # Baseline
    LogEntry(timestamp="2024-05-15T15:00:00Z", level="INFO",    service="redis-cluster",     message="Cluster healthy — 6 nodes, 3 primary, 3 replica"),
    LogEntry(timestamp="2024-05-15T15:00:01Z", level="INFO",    service="load-balancer",     message="Traffic balanced — region-A: 52%, region-B: 48%"),
    # Red herring — deploy
    LogEntry(timestamp="2024-05-15T15:15:00Z", level="INFO",    service="api-server",        message="Deployed v3.1.2 — minor UI changes, no backend impact"),
    # Network partition begins
    LogEntry(timestamp="2024-05-15T15:20:00Z", level="ERROR",   service="redis-cluster",     message="Node redis-4 lost connection to redis-1,2,3 — network timeout"),
    LogEntry(timestamp="2024-05-15T15:20:01Z", level="ERROR",   service="redis-cluster",     message="Node redis-5 lost connection to redis-1,2,3 — network timeout"),
    LogEntry(timestamp="2024-05-15T15:20:02Z", level="CRITICAL",service="redis-cluster",     message="Split brain detected — cluster partitioned into 2 groups"),
    LogEntry(timestamp="2024-05-15T15:20:05Z", level="ERROR",   service="session-service",   message="Session write conflict — key 'sess:U-8821' has diverging values across cluster"),
    LogEntry(timestamp="2024-05-15T15:20:10Z", level="ERROR",   service="user-service",      message="User preference read: inconsistent data — got v1 from node-4, v3 from node-1"),
    LogEntry(timestamp="2024-05-15T15:21:00Z", level="ERROR",   service="session-service",   message="Session invalidation failing on region-B nodes — partition isolating writes"),
    LogEntry(timestamp="2024-05-15T15:21:30Z", level="WARNING", service="postgres-replica",  message="Replication lag 800ms — NOT related to cache partition"),
    LogEntry(timestamp="2024-05-15T15:22:00Z", level="CRITICAL",service="session-service",   message="Users in region-B experiencing stale sessions and auth failures"),
    LogEntry(timestamp="2024-05-15T15:22:05Z", level="ERROR",   service="redis-cluster",     message="Quorum lost on partition-B side — refusing writes to maintain consistency"),
    LogEntry(timestamp="2024-05-15T15:22:10Z", level="ERROR",   service="api-server",        message="Cache miss storm — 14,000 req/s hitting postgres due to Redis errors"),
    LogEntry(timestamp="2024-05-15T15:22:15Z", level="CRITICAL",service="postgres-primary",  message="Connection pool exhausted — Redis cache miss storm overwhelming DB"),
    LogEntry(timestamp="2024-05-15T15:23:00Z", level="ERROR",   service="load-balancer",     message="Region-B health checks degraded — 34% of requests failing"),
    LogEntry(timestamp="2024-05-15T15:23:05Z", level="INFO",    service="postgres-replica",  message="Replication lag normalised — replica healthy"),
    LogEntry(timestamp="2024-05-15T15:24:00Z", level="CRITICAL",service="redis-cluster",     message="Partition duration 4min — data divergence growing, manual intervention required"),
    LogEntry(timestamp="2024-05-15T15:24:10Z", level="ERROR",   service="session-service",   message="Cannot determine authoritative session data — serving errors to region-B users"),
]

METRICS = SystemMetrics(
    cpu_percent=91.0,
    memory_percent=78.0,
    disk_percent=52.0,
    active_connections=14000,
    request_rate=14000.0,
    error_rate=34.0,
)

ALERTS = [
    Alert(alert_id="ALT-701", severity="CRITICAL", service="redis-cluster",
          message="Split brain — cluster partitioned, quorum lost on region-B", triggered_at="2024-05-15T15:20:02Z"),
    Alert(alert_id="ALT-702", severity="CRITICAL", service="postgres-primary",
          message="Connection pool exhausted — cache miss storm from Redis partition", triggered_at="2024-05-15T15:22:15Z"),
    Alert(alert_id="ALT-703", severity="CRITICAL", service="session-service",
          message="Auth failures in region-B — split brain causing stale sessions", triggered_at="2024-05-15T15:22:00Z"),
    Alert(alert_id="ALT-704", severity="HIGH",     service="load-balancer",
          message="Region-B error rate 34% — users impacted", triggered_at="2024-05-15T15:23:00Z"),
]


def get_scenario() -> Dict[str, Any]:
    return {
        "task_id": "task7",
        "scenario_name": "network_partition_split_brain",
        "task_description": TASK_DESCRIPTION,
        "all_logs": ALL_LOGS,
        "metrics": METRICS,
        "alerts": ALERTS,
        "ground_truth": GROUND_TRUTH,
        "max_steps": 30,
        "difficulty": "hard",
    }



def grade(state: EpisodeState) -> float:
    """Task 7 — Hard: Network partition. Requires correlating 5+ services."""
    score = 0.0
    gt = GROUND_TRUTH

    if state.root_cause_marked == gt["root_cause"]:
        score += 0.35
    if state.classification_marked == gt["classification"]:
        score += 0.20
    if state.resolution_action == gt["resolution"]:
        score += 0.30
    elif state.resolution_action and "redis" in state.resolution_action:
        score += 0.12

    # Investigation depth bonus: hard task needs multi-service correlation
    investigation_score = 0.0
    if "redis-cluster" in state.services_inspected:
        investigation_score += 0.05
    if "session-service" in state.services_inspected:
        investigation_score += 0.03
    if len(state.services_inspected) >= 3:
        investigation_score += 0.02
    score += min(investigation_score, 0.10)

    kw = " ".join(state.keywords_filtered).lower()
    if "partition" in kw or "split" in kw:
        score += 0.05

    # Red herring penalties
    if state.root_cause_marked == "memory_leak":
        score -= 0.15  # user-service memory was a red herring
    if state.root_cause_marked and "deploy" in state.root_cause_marked:
        score -= 0.15  # the deploy was unrelated
    if any("restart_service" in str(a) and "postgres" in str(a)
           for a in state.actions_history):
        score -= 0.10  # postgres was a symptom not the cause

    score -= 0.05 * state.wrong_action_count
    score -= 0.10 * state.destructive_action_count

    return round(max(0.01, min(0.99, score)), 4)
