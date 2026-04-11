from typing import Callable
from .scenarios import get_grader

# Scores must be strictly in open interval (0, 1) — validator requirement
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def grade_task(task_name: str, state) -> float:
    """
    Central grading entry point.
    Returns score strictly in (0.01, 0.99) — never 0.0 or 1.0.
    """
    grader_fn: Callable = get_grader(task_name)
    score = grader_fn(state)
    return round(max(_SCORE_MIN, min(_SCORE_MAX, float(score))), 4)


def evaluate_all_tasks(env_class, tasks=None) -> dict:
    if tasks is None:
        tasks = ["task1", "task2", "task3", "task4", "task5", "task6", "task7"]
    results = {}
    for task in tasks:
        env = env_class(task_name=task)
        obs = env.reset()
        done = False
        while not done:
            action = _baseline_policy(obs, task)
            obs, _, done, _ = env.step(action)
        results[task] = grade_task(task, env.state())
    return results


def _baseline_policy(obs, task_id="task1"):
    from .models import Action
    step = obs.step_count
    sequences = {
        "task1": [("filter_logs","error"),("inspect_service","api-server"),("mark_root_cause","oom_kill"),("classify_issue","infrastructure_failure"),("resolve_incident","restart_service:api-server")],
        "task2": [("filter_logs","memory"),("inspect_service","session-manager"),("mark_root_cause","memory_leak"),("classify_issue","application_bug"),("resolve_incident","restart_service:session-manager")],
        "task3": [("filter_logs","circuit"),("inspect_service","order-service"),("mark_root_cause","misconfigured_circuit_breaker"),("classify_issue","configuration_error"),("resolve_incident","scale_service:order-service")],
        "task4": [("filter_logs","disk"),("inspect_service","log-rotator"),("mark_root_cause","disk_full"),("classify_issue","infrastructure_failure"),("resolve_incident","restart_service:log-rotator")],
        "task5": [("filter_logs","deadlock"),("inspect_service","payment-service"),("mark_root_cause","deadlock"),("classify_issue","application_bug"),("resolve_incident","restart_service:payment-service")],
        "task6": [("filter_logs","error"),("inspect_service","checkout-service"),("mark_root_cause","dependency_failure"),("classify_issue","dependency_failure"),("resolve_incident","rollback_deploy:checkout-service")],
        "task7": [("filter_logs","partition"),("inspect_service","redis-cluster"),("mark_root_cause","network_partition"),("classify_issue","infrastructure_failure"),("resolve_incident","restart_service:redis-cluster")],
    }
    seq = sequences.get(task_id, sequences["task1"])
    idx = min(step, len(seq) - 1)
    at, tgt = seq[idx]
    return Action(action_type=at, target=tgt)
