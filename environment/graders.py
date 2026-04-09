from typing import Callable
from .scenarios import get_grader

# Strict open interval bounds — validator requires score in (0, 1), not [0, 1]
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def grade_task(task_name: str, state) -> float:
    """
    Central grading entry point.
    Returns a score strictly in (0, 1) — never exactly 0.0 or 1.0.
    """
    grader_fn: Callable = get_grader(task_name)
    score = grader_fn(state)
    # Clamp to open interval (0.01, 0.99)
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
            action = _baseline_policy(obs)
            obs, reward, done, _ = env.step(action)
        final_state = env.state()
        score = grade_task(task, final_state)
        results[task] = score
    return results


def _baseline_policy(observation):
    from .models import Action
    if observation.step_count == 0:
        return Action(action_type="filter_logs", target="error")
    if observation.step_count == 1:
        return Action(action_type="filter_logs", target="memory")
    if observation.step_count == 2:
        return Action(action_type="inspect_service", target="api-server")
    if observation.step_count == 3:
        return Action(action_type="mark_root_cause", target="oom_kill")
    if observation.step_count == 4:
        return Action(action_type="classify_issue", target="infrastructure_failure")
    return Action(action_type="resolve_incident", target="restart_service:api-server")