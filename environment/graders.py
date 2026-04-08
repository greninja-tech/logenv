from typing import Callable
from .scenarios import get_grader


def grade_task(task_name: str, state) -> float:
    """
    Central grading entry point.

    Args:
        task_name (str): task identifier (task1, task2, task3)
        state (EpisodeState): final trajectory state

    Returns:
        float: score between 0.0 and 1.0
    """
    grader_fn: Callable = get_grader(task_name)
    score = grader_fn(state)

    # Safety clamp (important for spec compliance)
    return max(0.0, min(1.0, float(score)))


def evaluate_all_tasks(env_class, tasks=None) -> dict:
    """
    Runs evaluation across all tasks using the given environment class.

    Args:
        env_class: your environment class (LogEnv)
        tasks (list): optional list of tasks

    Returns:
        dict: task -> score
    """
    if tasks is None:
        tasks = ["task1", "task2", "task3"]

    results = {}

    for task in tasks:
        env = env_class(task_name=task)
        obs = env.reset()

        done = False
        while not done:
            # Dummy baseline policy (very simple)
            action = _baseline_policy(obs)
            obs, reward, done, _ = env.step(action)

        final_state = env.state()
        score = grade_task(task, final_state)
        results[task] = score

    return results


# ---------------- BASELINE POLICY ----------------

def _baseline_policy(observation):
    """
    Simple heuristic policy (must be lightweight for <20min runtime).
    You can improve this later.
    """
    from .models import Action

    # naive heuristic
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