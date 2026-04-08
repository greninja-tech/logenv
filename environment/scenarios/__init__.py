from .task1 import get_scenario as get_task1, grade as grade_task1
from .task2 import get_scenario as get_task2, grade as grade_task2
from .task3 import get_scenario as get_task3, grade as grade_task3


# Registry for loading scenarios
TASK_REGISTRY = {
    "task1": get_task1,
    "task2": get_task2,
    "task3": get_task3,
}

# Registry for graders
GRADER_REGISTRY = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}


def load_task(task_name: str):
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}")
    return TASK_REGISTRY[task_name]()


def get_grader(task_name: str):
    if task_name not in GRADER_REGISTRY:
        raise ValueError(f"No grader found for task: {task_name}")
    return GRADER_REGISTRY[task_name]