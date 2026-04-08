from .task1 import get_scenario as get_task1, grade as grade_task1
from .task2 import get_scenario as get_task2, grade as grade_task2
from .task3 import get_scenario as get_task3, grade as grade_task3
from .task4 import get_scenario as get_task4, grade as grade_task4
from .task5 import get_scenario as get_task5, grade as grade_task5
from .task6 import get_scenario as get_task6, grade as grade_task6
from .task7 import get_scenario as get_task7, grade as grade_task7


TASK_REGISTRY = {
    "task1": get_task1,
    "task2": get_task2,
    "task3": get_task3,
    "task4": get_task4,
    "task5": get_task5,
    "task6": get_task6,
    "task7": get_task7,
}

GRADER_REGISTRY = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
    "task4": grade_task4,
    "task5": grade_task5,
    "task6": grade_task6,
    "task7": grade_task7,
}


def load_task(task_name: str):
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_name]()


def get_grader(task_name: str):
    if task_name not in GRADER_REGISTRY:
        raise ValueError(f"No grader found for task: {task_name}")
    return GRADER_REGISTRY[task_name]