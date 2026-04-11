import random
from typing import Tuple, Dict, Any, List

from .models import Observation, Action, Reward, EpisodeState
from .scenarios import load_task, get_grader


class LogEnv:
    def __init__(self, task_name: str = "task1"):
        self.task_name = task_name
        self.task_data = None
        self.state_data = None
        self.grader = None
        self._rng = random.Random()

    # ---------------- RESET ----------------
    def reset(self, seed: int = None) -> Observation:
        """
        Reset the environment.
        seed: optional int for reproducible log ordering.
              If None, shuffles differently each episode.
        """
        self.task_data = load_task(self.task_name)
        self.grader = get_grader(self.task_name)

        self._rng.seed(seed)
        all_logs = self._shuffle_logs(self.task_data["all_logs"])

        # Initial partial visibility
        initial_logs = all_logs[:5]

        self.state_data = EpisodeState(
            visible_logs=initial_logs,
            all_logs=all_logs,
            metrics=self.task_data["metrics"],
            alerts=self.task_data["alerts"],
            step_count=0,
            max_steps=self.task_data["max_steps"],
            services_inspected=[],
            keywords_filtered=[],
            root_cause_marked=None,
            classification_marked=None,
            resolution_action=None,
            wrong_action_count=0,
            destructive_action_count=0,
            actions_history=[]
        )

        return self._get_observation()


    def _shuffle_logs(self, all_logs):
        """
        Shuffle INFO/noise logs while keeping WARNING/ERROR/CRITICAL
        logs in their original chronological order.
        This provides episode variation without breaking causal chains.
        """
        signal_levels = {"WARNING", "ERROR", "CRITICAL"}
        noise  = [l for l in all_logs if l.level not in signal_levels]
        signal = [l for l in all_logs if l.level in signal_levels]
        self._rng.shuffle(noise)
        result = list(signal)
        for n in noise:
            result.insert(self._rng.randint(0, len(result)), n)
        return result

    # ---------------- STEP ----------------
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        self.state_data.step_count += 1
        self.state_data.actions_history.append({
            "action_type": action.action_type,
            "target": action.target
        })

        action_type = action.action_type
        target = action.target

        if action_type == "filter_logs":
            reward += self._handle_filter_logs(target)

        elif action_type == "inspect_service":
            reward += self._handle_inspect_service(target)

        elif action_type == "mark_root_cause":
            reward += self._handle_root_cause(target)

        elif action_type == "classify_issue":
            reward += self._handle_classification(target)

        elif action_type == "resolve_incident":
            reward += self._handle_resolution(target)
            done = True  # resolving ends episode

        else:
            reward -= 0.1
            self.state_data.wrong_action_count += 1
            info["error"] = f"unknown action_type: {action_type}"

        # Step limit
        if self.state_data.step_count >= self.state_data.max_steps:
            done = True

        # Final grading on episode end
        if done:
            final_score = self.grader(self.state_data)
            reward += final_score
            info["final_score"] = final_score

        return self._get_observation(), round(reward, 4), done, info

    # ---------------- STATE ----------------
    def state(self) -> EpisodeState:
        return self.state_data

    # ---------------- OBSERVATION ----------------
    def _get_observation(self) -> Observation:
        return Observation(
            logs=self.state_data.visible_logs,
            metrics=self.state_data.metrics,
            alerts=self.state_data.alerts,
            step_count=self.state_data.step_count,
        )

    # ---------------- ACTION HANDLERS ----------------

    def _handle_filter_logs(self, keyword: str) -> float:
        if not keyword:
            self.state_data.wrong_action_count += 1
            return -0.05

        keyword = keyword.lower()
        filtered = [
            log for log in self.state_data.all_logs
            if keyword in log.message.lower()
            or keyword in log.service.lower()
            or keyword in log.level.lower()
        ]

        if not filtered:
            self.state_data.wrong_action_count += 1
            return -0.05

        self.state_data.visible_logs = filtered[:20]
        self.state_data.keywords_filtered.append(keyword)

        if keyword in ["memory", "heap", "oom", "circuit", "error",
                        "disk", "deadlock", "partition", "lock", "gateway"]:
            return 0.1
        return 0.05

    def _handle_inspect_service(self, service: str) -> float:
        if not service:
            self.state_data.wrong_action_count += 1
            return -0.05

        logs = [
            log for log in self.state_data.all_logs
            if log.service == service
        ]

        if not logs:
            self.state_data.wrong_action_count += 1
            return -0.05

        self.state_data.visible_logs = logs[:20]

        if service not in self.state_data.services_inspected:
            self.state_data.services_inspected.append(service)
            return 0.1

        return 0.02

    def _handle_root_cause(self, cause: str) -> float:
        self.state_data.root_cause_marked = cause
        gt = self.task_data["ground_truth"]["root_cause"]

        if cause == gt:
            return 0.3
        else:
            self.state_data.wrong_action_count += 1
            return -0.1

    def _handle_classification(self, classification: str) -> float:
        self.state_data.classification_marked = classification
        gt = self.task_data["ground_truth"]["classification"]

        if classification == gt:
            return 0.2
        else:
            self.state_data.wrong_action_count += 1
            return -0.1

    def _handle_resolution(self, action: str) -> float:
        self.state_data.resolution_action = action
        gt = self.task_data["ground_truth"]["resolution"]

        if action == gt:
            return 0.5
        elif action and self.task_data["ground_truth"]["affected_service"] in action:
            return 0.2  # partial credit
        else:
            self.state_data.destructive_action_count += 1
            return -0.2
