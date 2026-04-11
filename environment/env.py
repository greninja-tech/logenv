import random
from typing import Tuple, Dict, Any, List

from .models import Observation, Action, Reward, EpisodeState
from .scenarios import load_task, get_grader


class LogEnv:
    """
    OpenEnv-compliant Log Analysis & Incident Response environment.

    Key improvements over baseline:
    - reset(seed) for reproducible randomisation
    - Log shuffling: noise logs randomised each episode, critical logs preserved in order
    - Step-level reward signal during investigation (not just at end)
    - Diminishing returns on repeated same-keyword searches
    - Graded partial credit for investigation quality
    """

    def __init__(self, task_name: str = "task1"):
        self.task_name = task_name
        self.task_data = None
        self.state_data = None
        self.grader = None
        self._rng = random.Random()

    # ── RESET ──────────────────────────────────────────────────────────
    def reset(self, seed: int = None) -> Observation:
        """
        Reset the environment for a new episode.
        seed: optional integer for reproducible log shuffling.
              If None, randomises differently each episode.
        """
        self.task_data = load_task(self.task_name)
        self.grader = get_grader(self.task_name)

        if seed is not None:
            self._rng.seed(seed)
        else:
            self._rng.seed(None)

        shuffled_logs = self._shuffle_logs(self.task_data["all_logs"])

        self.state_data = EpisodeState(
            visible_logs=shuffled_logs[:5],
            all_logs=shuffled_logs,
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
            actions_history=[],
        )

        return self._get_observation()

    def _shuffle_logs(self, all_logs):
        """
        Shuffle noise/INFO logs while preserving the chronological order
        of WARNING/ERROR/CRITICAL logs (the signal logs stay in sequence).
        This makes each episode unique without destroying the causal chain.
        """
        signal_levels = {"WARNING", "ERROR", "CRITICAL"}
        noise_logs = [l for l in all_logs if l.level not in signal_levels]
        signal_logs = [l for l in all_logs if l.level in signal_levels]

        # Shuffle only the noise logs
        self._rng.shuffle(noise_logs)

        # Interleave: put noise logs at random positions among signal logs
        result = list(signal_logs)
        for noise_log in noise_logs:
            insert_pos = self._rng.randint(0, len(result))
            result.insert(insert_pos, noise_log)

        return result

    # ── STEP ───────────────────────────────────────────────────────────
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        reward = 0.0
        done = False
        info = {}

        self.state_data.step_count += 1
        self.state_data.actions_history.append({
            "action_type": action.action_type,
            "target": action.target,
        })

        at = action.action_type
        target = action.target

        if at == "filter_logs":
            reward += self._handle_filter_logs(target)

        elif at == "inspect_service":
            reward += self._handle_inspect_service(target)

        elif at == "mark_root_cause":
            reward += self._handle_root_cause(target)

        elif at == "classify_issue":
            reward += self._handle_classification(target)

        elif at == "resolve_incident":
            reward += self._handle_resolution(target)
            done = True

        else:
            reward -= 0.10
            self.state_data.wrong_action_count += 1
            info["error"] = f"unknown action_type: {at}"

        # Step limit
        if self.state_data.step_count >= self.state_data.max_steps:
            done = True

        # Final grading on episode end — add grader score to reward
        if done:
            final_score = self.grader(self.state_data)
            reward += final_score
            info["final_score"] = final_score

        return self._get_observation(), round(reward, 4), done, info

    # ── STATE ──────────────────────────────────────────────────────────
    def state(self) -> EpisodeState:
        return self.state_data

    # ── OBSERVATION ────────────────────────────────────────────────────
    def _get_observation(self) -> Observation:
        return Observation(
            logs=self.state_data.visible_logs,
            metrics=self.state_data.metrics,
            alerts=self.state_data.alerts,
            step_count=self.state_data.step_count,
        )

    # ── ACTION HANDLERS ────────────────────────────────────────────────

    def _handle_filter_logs(self, keyword: str) -> float:
        if not keyword:
            self.state_data.wrong_action_count += 1
            return -0.05

        keyword = keyword.lower()

        # Diminishing returns: repeating same keyword gives less reward
        prior_uses = self.state_data.keywords_filtered.count(keyword)
        if prior_uses >= 2:
            return 0.0  # no reward for spamming same keyword
        if prior_uses == 1:
            multiplier = 0.3
        else:
            multiplier = 1.0

        filtered = [
            log for log in self.state_data.all_logs
            if keyword in log.message.lower() or keyword in log.service.lower()
        ]

        if not filtered:
            self.state_data.wrong_action_count += 1
            return -0.05

        self.state_data.visible_logs = filtered[:20]
        self.state_data.keywords_filtered.append(keyword)

        # Reward scales with relevance of keyword to the actual incident
        gt = self.task_data["ground_truth"]
        high_value = ["memory", "heap", "oom", "circuit", "deadlock",
                       "disk", "partition", "lock", "dependency", "gateway"]
        base = 0.10 if keyword in high_value else 0.05
        return round(base * multiplier, 4)

    def _handle_inspect_service(self, service: str) -> float:
        if not service:
            self.state_data.wrong_action_count += 1
            return -0.05

        logs = [log for log in self.state_data.all_logs if log.service == service]

        if not logs:
            self.state_data.wrong_action_count += 1
            return -0.05

        self.state_data.visible_logs = logs[:20]

        # First-time inspection of the root cause service gives higher reward
        gt_service = self.task_data["ground_truth"]["affected_service"]
        if service not in self.state_data.services_inspected:
            self.state_data.services_inspected.append(service)
            return 0.15 if service == gt_service else 0.08
        else:
            return 0.02  # repeat inspection: tiny reward

    def _handle_root_cause(self, cause: str) -> float:
        self.state_data.root_cause_marked = cause
        gt = self.task_data["ground_truth"]["root_cause"]

        if cause == gt:
            # Bonus if agent inspected the right service first
            gt_service = self.task_data["ground_truth"]["affected_service"]
            if gt_service in self.state_data.services_inspected:
                return 0.35  # investigated properly then identified
            return 0.30
        else:
            self.state_data.wrong_action_count += 1
            return -0.10

    def _handle_classification(self, classification: str) -> float:
        self.state_data.classification_marked = classification
        gt = self.task_data["ground_truth"]["classification"]

        if classification == gt:
            return 0.20
        else:
            self.state_data.wrong_action_count += 1
            return -0.10

    def _handle_resolution(self, action: str) -> float:
        self.state_data.resolution_action = action
        gt = self.task_data["ground_truth"]["resolution"]
        gt_service = self.task_data["ground_truth"]["affected_service"]

        if action == gt:
            return 0.50
        elif action and gt_service in action:
            return 0.20  # right service, wrong resolution type
        else:
            self.state_data.destructive_action_count += 1
            return -0.20
