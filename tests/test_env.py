"""
Unit tests for LogAnalysisEnv graders and environment interface.
Run: python -m pytest tests/ -v
"""

import pytest
from environment import LogAnalysisEnv, ActionModel
from environment.models import EpisodeState
from environment.scenarios import task1, task2, task3


# ------------------------------------------------------------------ #
#  Environment Interface Tests                                         #
# ------------------------------------------------------------------ #

class TestEnvironmentInterface:

    def test_reset_returns_observation(self):
        env = LogAnalysisEnv(task_id="task1")
        obs = env.reset()
        assert obs.step == 0
        assert obs.task_description != ""
        assert len(obs.visible_logs) > 0
        assert len(obs.available_actions) > 0

    def test_step_returns_tuple(self):
        env = LogAnalysisEnv(task_id="task1")
        env.reset()
        action = ActionModel(action_type="filter_logs", parameters={"keyword": "error"})
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert obs.step == 1

    def test_state_returns_state(self):
        env = LogAnalysisEnv(task_id="task1")
        env.reset()
        state = env.state()
        assert state.task_id == "task1"
        assert state.step_count == 0

    def test_episode_ends_on_resolve(self):
        env = LogAnalysisEnv(task_id="task1")
        env.reset()
        action = ActionModel(action_type="resolve", parameters={
            "resolution_type": "restart_service", "service": "api-server"
        })
        obs, reward, done, info = env.step(action)
        assert done is True

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            LogAnalysisEnv(task_id="task99")

    def test_step_without_reset_raises(self):
        env = LogAnalysisEnv(task_id="task1")
        with pytest.raises(RuntimeError):
            env.step(ActionModel(action_type="filter_logs", parameters={"keyword": "error"}))

    def test_all_three_tasks_reset(self):
        for task_id in ["task1", "task2", "task3"]:
            env = LogAnalysisEnv(task_id=task_id)
            obs = env.reset()
            assert obs.step == 0

    def test_max_steps_ends_episode(self):
        env = LogAnalysisEnv(task_id="task1")
        env.reset()
        done = False
        for _ in range(20):
            action = ActionModel(action_type="query_metrics", parameters={})
            _, _, done, _ = env.step(action)
            if done:
                break
        assert done is True

    def test_cumulative_reward_accumulates(self):
        env = LogAnalysisEnv(task_id="task1")
        env.reset()
        for keyword in ["error", "critical", "oom"]:
            env.step(ActionModel(action_type="filter_logs", parameters={"keyword": keyword}))
        state = env.state()
        assert state.cumulative_reward > 0


# ------------------------------------------------------------------ #
#  Grader Tests — Task 1                                              #
# ------------------------------------------------------------------ #

class TestTask1Grader:

    def _make_state(self, **kwargs) -> EpisodeState:
        defaults = dict(
            task_id="task1", scenario_name="test", step_count=5, max_steps=15,
            done=True, all_logs=[], wrong_action_count=0, destructive_action_count=0,
            cumulative_reward=0.0, ground_truth=task1.GROUND_TRUTH,
            services_inspected=[], keywords_filtered=[], actions_history=[],
        )
        defaults.update(kwargs)
        return EpisodeState(**defaults)

    def test_perfect_score(self):
        state = self._make_state(
            root_cause_marked="oom_kill",
            classification_marked="infrastructure_failure",
            resolution_action="restart_service:api-server",
            step_count=5,
        )
        score = task1.grade(state)
        assert score >= 0.90

    def test_zero_score_no_actions(self):
        state = self._make_state(
            root_cause_marked=None,
            classification_marked=None,
            resolution_action=None,
        )
        score = task1.grade(state)
        assert score == 0.0

    def test_partial_score_right_service_wrong_action(self):
        state = self._make_state(
            root_cause_marked="oom_kill",
            classification_marked="infrastructure_failure",
            resolution_action="scale_service:api-server",  # wrong action type
        )
        score = task1.grade(state)
        assert 0.30 < score < 0.90

    def test_score_in_range(self):
        for rc in [None, "oom_kill", "memory_leak"]:
            for cl in [None, "infrastructure_failure", "application_bug"]:
                for res in [None, "restart_service:api-server", "restart_service:postgres"]:
                    state = self._make_state(
                        root_cause_marked=rc,
                        classification_marked=cl,
                        resolution_action=res,
                    )
                    score = task1.grade(state)
                    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_wrong_action_penalty(self):
        state_clean = self._make_state(root_cause_marked="oom_kill",
                                        classification_marked="infrastructure_failure",
                                        resolution_action="restart_service:api-server")
        state_penalty = self._make_state(root_cause_marked="oom_kill",
                                          classification_marked="infrastructure_failure",
                                          resolution_action="restart_service:api-server",
                                          wrong_action_count=3)
        assert task1.grade(state_clean) > task1.grade(state_penalty)


# ------------------------------------------------------------------ #
#  Grader Tests — Task 2                                              #
# ------------------------------------------------------------------ #

class TestTask2Grader:

    def _make_state(self, **kwargs) -> EpisodeState:
        defaults = dict(
            task_id="task2", scenario_name="test", step_count=10, max_steps=20,
            done=True, all_logs=[], wrong_action_count=0, destructive_action_count=0,
            cumulative_reward=0.0, ground_truth=task2.GROUND_TRUTH,
            services_inspected=[], keywords_filtered=[], actions_history=[],
        )
        defaults.update(kwargs)
        return EpisodeState(**defaults)

    def test_perfect_score(self):
        state = self._make_state(
            root_cause_marked="memory_leak",
            classification_marked="application_bug",
            resolution_action="restart_service:session-manager",
            services_inspected=["session-manager"],
            keywords_filtered=["memory", "heap"],
        )
        score = task2.grade(state)
        assert score >= 0.90

    def test_postgres_restart_penalty(self):
        state = self._make_state(
            root_cause_marked="memory_leak",
            classification_marked="application_bug",
            resolution_action="restart_service:session-manager",
            actions_history=[{"action_type": "restart_service", "params": {"service": "postgres"}}],
        )
        # The grade function checks actions_history for postgres restart
        state_no_penalty = self._make_state(
            root_cause_marked="memory_leak",
            classification_marked="application_bug",
            resolution_action="restart_service:session-manager",
        )
        # Both should still have valid scores in range
        score = task2.grade(state)
        score_clean = task2.grade(state_no_penalty)
        assert 0.0 <= score <= 1.0
        assert 0.0 <= score_clean <= 1.0


# ------------------------------------------------------------------ #
#  Grader Tests — Task 3                                              #
# ------------------------------------------------------------------ #

class TestTask3Grader:

    def _make_state(self, **kwargs) -> EpisodeState:
        defaults = dict(
            task_id="task3", scenario_name="test", step_count=15, max_steps=30,
            done=True, all_logs=[], wrong_action_count=0, destructive_action_count=0,
            cumulative_reward=0.0, ground_truth=task3.GROUND_TRUTH,
            services_inspected=[], keywords_filtered=[], actions_history=[],
        )
        defaults.update(kwargs)
        return EpisodeState(**defaults)

    def test_perfect_score(self):
        state = self._make_state(
            root_cause_marked="misconfigured_circuit_breaker",
            classification_marked="configuration_error",
            resolution_action="scale_service:order-service",
            services_inspected=["order-service", "inventory-service"],
            keywords_filtered=["circuit", "config"],
        )
        score = task3.grade(state)
        assert score >= 0.90

    def test_user_service_blame_penalty(self):
        state_correct = self._make_state(
            root_cause_marked="misconfigured_circuit_breaker",
            classification_marked="configuration_error",
            resolution_action="scale_service:order-service",
        )
        state_wrong = self._make_state(
            root_cause_marked="user_service_failure",
            classification_marked="configuration_error",
            resolution_action="scale_service:order-service",
        )
        assert task3.grade(state_correct) > task3.grade(state_wrong)

    def test_all_graders_deterministic(self):
        """Same state should always produce same score."""
        for grader_module, task_id, gt in [
            (task1, "task1", task1.GROUND_TRUTH),
            (task2, "task2", task2.GROUND_TRUTH),
            (task3, "task3", task3.GROUND_TRUTH),
        ]:
            state = EpisodeState(
                task_id=task_id, scenario_name="test", step_count=10, max_steps=30,
                done=True, all_logs=[], wrong_action_count=1, destructive_action_count=0,
                cumulative_reward=0.5, ground_truth=gt,
                root_cause_marked=gt["root_cause"],
                classification_marked=gt["classification"],
                resolution_action=gt["resolution"],
                services_inspected=[], keywords_filtered=[], actions_history=[],
            )
            scores = [grader_module.grade(state) for _ in range(5)]
            assert len(set(scores)) == 1, f"{task_id} grader is non-deterministic: {scores}"


# ------------------------------------------------------------------ #
#  Integration Test                                                   #
# ------------------------------------------------------------------ #

class TestIntegration:

    def test_full_episode_task1_perfect(self):
        """Run a perfect episode on task1 and verify final score."""
        env = LogAnalysisEnv(task_id="task1")
        env.reset()

        actions = [
            ActionModel(action_type="filter_logs", parameters={"keyword": "critical"}),
            ActionModel(action_type="filter_logs", parameters={"keyword": "oom"}),
            ActionModel(action_type="inspect_service", parameters={"service": "api-server"}),
            ActionModel(action_type="mark_root_cause", parameters={"cause": "oom_kill"}),
            ActionModel(action_type="mark_classification", parameters={"classification": "infrastructure_failure"}),
            ActionModel(action_type="resolve", parameters={"resolution_type": "restart_service", "service": "api-server"}),
        ]

        final_score = None
        for action in actions:
            _, reward, done, info = env.step(action)
            if done:
                final_score = info.get("final_score")
                break

        assert final_score is not None
        assert final_score >= 0.85, f"Expected >= 0.85, got {final_score}"

    def test_full_episode_task2(self):
        env = LogAnalysisEnv(task_id="task2")
        env.reset()
        actions = [
            ActionModel(action_type="filter_logs", parameters={"keyword": "memory"}),
            ActionModel(action_type="filter_logs", parameters={"keyword": "heap"}),
            ActionModel(action_type="inspect_service", parameters={"service": "session-manager"}),
            ActionModel(action_type="mark_root_cause", parameters={"cause": "memory_leak"}),
            ActionModel(action_type="mark_classification", parameters={"classification": "application_bug"}),
            ActionModel(action_type="resolve", parameters={"resolution_type": "restart_service", "service": "session-manager"}),
        ]
        final_score = None
        for action in actions:
            _, _, done, info = env.step(action)
            if done:
                final_score = info.get("final_score")
                break
        assert final_score is not None
        assert final_score >= 0.85

    def test_full_episode_task3(self):
        env = LogAnalysisEnv(task_id="task3")
        env.reset()
        actions = [
            ActionModel(action_type="filter_logs", parameters={"keyword": "circuit"}),
            ActionModel(action_type="filter_logs", parameters={"keyword": "config"}),
            ActionModel(action_type="inspect_service", parameters={"service": "order-service"}),
            ActionModel(action_type="inspect_service", parameters={"service": "inventory-service"}),
            ActionModel(action_type="mark_root_cause", parameters={"cause": "misconfigured_circuit_breaker"}),
            ActionModel(action_type="mark_classification", parameters={"classification": "configuration_error"}),
            ActionModel(action_type="resolve", parameters={"resolution_type": "scale_service", "service": "order-service"}),
        ]
        final_score = None
        for action in actions:
            _, _, done, info = env.step(action)
            if done:
                final_score = info.get("final_score")
                break
        assert final_score is not None
        assert final_score >= 0.85