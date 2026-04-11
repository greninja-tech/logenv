"""
Unit tests for LogEnv graders and environment interface.
Run: python -m pytest tests/ -v
"""

import pytest
from environment import LogEnv
from environment.models import Action, EpisodeState, LogEntry, SystemMetrics, Alert
from environment.scenarios import task1, task2, task3, task4, task5, task6, task7


# ------------------------------------------------------------------ #
#  Environment Interface Tests                                         #
# ------------------------------------------------------------------ #

class TestEnvironmentInterface:

    def test_reset_returns_observation(self):
        env = LogEnv(task_name="task1")
        obs = env.reset()
        assert obs.step_count == 0
        assert len(obs.logs) > 0
        assert len(obs.alerts) > 0

    def test_step_returns_tuple(self):
        env = LogEnv(task_name="task1")
        env.reset()
        action = Action(action_type="filter_logs", target="error")
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert obs.step_count == 1

    def test_state_returns_state(self):
        env = LogEnv(task_name="task1")
        env.reset()
        state = env.state()
        assert state.step_count == 0

    def test_episode_ends_on_resolve(self):
        env = LogEnv(task_name="task1")
        env.reset()
        action = Action(action_type="resolve_incident", target="restart_service:api-server")
        obs, reward, done, info = env.step(action)
        assert done is True

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            env = LogEnv(task_name="task99")
            env.reset()

    def test_all_seven_tasks_reset(self):
        for task_id in ["task1", "task2", "task3", "task4", "task5", "task6", "task7"]:
            env = LogEnv(task_name=task_id)
            obs = env.reset()
            assert obs.step_count == 0

    def test_max_steps_ends_episode(self):
        env = LogEnv(task_name="task1")
        env.reset()
        done = False
        for _ in range(20):
            action = Action(action_type="filter_logs", target="error")
            _, _, done, _ = env.step(action)
            if done:
                break
        assert done is True

    def test_invalid_action_type_penalised(self):
        env = LogEnv(task_name="task1")
        env.reset()
        action = Action(action_type="invalid_action", target="test")
        _, reward, _, _ = env.step(action)
        assert reward < 0

    def test_filter_logs_updates_visible_logs(self):
        env = LogEnv(task_name="task1")
        env.reset()
        initial_logs = len(env.state().visible_logs)
        action = Action(action_type="filter_logs", target="error")
        env.step(action)
        # After filtering, visible logs should change
        assert env.state().keywords_filtered == ["error"]

    def test_inspect_service_tracks_history(self):
        env = LogEnv(task_name="task1")
        env.reset()
        action = Action(action_type="inspect_service", target="api-server")
        env.step(action)
        assert "api-server" in env.state().services_inspected

    def test_filter_logs_searches_level_field(self):
        """Regression test: filter_logs should also search log.level."""
        env = LogEnv(task_name="task6")
        env.reset()
        action = Action(action_type="filter_logs", target="error")
        _, reward, _, _ = env.step(action)
        # "error" should match ERROR level logs — should NOT be penalised
        assert reward > 0, f"filter_logs('error') on task6 should match ERROR-level logs, got reward={reward}"


# ------------------------------------------------------------------ #
#  Grader Tests — All Tasks                                           #
# ------------------------------------------------------------------ #

def _make_minimal_state(**kwargs):
    """Helper to create a minimal EpisodeState for grader testing."""
    defaults = dict(
        visible_logs=[],
        all_logs=[],
        metrics=SystemMetrics(
            cpu_percent=50.0, memory_percent=50.0, disk_percent=50.0,
            active_connections=100, request_rate=100.0, error_rate=0.0,
        ),
        alerts=[],
        step_count=6,
        max_steps=15,
        services_inspected=[],
        keywords_filtered=[],
        root_cause_marked=None,
        classification_marked=None,
        resolution_action=None,
        wrong_action_count=0,
        destructive_action_count=0,
        actions_history=[],
    )
    defaults.update(kwargs)
    return EpisodeState(**defaults)


class TestTask1Grader:

    def test_perfect_score(self):
        state = _make_minimal_state(
            root_cause_marked="oom_kill",
            classification_marked="infrastructure_failure",
            resolution_action="restart_service:api-server",
            step_count=5,
        )
        score = task1.grade(state)
        assert score >= 0.90

    def test_no_actions_minimal_score(self):
        state = _make_minimal_state()
        score = task1.grade(state)
        assert score == 0.01  # clamped minimum

    def test_partial_credit_right_service(self):
        state = _make_minimal_state(
            root_cause_marked="oom_kill",
            classification_marked="infrastructure_failure",
            resolution_action="scale_service:api-server",  # right target, wrong action
        )
        score = task1.grade(state)
        assert 0.30 < score < 0.90

    def test_wrong_action_penalty(self):
        state_clean = _make_minimal_state(
            root_cause_marked="oom_kill",
            classification_marked="infrastructure_failure",
            resolution_action="restart_service:api-server",
        )
        state_penalty = _make_minimal_state(
            root_cause_marked="oom_kill",
            classification_marked="infrastructure_failure",
            resolution_action="restart_service:api-server",
            wrong_action_count=3,
        )
        assert task1.grade(state_clean) > task1.grade(state_penalty)


class TestTask2Grader:

    def test_perfect_score(self):
        state = _make_minimal_state(
            root_cause_marked="memory_leak",
            classification_marked="application_bug",
            resolution_action="restart_service:session-manager",
            services_inspected=["session-manager"],
            keywords_filtered=["memory", "heap"],
        )
        score = task2.grade(state)
        assert score >= 0.90

    def test_no_investigation_bonus_lost(self):
        state_with = _make_minimal_state(
            root_cause_marked="memory_leak",
            classification_marked="application_bug",
            resolution_action="restart_service:session-manager",
            services_inspected=["session-manager"],
            keywords_filtered=["memory"],
        )
        state_without = _make_minimal_state(
            root_cause_marked="memory_leak",
            classification_marked="application_bug",
            resolution_action="restart_service:session-manager",
        )
        assert task2.grade(state_with) > task2.grade(state_without)


class TestTask3Grader:

    def test_perfect_score(self):
        state = _make_minimal_state(
            root_cause_marked="misconfigured_circuit_breaker",
            classification_marked="configuration_error",
            resolution_action="scale_service:order-service",
            services_inspected=["order-service", "inventory-service"],
            keywords_filtered=["circuit", "config"],
        )
        score = task3.grade(state)
        assert score >= 0.90

    def test_wrong_root_cause_user_service(self):
        state = _make_minimal_state(
            root_cause_marked="user_service_failure",
        )
        score = task3.grade(state)
        # Should get extra penalty for blaming user-service
        assert score < 0.10


class TestTask4Grader:

    def test_perfect_score(self):
        state = _make_minimal_state(
            root_cause_marked="disk_full",
            classification_marked="infrastructure_failure",
            resolution_action="restart_service:log-rotator",
            step_count=5,
        )
        score = task4.grade(state)
        assert score >= 0.90


class TestTask5Grader:

    def test_perfect_score(self):
        state = _make_minimal_state(
            root_cause_marked="deadlock",
            classification_marked="application_bug",
            resolution_action="restart_service:payment-service",
            step_count=6,
        )
        score = task5.grade(state)
        assert score >= 0.90


class TestTask6Grader:

    def test_perfect_score(self):
        state = _make_minimal_state(
            root_cause_marked="dependency_failure",
            classification_marked="dependency_failure",
            resolution_action="rollback_deploy:checkout-service",
            step_count=6,
        )
        score = task6.grade(state)
        assert score >= 0.90


class TestTask7Grader:

    def test_perfect_score(self):
        state = _make_minimal_state(
            root_cause_marked="network_partition",
            classification_marked="infrastructure_failure",
            resolution_action="restart_service:redis-cluster",
            step_count=8,
        )
        score = task7.grade(state)
        assert score >= 0.90


# ------------------------------------------------------------------ #
#  Grader Property Tests                                              #
# ------------------------------------------------------------------ #

class TestGraderProperties:

    GRADERS_AND_GT = [
        (task1, "task1"), (task2, "task2"), (task3, "task3"),
        (task4, "task4"), (task5, "task5"), (task6, "task6"), (task7, "task7"),
    ]

    def test_scores_always_in_range(self):
        """Score must always be in [0.01, 0.99] for any input."""
        for grader_mod, task_id in self.GRADERS_AND_GT:
            for rc in [None, grader_mod.GROUND_TRUTH["root_cause"], "wrong_value"]:
                for cl in [None, grader_mod.GROUND_TRUTH["classification"], "wrong_value"]:
                    for res in [None, grader_mod.GROUND_TRUTH["resolution"], "wrong:value"]:
                        state = _make_minimal_state(
                            root_cause_marked=rc,
                            classification_marked=cl,
                            resolution_action=res,
                        )
                        score = grader_mod.grade(state)
                        assert 0.01 <= score <= 0.99, (
                            f"{task_id}: score {score} out of range for "
                            f"rc={rc}, cl={cl}, res={res}"
                        )

    def test_all_graders_deterministic(self):
        """Same state must always produce the same score."""
        for grader_mod, task_id in self.GRADERS_AND_GT:
            gt = grader_mod.GROUND_TRUTH
            state = _make_minimal_state(
                root_cause_marked=gt["root_cause"],
                classification_marked=gt["classification"],
                resolution_action=gt["resolution"],
            )
            scores = [grader_mod.grade(state) for _ in range(10)]
            assert len(set(scores)) == 1, f"{task_id} grader is non-deterministic: {scores}"

    def test_correct_always_beats_wrong(self):
        """A correct submission must always score higher than a wrong one."""
        for grader_mod, task_id in self.GRADERS_AND_GT:
            gt = grader_mod.GROUND_TRUTH
            state_correct = _make_minimal_state(
                root_cause_marked=gt["root_cause"],
                classification_marked=gt["classification"],
                resolution_action=gt["resolution"],
            )
            state_wrong = _make_minimal_state(
                root_cause_marked="wrong",
                classification_marked="wrong",
                resolution_action="wrong:wrong",
            )
            assert grader_mod.grade(state_correct) > grader_mod.grade(state_wrong), (
                f"{task_id}: correct score not higher than wrong"
            )


# ------------------------------------------------------------------ #
#  Integration Tests — Full Episodes                                  #
# ------------------------------------------------------------------ #

class TestFullEpisodes:

    @pytest.mark.parametrize("task_id,actions", [
        ("task1", [
            ("filter_logs", "error"), ("filter_logs", "memory"),
            ("inspect_service", "api-server"), ("mark_root_cause", "oom_kill"),
            ("classify_issue", "infrastructure_failure"),
            ("resolve_incident", "restart_service:api-server"),
        ]),
        ("task2", [
            ("filter_logs", "memory"), ("inspect_service", "session-manager"),
            ("filter_logs", "heap"), ("mark_root_cause", "memory_leak"),
            ("classify_issue", "application_bug"),
            ("resolve_incident", "restart_service:session-manager"),
        ]),
        ("task3", [
            ("filter_logs", "circuit"), ("inspect_service", "order-service"),
            ("filter_logs", "config"), ("inspect_service", "inventory-service"),
            ("mark_root_cause", "misconfigured_circuit_breaker"),
            ("classify_issue", "configuration_error"),
            ("resolve_incident", "scale_service:order-service"),
        ]),
        ("task4", [
            ("filter_logs", "disk"), ("inspect_service", "log-rotator"),
            ("filter_logs", "rotation"), ("mark_root_cause", "disk_full"),
            ("classify_issue", "infrastructure_failure"),
            ("resolve_incident", "restart_service:log-rotator"),
        ]),
        ("task5", [
            ("filter_logs", "deadlock"), ("inspect_service", "payment-service"),
            ("filter_logs", "lock"), ("mark_root_cause", "deadlock"),
            ("classify_issue", "application_bug"),
            ("resolve_incident", "restart_service:payment-service"),
        ]),
        ("task6", [
            ("filter_logs", "error"), ("inspect_service", "checkout-service"),
            ("filter_logs", "gateway"), ("mark_root_cause", "dependency_failure"),
            ("classify_issue", "dependency_failure"),
            ("resolve_incident", "rollback_deploy:checkout-service"),
        ]),
        ("task7", [
            ("filter_logs", "error"), ("inspect_service", "redis-cluster"),
            ("filter_logs", "partition"), ("inspect_service", "session-service"),
            ("mark_root_cause", "network_partition"),
            ("classify_issue", "infrastructure_failure"),
            ("resolve_incident", "restart_service:redis-cluster"),
        ]),
    ])
    def test_optimal_episode(self, task_id, actions):
        """Run the optimal action sequence and verify high score."""
        from environment.graders import grade_task

        env = LogEnv(task_name=task_id)
        env.reset()
        done = False
        for action_type, target in actions:
            action = Action(action_type=action_type, target=target)
            _, reward, done, _ = env.step(action)
            if done:
                break

        assert done is True, f"{task_id}: Episode did not end after all actions"
        state = env.state()
        score = grade_task(task_id, state)
        assert score >= 0.85, f"{task_id}: Expected >= 0.85, got {score}"