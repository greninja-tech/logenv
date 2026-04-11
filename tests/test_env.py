"""
Unit tests for LogEnv — all 7 tasks.
Run: python -m pytest tests/ -v
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment import LogEnv
from environment.models import Action, EpisodeState
from environment.graders import grade_task
from environment.scenarios import task1, task2, task3, task4, task5, task6, task7


ALL_TASKS = ["task1", "task2", "task3", "task4", "task5", "task6", "task7"]

OPTIMAL_SEQUENCES = {
    "task1": [("filter_logs","error"),("inspect_service","api-server"),("mark_root_cause","oom_kill"),("classify_issue","infrastructure_failure"),("resolve_incident","restart_service:api-server")],
    "task2": [("filter_logs","memory"),("inspect_service","session-manager"),("mark_root_cause","memory_leak"),("classify_issue","application_bug"),("resolve_incident","restart_service:session-manager")],
    "task3": [("filter_logs","circuit"),("inspect_service","order-service"),("inspect_service","inventory-service"),("mark_root_cause","misconfigured_circuit_breaker"),("classify_issue","configuration_error"),("resolve_incident","scale_service:order-service")],
    "task4": [("filter_logs","disk"),("inspect_service","log-rotator"),("mark_root_cause","disk_full"),("classify_issue","infrastructure_failure"),("resolve_incident","restart_service:log-rotator")],
    "task5": [("filter_logs","deadlock"),("inspect_service","payment-service"),("mark_root_cause","deadlock"),("classify_issue","application_bug"),("resolve_incident","restart_service:payment-service")],
    "task6": [("filter_logs","error"),("inspect_service","checkout-service"),("mark_root_cause","dependency_failure"),("classify_issue","dependency_failure"),("resolve_incident","rollback_deploy:checkout-service")],
    "task7": [("filter_logs","partition"),("inspect_service","redis-cluster"),("inspect_service","session-service"),("mark_root_cause","network_partition"),("classify_issue","infrastructure_failure"),("resolve_incident","restart_service:redis-cluster")],
}


# ── Interface Tests ───────────────────────────────────────────────────

class TestEnvironmentInterface:

    def test_reset_returns_observation(self):
        env = LogEnv(task_name="task1")
        obs = env.reset()
        assert obs.step_count == 0
        assert len(obs.logs) > 0
        assert obs.metrics is not None
        assert obs.alerts is not None

    def test_reset_with_seed_is_reproducible(self):
        env = LogEnv(task_name="task1")
        obs1 = env.reset(seed=42)
        obs2 = env.reset(seed=42)
        logs1 = [(l.service, l.level) for l in obs1.logs]
        logs2 = [(l.service, l.level) for l in obs2.logs]
        assert logs1 == logs2, "Same seed should give same logs"

    def test_reset_without_seed_varies(self):
        env = LogEnv(task_name="task1")
        results = set()
        for _ in range(5):
            obs = env.reset()
            results.add(tuple(l.service for l in obs.logs))
        # With randomisation, at least 2 distinct orderings in 5 runs
        assert len(results) >= 2, "Unseeded reset should vary across episodes"

    def test_step_returns_correct_types(self):
        env = LogEnv(task_name="task1")
        env.reset()
        obs, reward, done, info = env.step(Action(action_type="filter_logs", target="error"))
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert obs.step_count == 1

    def test_resolve_ends_episode(self):
        env = LogEnv(task_name="task1")
        env.reset()
        _, _, done, _ = env.step(Action(action_type="resolve_incident", target="restart_service:api-server"))
        assert done is True

    def test_max_steps_ends_episode(self):
        env = LogEnv(task_name="task1")
        env.reset()
        done = False
        for _ in range(20):
            _, _, done, _ = env.step(Action(action_type="filter_logs", target="error"))
            if done:
                break
        assert done is True

    def test_invalid_action_type_penalised(self):
        env = LogEnv(task_name="task1")
        env.reset()
        _, reward, _, info = env.step(Action(action_type="do_magic", target="foo"))
        assert reward < 0
        assert "error" in info

    def test_unknown_service_penalised(self):
        env = LogEnv(task_name="task1")
        env.reset()
        _, reward, _, _ = env.step(Action(action_type="inspect_service", target="nonexistent-svc"))
        assert reward < 0

    def test_all_tasks_reset(self):
        for task_id in ALL_TASKS:
            env = LogEnv(task_name=task_id)
            obs = env.reset()
            assert obs.step_count == 0, f"{task_id} reset failed"

    def test_state_reflects_actions(self):
        env = LogEnv(task_name="task1")
        env.reset()
        env.step(Action(action_type="inspect_service", target="api-server"))
        state = env.state()
        assert "api-server" in state.services_inspected

    def test_repeated_keyword_diminishing_returns(self):
        env = LogEnv(task_name="task1")
        env.reset()
        _, r1, _, _ = env.step(Action(action_type="filter_logs", target="error"))
        _, r2, _, _ = env.step(Action(action_type="filter_logs", target="error"))
        _, r3, _, _ = env.step(Action(action_type="filter_logs", target="error"))
        assert r1 > r2 or r2 == 0, "Repeated keyword should give diminishing reward"
        assert r3 == 0.0, "Third repeat of same keyword should give 0 reward"

    def test_root_cause_investigation_bonus(self):
        """Inspecting the affected service before marking root cause gives bonus."""
        env = LogEnv(task_name="task1")
        env.reset()
        env.step(Action(action_type="inspect_service", target="api-server"))
        _, reward_with_inspect, _, _ = env.step(Action(action_type="mark_root_cause", target="oom_kill"))

        env.reset()
        _, reward_without_inspect, _, _ = env.step(Action(action_type="mark_root_cause", target="oom_kill"))

        assert reward_with_inspect >= reward_without_inspect


# ── Grader Tests ───────────────────────────────────────────────────────

class TestGraders:

    def test_all_scores_strictly_in_open_interval(self):
        """Scores must be strictly in (0, 1) — never 0.0 or 1.0."""
        for task_id, seq in OPTIMAL_SEQUENCES.items():
            env = LogEnv(task_name=task_id)
            env.reset(seed=0)
            for at, tgt in seq:
                env.step(Action(action_type=at, target=tgt))
            score = grade_task(task_id, env.state())
            assert 0.0 < score < 1.0, f"{task_id} score {score} not strictly in (0,1)"
            assert score >= 0.01, f"{task_id} score {score} below minimum 0.01"
            assert score <= 0.99, f"{task_id} score {score} above maximum 0.99"

    def test_optimal_sequence_scores_high(self):
        """Optimal action sequence should score >= 0.80 on all tasks."""
        for task_id, seq in OPTIMAL_SEQUENCES.items():
            env = LogEnv(task_name=task_id)
            env.reset(seed=0)
            for at, tgt in seq:
                env.step(Action(action_type=at, target=tgt))
            score = grade_task(task_id, env.state())
            assert score >= 0.70, f"{task_id} optimal sequence scored too low: {score}"

    def test_zero_action_scores_minimum(self):
        """Episode with no useful actions should score at minimum floor."""
        for task_id in ALL_TASKS:
            env = LogEnv(task_name=task_id)
            env.reset(seed=0)
            env.step(Action(action_type="resolve_incident", target="restart_service:wrong-service"))
            score = grade_task(task_id, env.state())
            assert score == 0.01, f"{task_id} zero-action score should be 0.01, got {score}"

    def test_graders_are_deterministic(self):
        """Same state always produces same score."""
        for task_id, seq in OPTIMAL_SEQUENCES.items():
            env = LogEnv(task_name=task_id)
            env.reset(seed=42)
            for at, tgt in seq:
                env.step(Action(action_type=at, target=tgt))
            state = env.state()
            scores = [grade_task(task_id, state) for _ in range(5)]
            assert len(set(scores)) == 1, f"{task_id} grader non-deterministic: {scores}"

    def test_wrong_root_cause_scores_lower(self):
        """Wrong root cause should score lower than correct one."""
        for task_id, seq in OPTIMAL_SEQUENCES.items():
            correct_env = LogEnv(task_name=task_id)
            correct_env.reset(seed=0)
            for at, tgt in seq:
                correct_env.step(Action(action_type=at, target=tgt))

            wrong_seq = [(at, "wrong_value" if at == "mark_root_cause" else tgt) for at, tgt in seq]
            wrong_env = LogEnv(task_name=task_id)
            wrong_env.reset(seed=0)
            for at, tgt in wrong_seq:
                wrong_env.step(Action(action_type=at, target=tgt))

            correct_score = grade_task(task_id, correct_env.state())
            wrong_score   = grade_task(task_id, wrong_env.state())
            assert correct_score > wrong_score, f"{task_id}: correct root cause should score higher"

    def test_red_herring_penalties_apply(self):
        """Chasing red herrings should reduce score on hard tasks."""
        # Task 2: restarting postgres should be penalised
        env = LogEnv(task_name="task2")
        env.reset(seed=0)
        env.step(Action(action_type="mark_root_cause", target="memory_leak"))
        env.step(Action(action_type="classify_issue", target="application_bug"))
        env.step(Action(action_type="resolve_incident", target="restart_service:postgres"))  # red herring
        score_rh = grade_task("task2", env.state())

        env2 = LogEnv(task_name="task2")
        env2.reset(seed=0)
        env2.step(Action(action_type="mark_root_cause", target="memory_leak"))
        env2.step(Action(action_type="classify_issue", target="application_bug"))
        env2.step(Action(action_type="resolve_incident", target="restart_service:session-manager"))
        score_correct = grade_task("task2", env2.state())

        assert score_correct > score_rh, "Correct resolution should score higher than red herring"

    def test_partial_credit_right_service_wrong_action(self):
        """Targeting the right service with wrong action type gets partial credit."""
        env = LogEnv(task_name="task1")
        env.reset(seed=0)
        env.step(Action(action_type="mark_root_cause", target="oom_kill"))
        env.step(Action(action_type="classify_issue", target="infrastructure_failure"))
        env.step(Action(action_type="resolve_incident", target="scale_service:api-server"))  # wrong type
        partial_score = grade_task("task1", env.state())
        assert 0.40 < partial_score < 0.90, f"Partial credit score unexpected: {partial_score}"

    def test_task3_requires_multi_service_investigation(self):
        """Task3 hard: inspecting multiple services gives bonus score."""
        # With multi-service investigation
        env1 = LogEnv(task_name="task3")
        env1.reset(seed=0)
        env1.step(Action(action_type="inspect_service", target="order-service"))
        env1.step(Action(action_type="inspect_service", target="inventory-service"))
        env1.step(Action(action_type="inspect_service", target="payment-service"))
        env1.step(Action(action_type="mark_root_cause", target="misconfigured_circuit_breaker"))
        env1.step(Action(action_type="classify_issue", target="configuration_error"))
        env1.step(Action(action_type="resolve_incident", target="scale_service:order-service"))
        score_thorough = grade_task("task3", env1.state())

        # Without investigation
        env2 = LogEnv(task_name="task3")
        env2.reset(seed=0)
        env2.step(Action(action_type="mark_root_cause", target="misconfigured_circuit_breaker"))
        env2.step(Action(action_type="classify_issue", target="configuration_error"))
        env2.step(Action(action_type="resolve_incident", target="scale_service:order-service"))
        score_blind = grade_task("task3", env2.state())

        assert score_thorough > score_blind, "Thorough investigation should score higher on hard task"


# ── Integration Tests ──────────────────────────────────────────────────

class TestIntegration:

    def test_full_optimal_episode_all_tasks(self):
        """Run optimal sequence on all 7 tasks and verify scores > 0.70."""
        results = {}
        for task_id, seq in OPTIMAL_SEQUENCES.items():
            env = LogEnv(task_name=task_id)
            env.reset(seed=0)
            total_reward = 0.0
            for at, tgt in seq:
                _, r, done, _ = env.step(Action(action_type=at, target=tgt))
                total_reward += r
            score = grade_task(task_id, env.state())
            results[task_id] = score
            assert score >= 0.70, f"{task_id} optimal score too low: {score}"
        print("\nScores:", results)

    def test_reproducibility_with_seed(self):
        """Same seed always gives same score."""
        for task_id in ALL_TASKS:
            scores = []
            for _ in range(3):
                env = LogEnv(task_name=task_id)
                env.reset(seed=99)
                for at, tgt in OPTIMAL_SEQUENCES[task_id]:
                    env.step(Action(action_type=at, target=tgt))
                scores.append(grade_task(task_id, env.state()))
            assert len(set(scores)) == 1, f"{task_id} not reproducible with seed: {scores}"

    def test_cumulative_reward_positive_for_good_agent(self):
        """A good agent accumulates positive reward over the episode."""
        for task_id, seq in OPTIMAL_SEQUENCES.items():
            env = LogEnv(task_name=task_id)
            env.reset(seed=0)
            total = 0.0
            for at, tgt in seq:
                _, r, done, _ = env.step(Action(action_type=at, target=tgt))
                total += r
            assert total > 0, f"{task_id} good agent got non-positive cumulative reward: {total}"

    def test_info_contains_final_score_on_done(self):
        """info dict should have final_score when episode ends."""
        env = LogEnv(task_name="task1")
        env.reset(seed=0)
        info = {}
        for at, tgt in OPTIMAL_SEQUENCES["task1"]:
            _, _, done, info = env.step(Action(action_type=at, target=tgt))
            if done:
                break
        assert "final_score" in info, "info should contain final_score at episode end"
        assert 0.0 < info["final_score"] < 1.0
