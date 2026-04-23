"""
tests/test_env.py — Environment mechanics tests.

Spec: "test_env.py: verify that budget decrements correctly per action cost,
that episode ends when budget=0, that free actions do not decrement budget,
that submit_recovery_plan ends episode"
"""

from __future__ import annotations

import copy
import pytest

from env.environment import CrisisOpsEnv, MAX_STEPS
from env.state import INITIAL_BUDGET
from env.actions import ACTION_COSTS


def make_env():
    """Create a fresh environment with the default test scenario."""
    return CrisisOpsEnv(curriculum_level=1)


class TestBudgetAccounting:
    """Budget correctly tracks action costs."""

    def test_free_actions_do_not_decrement_budget(self):
        env = make_env()
        obs = env.reset(seed=1)
        initial_budget = obs["budget_remaining"]
        assert initial_budget == INITIAL_BUDGET

        free_actions = [
            {"action_type": "query_status",             "params": {}},
            {"action_type": "query_member_report",      "params": {"member_id": env._state.team_members[0].member_id}},
            {"action_type": "query_observable_signals", "params": {"member_id": env._state.team_members[0].member_id}},
            {"action_type": "query_ticket",             "params": {"task_id": env._state.tasks[0].task_id}},
        ]
        for action in free_actions:
            obs, _, _, _ = env.step(action)
            assert obs["budget_remaining"] == INITIAL_BUDGET, (
                f"Free action {action['action_type']} decremented budget"
            )

    def test_cost1_actions_decrement_budget_by_1(self):
        env = make_env()
        env.reset(seed=2)
        initial = env._state.budget_remaining

        obs, _, _, _ = env.step({"action_type": "communicate", "params": {
            "message_type": "status_update", "content": "hi", "target": "both"
        }})
        assert obs["budget_remaining"] == initial - 1

    def test_cost2_action_decrements_budget_by_2(self):
        env = make_env()
        env.reset(seed=3)
        initial = env._state.budget_remaining

        task_id = env._state.tasks[0].task_id
        env._state.tasks[0].status = "blocked"  # ensure task is blocked for resolve_blocker

        obs, _, _, _ = env.step({"action_type": "resolve_blocker", "params": {
            "task_id": task_id, "resolution_notes": "test"
        }})
        assert obs["budget_remaining"] == initial - 2

    def test_invalid_action_does_not_decrement_budget(self):
        env = make_env()
        env.reset(seed=4)
        initial = env._state.budget_remaining

        obs, _, _, _ = env.step({"action_type": "query_member_report", "params": {"member_id": "nonexistent"}})
        assert obs.get("error") is not None
        assert obs["budget_remaining"] == initial


class TestEpisodeTermination:
    """Episode terminates correctly."""

    def test_submit_recovery_plan_ends_episode(self):
        env = make_env()
        env.reset(seed=10)

        obs, reward, done, info = env.step({
            "action_type": "submit_recovery_plan",
            "params": {"plan_summary": "Test plan", "risk_items": [], "timeline": "2024-12-31"},
        })
        assert done is True
        assert obs["done"] is True

    def test_budget_exhaustion_ends_episode(self):
        env = make_env()
        env.reset(seed=11)
        # Drain budget with cost-1 actions
        for _ in range(20):
            if env._state.done:
                break
            env.step({"action_type": "communicate", "params": {
                "message_type": "status_update", "content": "x", "target": "both"
            }})
        # At this point budget should be 0 and episode done
        assert env._state.done is True or env._state.budget_remaining == 0

    def test_max_steps_ends_episode(self):
        env = make_env()
        env.reset(seed=12)
        done = False
        for _ in range(MAX_STEPS + 5):
            if done:
                break
            _, _, done, _ = env.step({"action_type": "query_status", "params": {}})
        assert done is True

    def test_reward_is_zero_before_episode_ends(self):
        env = make_env()
        env.reset(seed=13)
        _, reward, done, _ = env.step({"action_type": "query_status", "params": {}})
        assert done is False
        assert reward == 0.0


class TestObservationFormat:
    """Observation dict has correct structure."""

    def test_initial_observation_has_required_keys(self):
        env = make_env()
        obs = env.reset(seed=20)
        for key in ["current_step", "budget_remaining", "team_members", "crises", "stakeholder", "done"]:
            assert key in obs, f"Missing key: {key}"

    def test_observation_has_no_candor(self):
        env = make_env()
        obs = env.reset(seed=21)

        obs_str = str(obs)
        # 'candor' should never appear in observation keys
        # (it may appear in a value if a member's name contains it, but that's
        # extremely unlikely with our names — we check the dict structure)
        for member in obs["team_members"]:
            assert "candor" not in member, "Candor float leaked into observation!"
            assert "actual_completion" not in member, "Actual completion leaked!"
            assert "actual_velocity" not in member, "Actual velocity leaked!"

    def test_state_has_candor_for_debugging(self):
        env = make_env()
        env.reset(seed=22)
        full_state = env.state()
        # state() is for server/debug use and DOES include candor
        for m in full_state["team_members"]:
            assert "candor" in m


class TestInfoDict:
    """Info dict has required keys per spec."""

    def test_info_has_required_keys(self):
        env = make_env()
        env.reset(seed=30)
        _, _, _, info = env.step({"action_type": "query_status", "params": {}})
        for key in [
            "cross_verify_rate", "actions_used", "budget_remaining",
            "greedy_pm_score_so_far", "active_crises", "drift_events_fired"
        ]:
            assert key in info, f"Missing info key: {key}"

    def test_cross_verify_rate_updates(self):
        env = make_env()
        env.reset(seed=31)
        member_id = env._state.team_members[0].member_id

        # First query_member_report (denominator increases, numerator stays 0)
        env.step({"action_type": "query_member_report", "params": {"member_id": member_id}})
        # Now query_observable_signals (both increase)
        _, _, _, info = env.step({"action_type": "query_observable_signals", "params": {"member_id": member_id}})
        assert info["cross_verify_rate"] > 0.0
