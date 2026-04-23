"""
tests/test_reward.py — Reward function tests.

Spec:
    "Construct a state where agent clearly outperforms greedy (all crises resolved,
     client satisfaction 8, morale 7) and verify reward > 0.15.
     Construct a state where agent performs worse and verify reward < 0."
"""

from __future__ import annotations

import copy

import pytest

from env.state import (
    ProjectState,
    TeamMember,
    Task,
    Crisis,
    StakeholderState,
    CANDOR_LEVEL_HONEST,
)
from reward.counterfactual import (
    project_score,
    counterfactual_reward,
    RECOVERY_WEIGHT,
    CLIENT_WEIGHT,
    MORALE_WEIGHT,
)


def make_resolved_state(
    client_satisfaction: float = 8.0,
    morale: float = 7.0,
    crisis_resolved: bool = True,
) -> ProjectState:
    """Build a state with specified satisfaction and resolution status."""
    member = TeamMember(
        member_id="m1", name="Alice", role="engineer",
        candor=0.9, candor_level=CANDOR_LEVEL_HONEST,
        actual_completion=1.0 if crisis_resolved else 0.2,
        actual_availability=0.9, actual_velocity=0.8,
        inflation_bias=0.02,
        reported_completion=1.0 if crisis_resolved else 0.2,
        reported_availability=0.9,
        assigned_task_ids=["t1"],
        morale=morale,
    )
    task = Task(
        task_id="t1", title="Fix bug", crisis_id="c1",
        assigned_member_id="m1",
        status="done" if crisis_resolved else "in_progress",
        is_critical_path=True, estimated_days=3.0,
        actual_progress=1.0 if crisis_resolved else 0.2,
    )
    crisis = Crisis(
        crisis_id="c1", crisis_type="integration_failure",
        severity=7.0,
        description="Test crisis",
        affected_task_ids=["t1"],
        is_resolved=crisis_resolved,
    )
    return ProjectState(
        team_members=[member],
        tasks=[task],
        crises=[crisis],
        stakeholder=StakeholderState(client_satisfaction=client_satisfaction),
    )


class TestProjectScore:
    """project_score uses all three components from actual state."""

    def test_perfect_state_scores_near_one(self):
        state = make_resolved_state(client_satisfaction=10.0, morale=10.0)
        score = project_score(state)
        assert score > 0.95, f"Perfect state scored {score:.3f}"

    def test_score_uses_recovery_component(self):
        resolved = make_resolved_state(crisis_resolved=True, client_satisfaction=5.0, morale=5.0)
        unresolved = make_resolved_state(crisis_resolved=False, client_satisfaction=5.0, morale=5.0)
        assert project_score(resolved) > project_score(unresolved)

    def test_score_uses_client_component(self):
        high_sat = make_resolved_state(client_satisfaction=9.0, morale=5.0)
        low_sat  = make_resolved_state(client_satisfaction=2.0, morale=5.0)
        assert project_score(high_sat) > project_score(low_sat)

    def test_score_uses_morale_component(self):
        high_morale = make_resolved_state(morale=9.0, client_satisfaction=5.0)
        low_morale  = make_resolved_state(morale=1.0, client_satisfaction=5.0)
        assert project_score(high_morale) > project_score(low_morale)

    def test_weight_constants_sum_to_one(self):
        total = RECOVERY_WEIGHT + CLIENT_WEIGHT + MORALE_WEIGHT
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


class TestCounterfactualReward:
    """reward = agent_score - greedy_score."""

    def test_agent_better_than_greedy_gives_positive_reward(self):
        """
        Spec: "Construct a state where agent clearly outperforms greedy
               (all crises resolved, client satisfaction 8, morale 7)
               and verify reward > 0.15"
        """
        # Agent: everything resolved, high satisfaction, good morale
        agent_state = make_resolved_state(
            crisis_resolved=True, client_satisfaction=8.0, morale=7.0
        )
        # Greedy: nothing resolved, low satisfaction, low morale
        greedy_state = make_resolved_state(
            crisis_resolved=False, client_satisfaction=4.0, morale=4.0
        )

        reward = counterfactual_reward(agent_state, greedy_state)
        assert reward > 0.15, (
            f"Expected reward > 0.15 when agent clearly outperforms greedy, got {reward:.3f}"
        )

    def test_agent_worse_than_greedy_gives_negative_reward(self):
        """
        Spec: "Construct a state where agent performs worse and verify reward < 0"
        """
        # Greedy: better state
        greedy_state = make_resolved_state(
            crisis_resolved=True, client_satisfaction=8.0, morale=7.0
        )
        # Agent: worse state
        agent_state = make_resolved_state(
            crisis_resolved=False, client_satisfaction=3.0, morale=3.0
        )

        reward = counterfactual_reward(agent_state, greedy_state)
        assert reward < 0, (
            f"Expected negative reward when agent underperforms greedy, got {reward:.3f}"
        )

    def test_equal_states_gives_zero_reward(self):
        """Equal final states should produce near-zero reward."""
        state = make_resolved_state(crisis_resolved=True, client_satisfaction=7.0, morale=6.0)
        state_copy = copy.deepcopy(state)
        reward = counterfactual_reward(state, state_copy)
        assert abs(reward) < 1e-9, f"Equal states should give ~0 reward, got {reward:.6f}"

    def test_budget_exhaustion_penalises_agent(self):
        """Budget exhaustion should lower the agent's effective score."""
        agent_state = make_resolved_state(
            crisis_resolved=True, client_satisfaction=8.0, morale=7.0
        )
        greedy_state = make_resolved_state(
            crisis_resolved=False, client_satisfaction=4.0, morale=4.0
        )
        reward_normal    = counterfactual_reward(agent_state, greedy_state, agent_budget_exhausted=False)
        reward_exhausted = counterfactual_reward(agent_state, greedy_state, agent_budget_exhausted=True)
        assert reward_exhausted < reward_normal, (
            "Budget exhaustion should reduce reward"
        )
