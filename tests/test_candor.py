"""
tests/test_candor.py — Candor system tests.

Spec:
    "for candor=0.1, verify reported_completion > actual_completion + 0.3
     in 90%+ of samples.
     For candor=0.9, verify |reported - actual| < 0.15 in 90%+ of samples.
     Verify observable signals are derived from actual state only."
"""

from __future__ import annotations

import random

import pytest

from env.candor import (
    compute_reported_completion,
    sample_inflation_bias,
    get_observable_signals,
    compute_ticket_age_days,
    compute_commits_last_72h,
    INFLATION_BIAS_PARAMS,
    VELOCITY_STALL_THRESHOLD,
)
from env.state import (
    TeamMember,
    ProjectState,
    Task,
    StakeholderState,
    CANDOR_LEVEL_SELF_PRESERVATION,
    CANDOR_LEVEL_HONEST,
    CANDOR_LEVEL_OPTIMISM_BIAS,
)


def make_member(
    candor: float,
    candor_level: str,
    actual_completion: float = 0.3,
    actual_velocity: float = 0.5,
    actual_availability: float = 0.8,
) -> TeamMember:
    """Build a TeamMember for testing with explicit candor/level."""
    rng = random.Random(42)
    bias = sample_inflation_bias(candor_level, rng)
    reported = compute_reported_completion(actual_completion, candor, bias)
    return TeamMember(
        member_id="test_member",
        name="Test",
        role="engineer",
        candor=candor,
        candor_level=candor_level,
        actual_completion=actual_completion,
        actual_availability=actual_availability,
        actual_velocity=actual_velocity,
        inflation_bias=bias,
        reported_completion=reported,
        reported_availability=0.8,
        assigned_task_ids=["t1"],
    )


def make_state_with_member(member: TeamMember) -> ProjectState:
    """Wrap a member in a minimal ProjectState for signal computation."""
    task = Task(
        task_id="t1",
        title="Test task",
        crisis_id=None,
        assigned_member_id=member.member_id,
        status="in_progress",
        is_critical_path=True,
        estimated_days=3.0,
        actual_progress=member.actual_completion,
    )
    return ProjectState(
        team_members=[member],
        tasks=[task],
        crises=[],
        stakeholder=StakeholderState(),
    )


class TestLowCandorOverreports:
    """Low-candor members must over-report completion significantly."""

    def test_low_candor_overreports_in_90pct_of_samples(self):
        """
        Spec: "for candor=0.1, verify reported_completion > actual_completion + 0.3
               in 90%+ of samples"
        """
        rng = random.Random(0)
        actual = 0.2
        candor = 0.1
        level = CANDOR_LEVEL_SELF_PRESERVATION
        n_samples = 200
        overreport_threshold = 0.3

        hits = 0
        for _ in range(n_samples):
            bias = sample_inflation_bias(level, rng)
            reported = compute_reported_completion(actual, candor, bias)
            if reported > actual + overreport_threshold:
                hits += 1

        hit_rate = hits / n_samples
        assert hit_rate >= 0.90, (
            f"Low-candor overreport rate {hit_rate:.2%} < 90% "
            f"(reported - actual > {overreport_threshold})"
        )


class TestHighCandorAccuracy:
    """High-candor members must report near-truth."""

    def test_high_candor_accurate_in_90pct_of_samples(self):
        """
        Spec: "For candor=0.9, verify |reported - actual| < 0.15 in 90%+ of samples"
        """
        rng = random.Random(1)
        actual = 0.5
        candor = 0.9
        level = CANDOR_LEVEL_HONEST
        n_samples = 200
        accuracy_threshold = 0.15

        hits = 0
        for _ in range(n_samples):
            bias = sample_inflation_bias(level, rng)
            reported = compute_reported_completion(actual, candor, bias)
            if abs(reported - actual) < accuracy_threshold:
                hits += 1

        hit_rate = hits / n_samples
        assert hit_rate >= 0.90, (
            f"High-candor accuracy rate {hit_rate:.2%} < 90% "
            f"(|reported - actual| < {accuracy_threshold})"
        )


class TestObservableSignalsDerivedFromActual:
    """Observable signals must never reflect reported state."""

    def test_ticket_age_derived_from_actual_velocity(self):
        """
        A stalled member (actual_velocity=0) should have high ticket age
        regardless of reported_completion.
        """
        member = make_member(
            candor=0.1,
            candor_level=CANDOR_LEVEL_SELF_PRESERVATION,
            actual_completion=0.1,
            actual_velocity=0.0,
        )
        # Force reported completion high to simulate deception
        member.reported_completion = 0.95
        member.ticket_last_changed_step = 0

        state = make_state_with_member(member)
        state.current_step = 10

        # Ticket age should be > 0 since no actual change at step 0
        age = compute_ticket_age_days(member, current_step=10)
        assert age > 0, "Stalled member should have positive ticket age"

    def test_commits_zero_when_actual_velocity_zero(self):
        """A stalled member must return 0 commits regardless of reported state."""
        member = make_member(
            candor=0.1,
            candor_level=CANDOR_LEVEL_SELF_PRESERVATION,
            actual_velocity=0.0,
        )
        member.reported_completion = 0.99  # deceptive

        commits = compute_commits_last_72h(member)
        assert commits == 0, "Zero actual velocity must produce 0 commits signal"

    def test_commits_positive_when_actual_velocity_positive(self):
        """An active member must show positive commit count."""
        member = make_member(
            candor=0.9,
            candor_level=CANDOR_LEVEL_HONEST,
            actual_velocity=0.8,
        )
        commits = compute_commits_last_72h(member)
        assert commits > 0, "Positive actual velocity must produce > 0 commits"

    def test_signals_dict_has_correct_keys(self):
        """get_observable_signals must return all three required keys."""
        member = make_member(0.5, CANDOR_LEVEL_OPTIMISM_BIAS)
        state = make_state_with_member(member)
        signals = get_observable_signals(member, state)
        assert "ticket_age_days" in signals
        assert "commits_last_72h" in signals
        assert "peer_mentions" in signals

    def test_signals_do_not_include_candor(self):
        """Observable signals must never include the candor float."""
        member = make_member(0.1, CANDOR_LEVEL_SELF_PRESERVATION)
        state = make_state_with_member(member)
        signals = get_observable_signals(member, state)
        assert "candor" not in signals
        assert "reported_completion" not in signals
        assert "actual_completion" not in signals

    def test_deceptive_member_has_signal_contradiction(self):
        """
        A deceptive member (low candor, stalled actual velocity) should show
        a contradiction: high reported_completion but high ticket age / 0 commits.
        """
        member = make_member(
            candor=0.1,
            candor_level=CANDOR_LEVEL_SELF_PRESERVATION,
            actual_completion=0.1,
            actual_velocity=VELOCITY_STALL_THRESHOLD,
        )
        member.reported_completion = 0.9  # over-reported
        member.ticket_last_changed_step = 0
        state = make_state_with_member(member)
        state.current_step = 8

        signals = get_observable_signals(member, state)
        # Contradiction: high report but stalled signals
        assert member.reported_completion > 0.5
        assert signals["commits_last_72h"] == 0
        assert signals["ticket_age_days"] > 0
