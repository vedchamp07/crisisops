"""
reward/metrics.py — Secondary evaluation metrics for CrisisOps episodes.

Spec: "cross_verification_rate, actions_to_recovery, secondary metrics"

These metrics are logged alongside the primary counterfactual reward and
exposed in the info dict returned by step().  They are useful for understanding
agent behaviour without providing direct training signal.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from env.state import ProjectState
from reward.counterfactual import cross_verification_rate as _cross_verify_rate


def cross_verification_rate(state: ProjectState) -> float:
    """
    Fraction of member queries that used observable signals vs. self-reports.

    Spec: "cross_verification_rate = query_observable_signals_calls /
           total_member_query_calls — track this per episode and expose in info"

    A high rate indicates the agent is being appropriately sceptical of
    self-reports, which correlates with detecting deceptive members.
    """
    return _cross_verify_rate(state)


def actions_to_recovery(state: ProjectState) -> Optional[int]:
    """
    Number of actions taken to resolve all crises, or None if not all resolved.

    Lower is better for an efficient agent.  The greedy PM baseline rarely
    achieves full recovery, so this metric mainly applies to capable agents.
    """
    if any(not c.is_resolved for c in state.crises):
        return None
    return len(state.actions_used)


def budget_efficiency(state: ProjectState) -> float:
    """
    Fraction of initial budget consumed by non-free actions.

    Lower means the agent accomplished more with less expensive actions.
    Computed as: (INITIAL_BUDGET - budget_remaining) / INITIAL_BUDGET
    """
    from env.state import INITIAL_BUDGET
    spent = INITIAL_BUDGET - state.budget_remaining
    return spent / INITIAL_BUDGET if INITIAL_BUDGET > 0 else 0.0


def deception_detection_score(state: ProjectState) -> float:
    """
    Estimate how well the agent detected deceptive members.

    Proxy metric: for each member with candor < 0.5 (optimism_bias or
    self_preservation), check whether the agent called query_observable_signals
    for them at least once.  Returns fraction of deceptive members cross-checked.

    This is an upper-bound estimate — it can't verify what the agent *did*
    with the signals, only that it retrieved them.
    """
    deceptive = [
        m for m in state.team_members
        if m.candor < 0.5
    ]
    if not deceptive:
        return 1.0  # no deceptive members → trivially detected

    # We can't reconstruct per-member signal queries from the aggregate count,
    # so we use the cross_verify_rate as a proxy.
    return _cross_verify_rate(state)


def compute_all_metrics(state: ProjectState) -> Dict[str, object]:
    """
    Compute and return all secondary metrics as a single dict.

    Used by calibrate.py and the training loop for comprehensive logging.
    """
    return {
        "cross_verification_rate": cross_verification_rate(state),
        "actions_to_recovery": actions_to_recovery(state),
        "budget_efficiency": budget_efficiency(state),
        "deception_detection_score": deception_detection_score(state),
        "client_satisfaction_final": state.stakeholder.client_satisfaction,
        "exec_support_final": state.stakeholder.exec_support,
        "crises_resolved": sum(1 for c in state.crises if c.is_resolved),
        "total_crises": len(state.crises),
        "drift_events_acknowledged": sum(
            1 for e in state.active_drift_events if e.acknowledged
        ),
        "morale_avg_final": state.team_morale_avg(),
        "terminated_by_budget": state.terminated_by_budget,
    }
