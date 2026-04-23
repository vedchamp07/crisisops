"""
reward/counterfactual.py — project_score() function and counterfactual reward.

Spec: "COUNTERFACTUAL REWARD (counterfactual.py)"

project_score(state) = RECOVERY_WEIGHT   * recovery_pct
                     + CLIENT_WEIGHT     * client_satisfaction_normalized
                     + MORALE_WEIGHT     * team_morale_avg_normalized

All three components computed from ACTUAL state, NOT reported state.

reward = project_score(agent_final_state) - project_score(greedy_PM_final_state)

Secondary metric:
    cross_verification_rate = query_observable_signals_calls / total_member_query_calls
"""

from __future__ import annotations

from env.state import ProjectState

# ---------------------------------------------------------------------------
# Reward weight constants — spec: "The counterfactual reward weights (0.5, 0.3, 0.2)
# must be defined as named constants"
# ---------------------------------------------------------------------------
RECOVERY_WEIGHT = 0.5
CLIENT_WEIGHT   = 0.3
MORALE_WEIGHT   = 0.2

# Normalisation factor for team morale (stored on 0–10 scale)
MORALE_MAX = 10.0

# Calibration target ranges — spec: named constants
GREEDY_SCORE_TARGET_LOW  = 0.45
GREEDY_SCORE_TARGET_HIGH = 0.55
ORACLE_SCORE_TARGET_LOW  = 0.70
ORACLE_SCORE_TARGET_HIGH = 0.80
CALIBRATION_GAP_LOW      = 0.20
CALIBRATION_GAP_HIGH     = 0.35


def project_score(state: ProjectState) -> float:
    """
    Compute the composite project score from ACTUAL state.

    All three components are in [0, 1] before weighting.  The weighted sum
    is therefore also in [0, 1] for a well-calibrated episode.

    Spec formula:
        0.5 * recovery_pct + 0.3 * client_satisfaction_normalized + 0.2 * team_morale_avg

    Uses actual state exclusively:
        recovery_pct                → fraction of crises resolved
        client_satisfaction_normalized → state.stakeholder.client_satisfaction / 10
        team_morale_avg             → avg member morale / 10
    """
    recovery  = state.recovery_pct()
    client    = state.client_satisfaction_normalized()
    morale    = state.team_morale_avg() / MORALE_MAX

    # Clamp each component to [0, 1] for numerical safety
    recovery = max(0.0, min(1.0, recovery))
    client   = max(0.0, min(1.0, client))
    morale   = max(0.0, min(1.0, morale))

    return RECOVERY_WEIGHT * recovery + CLIENT_WEIGHT * client + MORALE_WEIGHT * morale


def counterfactual_reward(
    agent_final_state: ProjectState,
    greedy_final_state: ProjectState,
    agent_budget_exhausted: bool = False,
) -> float:
    """
    Compute the counterfactual reward for the RL training signal.

    Spec: "reward = project_score(agent_final_state) - project_score(greedy_PM_final_state)"

    The greedy PM is simulated to completion on a deep-copied initial state that
    is identical to the agent's initial state (this is handled by CrisisOpsEnv).

    A budget-exhaustion penalty of 0.30 is applied to the agent score when the
    agent runs out of budget without submitting a recovery plan.

    Returns a float in approximately [-1, 1].  Positive means agent beat greedy.
    """
    from env.environment import BUDGET_EXHAUSTION_PENALTY

    agent_score = project_score(agent_final_state)
    greedy_score = project_score(greedy_final_state)

    if agent_budget_exhausted:
        agent_score = max(0.0, agent_score - BUDGET_EXHAUSTION_PENALTY)

    return agent_score - greedy_score


def cross_verification_rate(state: ProjectState) -> float:
    """
    Compute the cross-verification rate for the episode so far.

    Spec: "cross_verification_rate = query_observable_signals_calls /
           total_member_query_calls"

    Returns 0.0 if no member queries have been made (avoids division by zero).
    """
    if state.total_member_query_calls == 0:
        return 0.0
    return state.cross_verify_calls / state.total_member_query_calls
