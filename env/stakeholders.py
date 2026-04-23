"""
stakeholders.py — Client and executive reactive agent state machines.

Spec: "STAKEHOLDER STATE MACHINES (stakeholders.py)"

Client state machine:
    satisfaction starts at 7.0
    -0.5 per step if no communication in last 5 steps
    -1.5 if bad news delivered without a solution
    +1.0 for proactive escalation with plan
    If satisfaction < 4 → triggers exec_escalation event

Executive state machine:
    support starts at 8.0
    -1.0 per exec_escalation event
    -0.5 if budget request made without updated timeline
    +0.5 for proactive risk communication
    If support < 5 → budget requests silently fail

Both satisfaction and support are in the step() observation (partially —
the agent sees the numbers but not the exact threshold/state-machine logic).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from env.state import (
    ProjectState,
    StakeholderState,
    CLIENT_ESCALATION_THRESHOLD,
    EXEC_SUPPORT_BUDGET_THRESHOLD,
    CLIENT_COMMUNICATION_WINDOW,
    CLIENT_DECAY_NO_COMM,
    CLIENT_DECAY_BAD_NEWS,
    CLIENT_GAIN_PROACTIVE,
    EXEC_DECAY_ESCALATION,
    EXEC_DECAY_BUDGET_NO_TIMELINE,
    EXEC_GAIN_RISK_COMM,
    DRIFT_ACK_WINDOW,
)

# ---------------------------------------------------------------------------
# Stakeholder satisfaction/support min/max bounds
# ---------------------------------------------------------------------------
SATISFACTION_MIN = 0.0
SATISFACTION_MAX = 10.0
SUPPORT_MIN = 0.0
SUPPORT_MAX = 10.0


def step_client_state_machine(state: ProjectState) -> bool:
    """
    Advance the client state machine by one step.

    Called at the END of each environment step, after actions have been
    processed.  Applies communication decay and checks the exec-escalation
    trigger.

    Returns True if an exec_escalation event was fired this step.
    """
    stakeholder = state.stakeholder
    current_step = state.current_step

    # Decay: no communication in last CLIENT_COMMUNICATION_WINDOW steps
    steps_since_comm = current_step - stakeholder.client_last_communicated_step
    if steps_since_comm >= CLIENT_COMMUNICATION_WINDOW:
        stakeholder.client_satisfaction = max(
            SATISFACTION_MIN,
            stakeholder.client_satisfaction - CLIENT_DECAY_NO_COMM,
        )

    # Clamp
    stakeholder.client_satisfaction = max(
        SATISFACTION_MIN,
        min(SATISFACTION_MAX, stakeholder.client_satisfaction),
    )

    # Exec escalation trigger
    if stakeholder.client_satisfaction < CLIENT_ESCALATION_THRESHOLD:
        _fire_exec_escalation(state)
        return True

    return False


def apply_bad_news_penalty(state: ProjectState, has_solution: bool) -> None:
    """
    Apply the client satisfaction penalty for bad news delivery.

    Spec: "Decreases 1.5 if bad news delivered without a solution"

    Called by action handlers (e.g., communicate with message_type="bad_news").
    If has_solution is True, no penalty is applied (agent did the right thing).
    """
    if not has_solution:
        state.stakeholder.client_satisfaction = max(
            SATISFACTION_MIN,
            state.stakeholder.client_satisfaction - CLIENT_DECAY_BAD_NEWS,
        )


def step_exec_state_machine(state: ProjectState) -> None:
    """
    Advance the executive state machine by one step.

    Currently passive — the exec reacts to events (exec_escalation, budget
    requests) rather than decaying per-step.  This function exists to allow
    future per-step exec logic without interface changes.
    """
    # Clamp support
    state.stakeholder.exec_support = max(
        SUPPORT_MIN,
        min(SUPPORT_MAX, state.stakeholder.exec_support),
    )


def _fire_exec_escalation(state: ProjectState) -> None:
    """
    Fire an exec_escalation event (client dissatisfaction reached threshold).

    Spec: "Decreases 1.0 per exec_escalation event"

    The escalation count is tracked for metrics.  The support decrease models
    exec frustration at being surprised by a client problem.
    """
    state.stakeholder.exec_escalation_count += 1
    state.stakeholder.exec_support = max(
        SUPPORT_MIN,
        state.stakeholder.exec_support - EXEC_DECAY_ESCALATION,
    )


def apply_drift_satisfaction_penalty(state: ProjectState) -> None:
    """
    Apply stakeholder satisfaction penalty for unacknowledged drift events.

    Spec: "the agent must take at least one planning action (update_timeline
    or communicate) acknowledging the drift within 3 steps or a stakeholder
    satisfaction penalty applies"

    Called by the environment when a drift event's acknowledgement deadline
    has passed without the agent acting.
    """
    # Penalty: -1.0 to client satisfaction (bad news not addressed)
    state.stakeholder.client_satisfaction = max(
        SATISFACTION_MIN,
        state.stakeholder.client_satisfaction - 1.0,
    )
    # And a mild exec penalty for poor change management
    state.stakeholder.exec_support = max(
        SUPPORT_MIN,
        state.stakeholder.exec_support - 0.3,
    )


def check_drift_deadlines(state: ProjectState) -> None:
    """
    Check all active drift events for missed acknowledgement deadlines.

    If a drift event was fired and not acknowledged by its deadline, apply
    the stakeholder satisfaction penalty and mark it as acknowledged (to avoid
    repeated penalties).

    Called once per step by the environment.
    """
    for event in state.active_drift_events:
        if not event.acknowledged and state.current_step > event.acknowledgement_deadline:
            apply_drift_satisfaction_penalty(state)
            event.acknowledged = True  # no double-penalty


def get_stakeholder_observation(state: ProjectState) -> dict:
    """
    Build the partial stakeholder observation exposed to the agent.

    The agent sees satisfaction/support levels (numeric) but NOT:
    - The exact thresholds (CLIENT_ESCALATION_THRESHOLD, etc.)
    - The state machine internals
    - How many steps until the next decay tick
    """
    return {
        "client_satisfaction": round(state.stakeholder.client_satisfaction, 2),
        "exec_support": round(state.stakeholder.exec_support, 2),
        "exec_escalation_count": state.stakeholder.exec_escalation_count,
    }
