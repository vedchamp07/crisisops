"""
schema_drift.py — Mid-episode requirement change event system.

Spec: "SCHEMA DRIFT (schema_drift.py)"

At Level 2+, one drift event fires at a randomly chosen step between 6 and 12.
Three event types:
    regulatory_change:   adds a compliance requirement that blocks one feature.
    client_scope_change: client deprioritises one feature, adds a new one.
    team_policy_change:  new mandatory review process; all PRs need second approver.

After a drift event fires, the NEXT step() observation includes a drift_event
field.  The agent has DRIFT_ACK_WINDOW steps to acknowledge via update_timeline
or communicate, else a stakeholder satisfaction penalty applies.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from env.state import (
    ProjectState,
    Task,
    DriftEvent,
    DRIFT_ACK_WINDOW,
    DRIFT_STEP_MIN,
    DRIFT_STEP_MAX,
)

# ---------------------------------------------------------------------------
# Drift event type constants
# ---------------------------------------------------------------------------
DRIFT_REGULATORY_CHANGE = "regulatory_change"
DRIFT_CLIENT_SCOPE_CHANGE = "client_scope_change"
DRIFT_TEAM_POLICY_CHANGE = "team_policy_change"

DRIFT_EVENT_TYPES = [
    DRIFT_REGULATORY_CHANGE,
    DRIFT_CLIENT_SCOPE_CHANGE,
    DRIFT_TEAM_POLICY_CHANGE,
]

# How much extra days are added to all tasks for team_policy_change
TEAM_POLICY_REVIEW_OVERHEAD_DAYS = 1.5

# How many tasks a client_scope_change deprioritises (1) and adds (1)
CLIENT_SCOPE_DEPRIORITISE_COUNT = 1

# Compliance block description template
REGULATORY_DESCRIPTION = (
    "Regulatory change: new data-retention compliance requirement blocks "
    "feature '{task_title}' from shipping until audit is complete."
)
CLIENT_SCOPE_DESCRIPTION = (
    "Client scope change: '{deprioritised}' deprioritised; new feature "
    "'{added}' added to critical path."
)
TEAM_POLICY_DESCRIPTION = (
    "Team policy change: all PRs now require a second approver. "
    f"+{TEAM_POLICY_REVIEW_OVERHEAD_DAYS:.1f} days added to all task estimates."
)


def choose_drift_step(rng: random.Random) -> int:
    """
    Sample the episode step at which the drift event will fire.

    Spec: "a drift event fires at a randomly chosen step between 6 and 12"
    """
    return rng.randint(DRIFT_STEP_MIN, DRIFT_STEP_MAX)


def choose_drift_type(rng: random.Random) -> str:
    """Pick one drift event type uniformly at random."""
    return rng.choice(DRIFT_EVENT_TYPES)


def build_regulatory_change(
    state: ProjectState, rng: random.Random
) -> Optional[DriftEvent]:
    """
    Build a regulatory_change drift event.

    Selects a non-blocked, non-resolved task (preferably on the critical path)
    and marks it compliance-blocked.  Returns None if no eligible task exists.
    """
    eligible = [
        t for t in state.tasks
        if not t.is_deprioritized and not t.is_compliance_blocked and t.status != "done"
    ]
    if not eligible:
        return None

    # Prefer critical-path tasks as they create the most disruption
    critical = [t for t in eligible if t.is_critical_path]
    target = rng.choice(critical) if critical else rng.choice(eligible)

    target.is_compliance_blocked = True
    if target.status == "in_progress":
        target.status = "blocked"

    description = REGULATORY_DESCRIPTION.format(task_title=target.title)
    event = DriftEvent(
        event_type=DRIFT_REGULATORY_CHANGE,
        step_fired=state.current_step,
        description=description,
        affected_task_ids=[target.task_id],
        acknowledged=False,
        acknowledgement_deadline=state.current_step + DRIFT_ACK_WINDOW,
    )
    return event


def build_client_scope_change(
    state: ProjectState, rng: random.Random
) -> Optional[DriftEvent]:
    """
    Build a client_scope_change drift event.

    Deprioritises one existing task and adds a new placeholder task to the
    critical path.  Models a real-world client pivot mid-sprint.
    """
    deprioritisable = [
        t for t in state.tasks
        if not t.is_deprioritized and not t.is_critical_path and t.status != "done"
    ]
    if not deprioritisable:
        # Fall back to any non-done task
        deprioritisable = [t for t in state.tasks if t.status != "done"]
    if not deprioritisable:
        return None

    to_deprioritise = rng.choice(deprioritisable)
    to_deprioritise.is_deprioritized = True
    to_deprioritise.status = "backlog"

    # Add a new critical-path task representing the client's new requirement
    new_task_id = f"drift_task_{state.current_step}"
    new_task = Task(
        task_id=new_task_id,
        title="New client requirement (scope change)",
        crisis_id=None,
        assigned_member_id=None,
        status="backlog",
        is_critical_path=True,
        estimated_days=rng.uniform(2.0, 5.0),
        actual_progress=0.0,
    )
    state.tasks.append(new_task)

    description = CLIENT_SCOPE_DESCRIPTION.format(
        deprioritised=to_deprioritise.title,
        added=new_task.title,
    )
    event = DriftEvent(
        event_type=DRIFT_CLIENT_SCOPE_CHANGE,
        step_fired=state.current_step,
        description=description,
        affected_task_ids=[to_deprioritise.task_id, new_task_id],
        acknowledged=False,
        acknowledgement_deadline=state.current_step + DRIFT_ACK_WINDOW,
    )
    return event


def build_team_policy_change(
    state: ProjectState, rng: random.Random
) -> Optional[DriftEvent]:
    """
    Build a team_policy_change drift event.

    Adds review overhead to ALL non-done tasks, modelling the cost of a new
    mandatory second-approver process.
    """
    affected_ids = []
    for task in state.tasks:
        if task.status != "done":
            task.review_overhead_days += TEAM_POLICY_REVIEW_OVERHEAD_DAYS
            affected_ids.append(task.task_id)

    event = DriftEvent(
        event_type=DRIFT_TEAM_POLICY_CHANGE,
        step_fired=state.current_step,
        description=TEAM_POLICY_DESCRIPTION,
        affected_task_ids=affected_ids,
        acknowledged=False,
        acknowledgement_deadline=state.current_step + DRIFT_ACK_WINDOW,
    )
    return event


def fire_drift_event(state: ProjectState, rng: random.Random) -> Optional[DriftEvent]:
    """
    Choose an event type and build the corresponding DriftEvent.

    Mutates state (tasks may be blocked/deprioritised/overhead-added).
    The caller is responsible for storing the returned event on
    state.pending_drift_event and state.active_drift_events.

    Returns None if no eligible event could be constructed (edge case in
    degenerate states where all tasks are done).
    """
    drift_type = choose_drift_type(rng)
    builders = {
        DRIFT_REGULATORY_CHANGE:  build_regulatory_change,
        DRIFT_CLIENT_SCOPE_CHANGE: build_client_scope_change,
        DRIFT_TEAM_POLICY_CHANGE:  build_team_policy_change,
    }
    event = builders[drift_type](state, rng)
    if event is not None:
        state.active_drift_events.append(event)
        state.pending_drift_event = event
    return event


def get_pending_drift_observation(state: ProjectState) -> Optional[dict]:
    """
    If a drift event was fired but not yet delivered to the agent, return its
    observation dict and clear the pending pointer.

    Called by environment.step() to attach drift info to the next observation.
    """
    event = state.pending_drift_event
    if event is None:
        return None

    state.pending_drift_event = None  # consumed; now in active_drift_events

    return {
        "drift_event": {
            "event_type": event.event_type,
            "description": event.description,
            "affected_task_ids": event.affected_task_ids,
            "steps_to_acknowledge": DRIFT_ACK_WINDOW,
        }
    }
