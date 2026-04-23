"""
reward/baseline.py — GreedyPMBaseline deterministic rule-based agent.

Spec: "GREEDY PM BASELINE (baseline.py)"

Rules:
    - Always trusts self-reports; NEVER calls query_observable_signals
    - Each step: picks the crisis with highest REPORTED severity
    - Picks the member with highest REPORTED availability for reassignment
    - Communicates once every 5 steps regardless of stakeholder state
    - No LLM, no randomness — fully deterministic given state

Calibration target: mean project_score 0.45–0.55 across 20 random Level 1 episodes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from env.state import (
    ProjectState,
    TeamMember,
    Crisis,
    Task,
)

# ---------------------------------------------------------------------------
# Greedy PM behaviour constants
# ---------------------------------------------------------------------------

# How often greedy PM sends a communication (steps between comms)
GREEDY_COMM_INTERVAL = 5

# Greedy PM always uses this message type (status update, no initiative)
GREEDY_COMM_MESSAGE_TYPE = "status_update"
GREEDY_COMM_CONTENT = "Standard status update from PM."


class GreedyPMBaseline:
    """
    Deterministic rule-based baseline PM agent.

    This agent always trusts what team members self-report.  It never checks
    observable signals, so deceptive members fool it consistently.  Its
    behaviour sets the counterfactual floor against which the RL agent is
    judged.

    The baseline is NOT an LLM and contains no randomness — given the same
    ProjectState it always produces the same action.
    """

    def __init__(self) -> None:
        """Initialise baseline with a step counter for communication timing."""
        self._steps_since_comm: int = 0
        self._step: int = 0
        # Track last reassignment step per task to prevent immediate ping-pong
        self._last_reassign_step: dict = {}  # task_id -> step
        self._reassign_cooldown: int = 3  # steps before re-reassignment allowed

    def act(self, state: ProjectState) -> Dict[str, Any]:
        """
        Choose the next action based purely on reported state.

        Priority order:
        1. Communicate if GREEDY_COMM_INTERVAL steps have elapsed
        2. If any crisis is unresolved: reassign to highest-availability member
        3. Otherwise: query_status (free action, effectively a no-op step)

        Spec: "It picks the crisis with highest reported severity each step"
              "It picks the member with highest reported availability for reassignment"
              "It communicates once every 5 steps regardless of stakeholder state"
        """
        self._steps_since_comm += 1
        self._step += 1

        # --- Communication every 5 steps ---
        if self._steps_since_comm >= GREEDY_COMM_INTERVAL:
            self._steps_since_comm = 0
            return self._communicate()

        # --- Pick highest-severity unresolved crisis ---
        unresolved = [c for c in state.crises if not c.is_resolved]
        if not unresolved:
            # No active crises → submit recovery plan
            return self._submit_plan()

        target_crisis = max(unresolved, key=lambda c: c.severity)

        # --- Find a task in that crisis that needs help ---
        blocking_task = self._find_blocking_task(target_crisis, state)
        if blocking_task is None:
            # All tasks in crisis are done or no tasks — query status
            return {"action_type": "query_status", "params": {}}

        # --- Reassign to highest reported availability member ---
        # Cooldown: don't re-reassign the same task within N steps (avoids worst-case ping-pong)
        last_step = self._last_reassign_step.get(blocking_task.task_id, -999)
        if self._step - last_step < self._reassign_cooldown:
            return {"action_type": "query_status", "params": {}}

        best_member = self._pick_best_member_reported(state, blocking_task.assigned_member_id)
        if best_member is None or best_member.member_id == blocking_task.assigned_member_id:
            # Can't improve assignment — communicate if due, else query
            return {"action_type": "query_status", "params": {}}

        self._last_reassign_step[blocking_task.task_id] = self._step
        return {
            "action_type": "reassign_task",
            "params": {
                "task_id": blocking_task.task_id,
                "to_member_id": best_member.member_id,
            },
        }

    def _communicate(self) -> Dict[str, Any]:
        """Return a communicate action with the standard greedy message."""
        return {
            "action_type": "communicate",
            "params": {
                "message_type": GREEDY_COMM_MESSAGE_TYPE,
                "content": GREEDY_COMM_CONTENT,
                "target": "both",
            },
        }

    def _submit_plan(self) -> Dict[str, Any]:
        """Submit recovery plan when no crises remain."""
        return {
            "action_type": "submit_recovery_plan",
            "params": {
                "plan_summary": "All crises resolved. Standard recovery complete.",
                "risk_items": [],
                "timeline": "",
            },
        }

    def _find_blocking_task(
        self, crisis: Crisis, state: ProjectState
    ) -> Optional[Task]:
        """
        Find the most critical in-progress or backlog task for the given crisis.

        Prefers critical-path tasks.  Returns None if all tasks are done.
        """
        eligible = []
        for tid in crisis.affected_task_ids:
            task = state.get_task(tid)
            if task and task.status != "done":
                eligible.append(task)

        if not eligible:
            return None

        # Prefer critical-path
        critical = [t for t in eligible if t.is_critical_path]
        if critical:
            return critical[0]
        return eligible[0]

    def _pick_best_member_reported(
        self, state: ProjectState, exclude_id: Optional[str]
    ) -> Optional[TeamMember]:
        """
        Pick the member with the highest REPORTED availability.

        Greedy PM trusts reports — it does NOT check observable signals.
        Excludes the currently-assigned member to avoid no-op reassignments.
        """
        candidates = [
            m for m in state.team_members
            if m.member_id != exclude_id
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.reported_availability)
