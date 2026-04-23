"""
baselines/random_agent.py — Random agent for sanity checking reward range.

Spec: "Random agent for sanity checking reward range"

Randomly samples a valid action each step.  Used to verify that the reward
distribution has the expected range and that random behaviour scores below
the greedy baseline.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from env.state import ProjectState
from env.actions import ACTION_COSTS, VALID_ACTION_TYPES

# Actions that require specific params we can auto-fill
_PARAM_TEMPLATES: Dict[str, callable] = {}


class RandomAgent:
    """
    Random agent that samples valid actions uniformly.

    Handles parameter generation for each action type so every sampled
    action is syntactically valid (though likely suboptimal).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Args:
            seed: Random seed for reproducibility.
        """
        self._rng = random.Random(seed)

    def act(self, state: ProjectState) -> Dict[str, Any]:
        """
        Sample a random valid action given the current state.

        Avoids actions that would immediately exhaust the budget.
        """
        available = [
            a for a in VALID_ACTION_TYPES
            if ACTION_COSTS[a] <= state.budget_remaining or ACTION_COSTS[a] == 0
        ]

        # Don't submit recovery plan until we've used at least 5 budget points
        budget_spent = 20 - state.budget_remaining  # INITIAL_BUDGET - remaining
        if budget_spent < 5 and "submit_recovery_plan" in available:
            available = [a for a in available if a != "submit_recovery_plan"]

        if not available:
            # Last resort
            return {"action_type": "submit_recovery_plan",
                    "params": {"plan_summary": "Random agent forced to submit."}}

        action_type = self._rng.choice(available)
        params = self._generate_params(action_type, state)
        return {"action_type": action_type, "params": params}

    def _generate_params(
        self, action_type: str, state: ProjectState
    ) -> Dict[str, Any]:
        """Generate syntactically valid params for the chosen action type."""
        if action_type in ("query_status", "consult_expert"):
            return {}

        if action_type in ("query_member_report", "query_observable_signals"):
            member = self._pick_member(state)
            return {"member_id": member} if member else {}

        if action_type == "query_ticket":
            task = self._pick_task(state)
            return {"task_id": task} if task else {}

        if action_type == "reassign_task":
            task = self._pick_task(state, exclude_done=True)
            member = self._pick_member(state)
            return {"task_id": task or "t1", "to_member_id": member or "dev_1"}

        if action_type == "communicate":
            msg_type = self._rng.choice([
                "status_update", "proactive_escalation_with_plan", "risk_communication"
            ])
            return {
                "message_type": msg_type,
                "content": "Random agent update.",
                "target": "both",
            }

        if action_type == "cut_scope":
            task = self._pick_task(state, non_critical=True)
            return {"task_id": task or "t1", "justification": "Random scope cut."}

        if action_type == "escalate_risk":
            crisis = self._pick_crisis(state)
            return {"crisis_id": crisis or "c1", "risk_description": "Random escalation."}

        if action_type == "request_resource":
            member = self._pick_member(state)
            return {"resource_type": "headcount", "target_member_id": member or "dev_1"}

        if action_type == "update_timeline":
            return {"new_completion_date": "2024-12-31", "task_estimates": {}}

        if action_type == "resolve_blocker":
            task = self._pick_task(state, blocked_only=True)
            return {
                "task_id": task or self._pick_task(state) or "t1",
                "resolution_notes": "Random blocker resolution.",
            }

        if action_type == "submit_recovery_plan":
            return {
                "plan_summary": "Random agent recovery plan.",
                "risk_items": [],
                "timeline": "2024-12-31",
            }

        return {}

    def _pick_member(self, state: ProjectState) -> Optional[str]:
        """Pick a random member id."""
        if not state.team_members:
            return None
        return self._rng.choice(state.team_members).member_id

    def _pick_task(
        self,
        state: ProjectState,
        exclude_done: bool = False,
        non_critical: bool = False,
        blocked_only: bool = False,
    ) -> Optional[str]:
        """Pick a random task id matching the given filters."""
        candidates = state.tasks
        if exclude_done:
            candidates = [t for t in candidates if t.status != "done"]
        if non_critical:
            candidates = [t for t in candidates if not t.is_critical_path]
        if blocked_only:
            candidates = [t for t in candidates if t.status == "blocked"]
        if not candidates:
            return None
        return self._rng.choice(candidates).task_id

    def _pick_crisis(self, state: ProjectState) -> Optional[str]:
        """Pick a random unresolved crisis id."""
        active = [c for c in state.crises if not c.is_resolved]
        if not active:
            return None
        return self._rng.choice(active).crisis_id
