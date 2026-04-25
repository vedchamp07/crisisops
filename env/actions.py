"""
actions.py — All 12 CrisisOps actions with cost tiers, side effects, and state transitions.

Spec: "ACTION SYSTEM (actions.py)"

Cost tiers:
    Free   (cost 0): query_status, query_member_report, query_observable_signals, query_ticket
    Cost-1 (cost 1): reassign_task, communicate, cut_scope, escalate_risk,
                     request_resource, update_timeline, consult_expert
    Cost-2 (cost 2): resolve_blocker
    Terminal        : submit_recovery_plan  (ends episode immediately)

Budget starts at 20.  Free actions never decrement it.  If budget reaches 0
before submit_recovery_plan is called, the episode ends with a budget-exhaustion
penalty (greedy PM gets its normal score; reward goes negative).

Input format (JSON dict):
    {"action_type": "query_member_report", "params": {"member_id": "dev_2"}}
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from env.state import (
    ProjectState,
    TeamMember,
    Task,
    Crisis,
    CANDOR_LEVEL_SELF_PRESERVATION,
    EXEC_SUPPORT_BUDGET_THRESHOLD,
    CLIENT_COMMUNICATION_WINDOW,
    CLIENT_GAIN_PROACTIVE,
    EXEC_DECAY_BUDGET_NO_TIMELINE,
    EXEC_GAIN_RISK_COMM,
    DRIFT_ACK_WINDOW,
)
from env.candor import get_observable_signals, update_ticket_change_step, apply_caught_effect
from env.stakeholders import apply_bad_news_penalty

# ---------------------------------------------------------------------------
# Cost tier constants — spec "ACTION SYSTEM"
# ---------------------------------------------------------------------------
ACTION_COST_FREE = 0
ACTION_COST_STANDARD = 1
ACTION_COST_HEAVY = 2

# Registry: action_type -> cost
ACTION_COSTS: Dict[str, int] = {
    # Free
    "query_status":              ACTION_COST_FREE,
    "query_member_report":       ACTION_COST_FREE,
    "query_observable_signals":  ACTION_COST_FREE,
    "query_ticket":              ACTION_COST_FREE,
    # Cost-1
    "reassign_task":             ACTION_COST_STANDARD,
    "communicate":               ACTION_COST_STANDARD,
    "cut_scope":                 ACTION_COST_STANDARD,
    "escalate_risk":             ACTION_COST_STANDARD,
    "request_resource":          ACTION_COST_STANDARD,
    "update_timeline":           ACTION_COST_STANDARD,
    "consult_expert":            ACTION_COST_STANDARD,
    "query_peer_opinion":       ACTION_COST_STANDARD,
    "force_truth":              ACTION_COST_STANDARD,
    "trigger_whistleblower":    ACTION_COST_STANDARD,
    # Cost-2
    "resolve_blocker":           ACTION_COST_HEAVY,
    # Terminal (cost-1 for budget but ends episode)
    "submit_recovery_plan":      ACTION_COST_STANDARD,
}

# All valid action types
VALID_ACTION_TYPES = set(ACTION_COSTS.keys())

# Expert advisor confidence threshold — crises with severity above this get
# a "critical" flag in the expert report
EXPERT_CRITICAL_SEVERITY = 7.0

# Morale boost when a task is reassigned to a more available member
REASSIGN_MORALE_BOOST = 0.3

# Morale decrease when scope is cut (team feels setback)
CUT_SCOPE_MORALE_PENALTY = 0.5

# Progress boost applied to a blocked task when resolve_blocker succeeds
RESOLVE_BLOCKER_PROGRESS_BOOST = 0.25

# Political capital costs for power actions
POLITICAL_CAPITAL_FORCE_TRUTH = 3.0      # PC cost to reveal a member's actual_completion
POLITICAL_CAPITAL_WHISTLEBLOWER = 6.0    # PC cost to reveal identity of worst liar

# Political capital earned from good PM behaviors
PC_EARN_PROACTIVE = 2.0                  # earned from proactive_escalation_with_plan
PC_EARN_TIMELINE = 1.0                   # earned from update_timeline before drift deadline

# Catch window: punitive action must happen within this many steps of cross-verify
CATCH_WINDOW_STEPS = 3

# Threshold for actual_completion below which a crisis is considered "unresolved"
CRISIS_RESOLUTION_COMPLETION_THRESHOLD = 0.90


@dataclass
class ActionResult:
    """
    Structured result returned by every action handler.

    ``observation`` is merged into the step() observation dict.
    ``error`` is set for invalid inputs (budget NOT decremented on error).
    ``done`` signals episode termination (submit_recovery_plan or budget=0).
    """

    observation: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    done: bool = False
    budget_decremented: bool = False


# ---------------------------------------------------------------------------
# Input validation helper
# ---------------------------------------------------------------------------

def validate_action(action: Dict[str, Any]) -> Optional[str]:
    """
    Validate the top-level action dict format.

    Returns an error string if invalid, else None.
    Budget is NOT decremented for invalid actions (spec: "invalid action_type
    or missing params return an error observation without decrementing budget").
    """
    if not isinstance(action, dict):
        return "Action must be a dict"
    if "action_type" not in action:
        return "Missing required field: action_type"
    if action["action_type"] not in VALID_ACTION_TYPES:
        return f"Unknown action_type: {action['action_type']!r}"
    if "params" not in action:
        return "Missing required field: params"
    if not isinstance(action["params"], dict):
        return "params must be a dict"
    return None


# ---------------------------------------------------------------------------
# FREE ACTIONS
# ---------------------------------------------------------------------------

def action_query_status(state: ProjectState, params: Dict[str, Any]) -> ActionResult:
    """
    Return a high-level summary of all active crises and overall project health.

    Free action — never decrements budget.  The observation includes only
    REPORTED figures and structural info (no raw candor, no actual_completion).
    """
    crises_obs = []
    for c in state.crises:
        crises_obs.append({
            "crisis_id": c.crisis_id,
            "crisis_type": c.crisis_type,
            "severity": c.severity,
            "description": c.description,
            "is_resolved": c.is_resolved,
            "affected_task_ids": c.affected_task_ids,
        })

    members_summary = []
    for m in state.team_members:
        members_summary.append({
            "member_id": m.member_id,
            "name": m.name,
            "role": m.role,
            "reported_completion": round(m.reported_completion, 3),
            "reported_availability": round(m.reported_availability, 3),
            "assigned_task_ids": m.assigned_task_ids,
        })

    return ActionResult(observation={
        "action_type": "query_status",
        "active_crises": [c for c in crises_obs if not c["is_resolved"]],
        "resolved_crises": [c for c in crises_obs if c["is_resolved"]],
        "team_summary": members_summary,
        "budget_remaining": state.budget_remaining,
        "current_step": state.current_step,
        "client_satisfaction": round(state.stakeholder.client_satisfaction, 2),
        "exec_support": round(state.stakeholder.exec_support, 2),
    })


def action_query_member_report(state: ProjectState, params: Dict[str, Any]) -> ActionResult:
    """
    Fetch the self-reported status from a specific team member.

    Free action.  Returns reported (potentially inflated) completion and
    availability.  Does NOT include actual figures or candor.

    Required params: member_id (str)
    """
    member_id = params.get("member_id")
    if not member_id:
        return ActionResult(error="Missing param: member_id")

    member = state.get_member(member_id)
    if member is None:
        return ActionResult(error=f"Unknown member_id: {member_id!r}")

    # Count this as a member query for cross_verify_rate denominator
    state.total_member_query_calls += 1

    observation = {
        "action_type": "query_member_report",
        "member_id": member.member_id,
        "name": member.name,
        "role": member.role,
        "reported_completion": round(member.reported_completion, 3),
        "reported_availability": round(member.reported_availability, 3),
        "assigned_task_ids": member.assigned_task_ids,
    }

    # Alibi injection: allied self-preservation members blame their stall on their ally.
    # The alibi is CONSISTENT (always blames the same task) so the agent can debunk it
    # by running query_observable_signals on the blamed ally.
    if (member.candor_level == CANDOR_LEVEL_SELF_PRESERVATION
            and member.alliance_id is not None):
        ally = _find_ally(member, state)
        if ally and ally.assigned_task_ids:
            # Always blame the lexicographically first task for consistency across calls
            blamed_task_id = min(ally.assigned_task_ids)
            blamed_task = state.get_task(blamed_task_id)
            if blamed_task:
                observation["alibi"] = {
                    "type": "dependency_block",
                    "claim": (
                        f"I'm blocked — waiting on '{blamed_task.title}' "
                        f"from {ally.name} to land before I can proceed."
                    ),
                    "blames_member_id": ally.member_id,
                    "blames_task_id": blamed_task_id,
                }

    return ActionResult(observation=observation)


def action_query_observable_signals(
    state: ProjectState, params: Dict[str, Any]
) -> ActionResult:
    """
    Fetch the objective observable signals for a team member.

    Free action.  Signals are computed ONLY from actual state — ticket age,
    commit activity, peer mention count — so the agent can cross-reference
    them against the member's self-report to detect deception.

    Required params: member_id (str)
    """
    member_id = params.get("member_id")
    if not member_id:
        return ActionResult(error="Missing param: member_id")

    member = state.get_member(member_id)
    if member is None:
        return ActionResult(error=f"Unknown member_id: {member_id!r}")

    signals = get_observable_signals(member, state)

    # Track both metrics for cross_verify_rate
    state.cross_verify_calls += 1
    state.total_member_query_calls += 1

    # Track cross-verification for dynamic candor and catch detection
    member.times_cross_verified += 1
    member.last_cross_verified_step = state.current_step

    return ActionResult(observation={
        "action_type": "query_observable_signals",
        "member_id": member.member_id,
        "signals": signals,
    })


def action_query_ticket(state: ProjectState, params: Dict[str, Any]) -> ActionResult:
    """
    Fetch the current status and metadata for a specific task/ticket.

    Free action.  Returns task status, assignment, critical-path flag, and
    whether it is compliance-blocked or deprioritized (from drift events).

    Required params: task_id (str)
    """
    task_id = params.get("task_id")
    if not task_id:
        return ActionResult(error="Missing param: task_id")

    task = state.get_task(task_id)
    if task is None:
        return ActionResult(error=f"Unknown task_id: {task_id!r}")

    return ActionResult(observation={
        "action_type": "query_ticket",
        "task_id": task.task_id,
        "title": task.title,
        "status": task.status,
        "assigned_member_id": task.assigned_member_id,
        "is_critical_path": task.is_critical_path,
        "estimated_days": task.estimated_days + task.review_overhead_days,
        "is_deprioritized": task.is_deprioritized,
        "is_compliance_blocked": task.is_compliance_blocked,
        "crisis_id": task.crisis_id,
    })


# ---------------------------------------------------------------------------
# COST-1 ACTIONS
# ---------------------------------------------------------------------------

def action_reassign_task(state: ProjectState, params: Dict[str, Any]) -> ActionResult:
    """
    Reassign a task from its current owner to a different team member.

    Side effects:
    - Updates task.assigned_member_id and both members' assigned_task_ids lists
    - Updates ticket_last_changed_step (resets ticket age signal)
    - Small morale boost for the receiving member (motivation from being trusted)
    - If new member has higher actual_availability, actual_velocity improves

    Required params: task_id (str), to_member_id (str)
    """
    task_id = params.get("task_id")
    to_member_id = params.get("to_member_id")
    if not task_id:
        return ActionResult(error="Missing param: task_id")
    if not to_member_id:
        return ActionResult(error="Missing param: to_member_id")

    task = state.get_task(task_id)
    if task is None:
        return ActionResult(error=f"Unknown task_id: {task_id!r}")

    new_member = state.get_member(to_member_id)
    if new_member is None:
        return ActionResult(error=f"Unknown to_member_id: {to_member_id!r}")

    old_member_id = task.assigned_member_id

    # Remove from old member's list
    if old_member_id:
        old_member = state.get_member(old_member_id)
        if old_member and task_id in old_member.assigned_task_ids:
            old_member.assigned_task_ids.remove(task_id)

    # Add to new member's list
    task.assigned_member_id = to_member_id
    if task_id not in new_member.assigned_task_ids:
        new_member.assigned_task_ids.append(task_id)

    # Update ticket change step (resets ticket_age_days signal)
    update_ticket_change_step(new_member, state.current_step)

    # Morale boost for new member
    new_member.morale = min(10.0, new_member.morale + REASSIGN_MORALE_BOOST)

    # If task was blocked, reassignment moves it back to in_progress
    if task.status == "blocked":
        task.status = "in_progress"

    # Catch detection: if PM is reassigning FROM a recently cross-verified self-preservation member,
    # that counts as catching the liar. Apply dynamic candor effect.
    if old_member_id:
        old_member = state.get_member(old_member_id)
        if (old_member is not None
                and old_member.candor_level == CANDOR_LEVEL_SELF_PRESERVATION
                and (state.current_step - old_member.last_cross_verified_step) <= CATCH_WINDOW_STEPS
                and not old_member.caught_this_episode):
            apply_caught_effect(old_member, state)
            # Merge catch info into the observation dict we're about to return
            catch_note = {
                "deception_catch": {
                    "member_id": old_member_id,
                    "member_name": old_member.name,
                    "effect": "candor_improved_inflation_reduced",
                    "political_capital_awarded": 3.0,
                }
            }
        else:
            catch_note = {}
    else:
        catch_note = {}

    result_obs = {
        "action_type": "reassign_task",
        "task_id": task_id,
        "from_member_id": old_member_id,
        "to_member_id": to_member_id,
        "task_status": task.status,
    }
    result_obs.update(catch_note)
    return ActionResult(observation=result_obs)


def action_communicate(state: ProjectState, params: Dict[str, Any]) -> ActionResult:
    """
    Send a communication update to stakeholders (client and/or exec).

    Side effects:
    - Resets client_last_communicated_step (prevents communication decay penalty)
    - If message_type == "proactive_escalation_with_plan": client_satisfaction += 1.0
    - Acknowledges any pending drift event (if not already done within deadline)

    Required params: message_type (str), content (str)
    Optional params: target ("client" | "exec" | "both", default "both")
    """
    message_type = params.get("message_type", "status_update")
    content = params.get("content", "")
    target = params.get("target", "both")

    state.stakeholder.client_last_communicated_step = state.current_step

    gain = 0.0
    if message_type == "proactive_escalation_with_plan":
        gain = CLIENT_GAIN_PROACTIVE
        state.stakeholder.client_satisfaction = min(
            10.0, state.stakeholder.client_satisfaction + gain
        )
        # Award political capital for proactive communication
        state.political_capital = min(20.0, state.political_capital + PC_EARN_PROACTIVE)
    elif message_type == "bad_news":
        # Spec: -1.5 if bad news is delivered without a solution.
        has_solution = bool(params.get("has_solution", False))
        apply_bad_news_penalty(state, has_solution)
    elif message_type == "risk_communication":
        state.stakeholder.exec_support = min(
            10.0, state.stakeholder.exec_support + EXEC_GAIN_RISK_COMM
        )

    # Drift acknowledgement: communicate counts as an acknowledging action
    _acknowledge_pending_drift(state)

    return ActionResult(observation={
        "action_type": "communicate",
        "message_type": message_type,
        "target": target,
        "client_satisfaction_after": round(state.stakeholder.client_satisfaction, 2),
        "exec_support_after": round(state.stakeholder.exec_support, 2),
        "satisfaction_delta": gain,
        "political_capital": state.political_capital,
    })


def action_cut_scope(state: ProjectState, params: Dict[str, Any]) -> ActionResult:
    """
    Deprioritise a non-critical task to reduce workload and free up capacity.

    Side effects:
    - Sets task.is_deprioritized = True and task.status = "backlog"
    - Small team morale penalty (team feels work was wasted)
    - Resolves crisis if ALL its affected tasks are now deprioritized or done

    Required params: task_id (str), justification (str)
    """
    task_id = params.get("task_id")
    if not task_id:
        return ActionResult(error="Missing param: task_id")

    task = state.get_task(task_id)
    if task is None:
        return ActionResult(error=f"Unknown task_id: {task_id!r}")

    if task.is_critical_path:
        return ActionResult(error="Cannot cut scope on a critical-path task")

    task.is_deprioritized = True
    task.status = "backlog"

    # Morale penalty for all members assigned to this task
    if task.assigned_member_id:
        member = state.get_member(task.assigned_member_id)
        if member:
            member.morale = max(0.0, member.morale - CUT_SCOPE_MORALE_PENALTY)

    return ActionResult(observation={
        "action_type": "cut_scope",
        "task_id": task_id,
        "task_deprioritized": True,
    })


def action_escalate_risk(state: ProjectState, params: Dict[str, Any]) -> ActionResult:
    """
    Formally escalate a risk to exec leadership.

    Side effects:
    - exec_support += EXEC_GAIN_RISK_COMM (proactive communication rewarded)
    - If crisis severity >= EXPERT_CRITICAL_SEVERITY, also triggers
      a simulated exec_escalation event (no support decrease — agent initiated it)
    - Acknowledges drift if pending

    Required params: crisis_id (str), risk_description (str)
    """
    crisis_id = params.get("crisis_id")
    if not crisis_id:
        return ActionResult(error="Missing param: crisis_id")

    crisis = state.get_crisis(crisis_id)
    if crisis is None:
        return ActionResult(error=f"Unknown crisis_id: {crisis_id!r}")

    state.stakeholder.exec_support = min(
        10.0, state.stakeholder.exec_support + EXEC_GAIN_RISK_COMM
    )

    # Catch detection: if escalating a crisis that contains tasks assigned to a recently
    # cross-verified self-preservation member, that counts as catching the liar.
    crisis_task_ids = set(crisis.affected_task_ids)
    for m in state.team_members:
        if (m.candor_level == CANDOR_LEVEL_SELF_PRESERVATION
                and (state.current_step - m.last_cross_verified_step) <= CATCH_WINDOW_STEPS
                and not m.caught_this_episode
                and any(tid in crisis_task_ids for tid in m.assigned_task_ids)):
            apply_caught_effect(m, state)
            break  # one catch per escalation action

    _acknowledge_pending_drift(state)

    return ActionResult(observation={
        "action_type": "escalate_risk",
        "crisis_id": crisis_id,
        "exec_support_after": round(state.stakeholder.exec_support, 2),
        "severity": crisis.severity,
    })


def action_request_resource(
    state: ProjectState, params: Dict[str, Any]
) -> ActionResult:
    """
    Request additional resources (budget, headcount, tooling) from exec.

    Spec: "silently fails and returns a failure observation if exec_support < 5"

    Side effects on success:
    - Notional resource is granted (modelled as availability boost for a member)
    - If agent did NOT call update_timeline recently: exec_support -= 0.5

    Required params: resource_type (str), target_member_id (str)
    """
    resource_type = params.get("resource_type", "budget")
    target_member_id = params.get("target_member_id")

    # Silent fail per spec if exec support is too low
    if state.stakeholder.exec_support < EXEC_SUPPORT_BUDGET_THRESHOLD:
        return ActionResult(observation={
            "action_type": "request_resource",
            "success": False,
            "reason": "exec_support_too_low",
        })

    # Penalty if no updated timeline was provided (exec expects context)
    if not state.stakeholder.last_budget_request_had_timeline:
        state.stakeholder.exec_support = max(
            0.0,
            state.stakeholder.exec_support - EXEC_DECAY_BUDGET_NO_TIMELINE,
        )

    # Grant resource: boost availability for target member if specified
    if target_member_id:
        member = state.get_member(target_member_id)
        if member:
            member.actual_availability = min(1.0, member.actual_availability + 0.2)
            member.actual_velocity = min(1.0, member.actual_velocity + 0.1)

    # Reset flag — next request needs a fresh update_timeline call
    state.stakeholder.last_budget_request_had_timeline = False

    return ActionResult(observation={
        "action_type": "request_resource",
        "success": True,
        "resource_type": resource_type,
        "target_member_id": target_member_id,
        "exec_support_after": round(state.stakeholder.exec_support, 2),
    })


def action_update_timeline(state: ProjectState, params: Dict[str, Any]) -> ActionResult:
    """
    Submit an updated project timeline to stakeholders.

    Side effects:
    - Sets timeline_updated_after_drift = True (used by request_resource)
    - Sets last_budget_request_had_timeline = True (exec sees the plan)
    - Acknowledges pending drift event
    - Adjusts task estimated_days based on new_estimate if provided

    Required params: new_completion_date (str — ISO date string)
    Optional params: task_estimates (dict[task_id, float])
    """
    new_completion_date = params.get("new_completion_date", "")
    task_estimates: Dict[str, float] = params.get("task_estimates", {})

    state.stakeholder.timeline_updated_after_drift = True
    state.stakeholder.last_budget_request_had_timeline = True
    # Award political capital for updating timeline (demonstrates planning discipline)
    state.political_capital = min(20.0, state.political_capital + PC_EARN_TIMELINE)

    # Apply new estimates
    for task_id, days in task_estimates.items():
        task = state.get_task(task_id)
        if task is not None:
            task.estimated_days = float(days)

    # Drift acknowledgement
    _acknowledge_pending_drift(state)

    return ActionResult(observation={
        "action_type": "update_timeline",
        "new_completion_date": new_completion_date,
        "tasks_updated": list(task_estimates.keys()),
        "political_capital": state.political_capital,
    })


def action_consult_expert(state: ProjectState, params: Dict[str, Any]) -> ActionResult:
    """
    Consult the rule-based senior PM advisor, who has access to TRUE state.

    Spec: "The advisor always has access to true state (it is the 'senior PM'
    who knows everything). This is the Snorkel AI bonus mechanism."

    The advisor is deterministic and rule-based — NOT an LLM.  It returns
    structured guidance based on the actual state of crises, tasks, and team.

    No params required; advisor inspects full state internally.
    """
    advice = _expert_advisor(state)

    return ActionResult(observation={
        "action_type": "consult_expert",
        "advice": advice,
    })


def action_query_peer_opinion(state: ProjectState, params: Dict[str, Any]) -> ActionResult:
    """
    Ask one team member (asker) for their assessment of another member's progress (subject).

    Cost-1 action. The quality of the opinion depends on the asker's hidden candor:

    - Honest asker (candor >= 0.80): returns subject's actual_completion ± 0.05 noise.
      This is the most reliable signal for detecting liars via triangulation.

    - Allied deceptive asker (same alliance_id as subject): returns subject's
      reported_completion - 0.02, mimicking "honest" slight underestimation while
      actually covering for the ally. The agent cannot tell this from a genuine estimate.

    - Unallied deceptive asker: returns actual_completion ± 0.15 (noisier but unbiased).

    The testimony_type is intentionally NOT in the observation — the agent must infer
    reliability by cross-referencing peer opinions with observable signals.

    Required params: asked_member_id (str), about_member_id (str)
    """
    asked_id = params.get("asked_member_id")
    about_id = params.get("about_member_id")

    if not asked_id:
        return ActionResult(error="Missing param: asked_member_id")
    if not about_id:
        return ActionResult(error="Missing param: about_member_id")
    if asked_id == about_id:
        return ActionResult(error="Cannot query a member about themselves")

    asked = state.get_member(asked_id)
    about = state.get_member(about_id)

    if asked is None:
        return ActionResult(error=f"Unknown asked_member_id: {asked_id!r}")
    if about is None:
        return ActionResult(error=f"Unknown about_member_id: {about_id!r}")

    # Deterministic noise seeded on step + member ids so the same query
    # at the same step returns the same result (idempotent within a step).
    import random as _random
    _rng = _random.Random(state.current_step * 1000 + hash(asked_id) % 997 + hash(about_id) % 997)

    allied = (
        asked.alliance_id is not None
        and asked.alliance_id == about.alliance_id
    )

    if asked.candor >= 0.80:
        # Honest member — gives accurate estimate with small noise
        noise = _rng.uniform(-0.05, 0.05)
        estimate = max(0.0, min(1.0, about.actual_completion + noise))
    elif allied:
        # Allied deceptive — covers for ally with a plausible slight underestimate
        estimate = max(0.0, min(1.0, about.reported_completion - 0.02))
    else:
        # Unallied deceptive — imprecise but not deliberately misleading
        noise = _rng.uniform(-0.15, 0.10)
        estimate = max(0.0, min(1.0, about.actual_completion + noise))

    return ActionResult(observation={
        "action_type": "query_peer_opinion",
        "asked_member_id": asked_id,
        "about_member_id": about_id,
        "peer_estimate_completion": round(estimate, 3),
        # NOTE: testimony reliability is NOT exposed — agent must infer from cross-referencing
    })


def action_force_truth(state: ProjectState, params: Dict[str, Any]) -> ActionResult:
    """
    Spend political capital to compel a member to reveal their actual completion.

    Cost: 1 budget + POLITICAL_CAPITAL_FORCE_TRUTH (3.0) PC.

    If PC is insufficient, returns a failure observation (budget is still spent —
    the PM tried and failed, which is costly). This incentivises the agent to
    build PC before using this action.

    Returns actual_completion and actual_availability of the target member.
    These are the ground-truth values never normally visible to the agent.

    Required params: member_id (str)
    """
    member_id = params.get("member_id")
    if not member_id:
        return ActionResult(error="Missing param: member_id")

    member = state.get_member(member_id)
    if member is None:
        return ActionResult(error=f"Unknown member_id: {member_id!r}")

    if state.political_capital < POLITICAL_CAPITAL_FORCE_TRUTH:
        return ActionResult(observation={
            "action_type": "force_truth",
            "success": False,
            "reason": "insufficient_political_capital",
            "political_capital_remaining": round(state.political_capital, 2),
            "required": POLITICAL_CAPITAL_FORCE_TRUTH,
        })

    state.political_capital -= POLITICAL_CAPITAL_FORCE_TRUTH

    return ActionResult(observation={
        "action_type": "force_truth",
        "success": True,
        "member_id": member_id,
        "actual_completion": round(member.actual_completion, 3),
        "actual_availability": round(member.actual_availability, 3),
        "political_capital_remaining": round(state.political_capital, 2),
    })


def action_trigger_whistleblower(state: ProjectState, params: Dict[str, Any]) -> ActionResult:
    """
    Spend political capital to activate an anonymous tip from an honest team member.

    Cost: 1 budget + POLITICAL_CAPITAL_WHISTLEBLOWER (6.0) PC.

    Reveals the member_id and name of the team member with the lowest current
    candor who has not already been caught this episode. The tip gives the agent
    a high-confidence starting point for cross-verification.

    If PC is insufficient, returns failure (budget still spent).

    No params required.
    """
    if state.political_capital < POLITICAL_CAPITAL_WHISTLEBLOWER:
        return ActionResult(observation={
            "action_type": "trigger_whistleblower",
            "success": False,
            "reason": "insufficient_political_capital",
            "political_capital_remaining": round(state.political_capital, 2),
            "required": POLITICAL_CAPITAL_WHISTLEBLOWER,
        })

    state.political_capital -= POLITICAL_CAPITAL_WHISTLEBLOWER

    # Reveal the uncaught member with lowest candor (most deceptive)
    candidates = [m for m in state.team_members if not m.caught_this_episode]
    if not candidates:
        candidates = list(state.team_members)  # fallback: all caught already

    worst = min(candidates, key=lambda m: m.candor)

    return ActionResult(observation={
        "action_type": "trigger_whistleblower",
        "success": True,
        "revealed_member_id": worst.member_id,
        "revealed_member_name": worst.name,
        "tip": (
            f"Anonymous tip: {worst.name} is significantly misrepresenting their "
            f"progress. Cross-verify with observable signals immediately."
        ),
        "political_capital_remaining": round(state.political_capital, 2),
    })


# ---------------------------------------------------------------------------
# COST-2 ACTION
# ---------------------------------------------------------------------------

def action_resolve_blocker(state: ProjectState, params: Dict[str, Any]) -> ActionResult:
    """
    Actively resolve a technical or process blocker on a task.

    Cost-2 action — most expensive single action.  Modelled as the PM
    spending significant time pairing, unblocking, or removing obstacles.

    Side effects:
    - Sets task.status = "in_progress" (from "blocked")
    - Boosts actual_progress by RESOLVE_BLOCKER_PROGRESS_BOOST
    - Updates ticket_last_changed_step (resets ticket age signal)
    - If task is on critical path: small morale boost to the whole team

    Required params: task_id (str), resolution_notes (str)
    """
    task_id = params.get("task_id")
    if not task_id:
        return ActionResult(error="Missing param: task_id")

    task = state.get_task(task_id)
    if task is None:
        return ActionResult(error=f"Unknown task_id: {task_id!r}")

    task.status = "in_progress"
    task.actual_progress = min(1.0, task.actual_progress + RESOLVE_BLOCKER_PROGRESS_BOOST)

    # Update ticket age signal for the assigned member
    if task.assigned_member_id:
        member = state.get_member(task.assigned_member_id)
        if member:
            update_ticket_change_step(member, state.current_step)
            if task.is_critical_path:
                member.morale = min(10.0, member.morale + 0.5)

    # Mark crisis as resolved if all its tasks are now done/near-done
    if task.crisis_id:
        crisis = state.get_crisis(task.crisis_id)
        if crisis:
            _check_crisis_resolution(crisis, state)

    return ActionResult(observation={
        "action_type": "resolve_blocker",
        "task_id": task_id,
        "new_status": task.status,
        "actual_progress": round(task.actual_progress, 3),
    })


# ---------------------------------------------------------------------------
# TERMINAL ACTION
# ---------------------------------------------------------------------------

def action_submit_recovery_plan(
    state: ProjectState, params: Dict[str, Any]
) -> ActionResult:
    """
    Submit the final recovery plan, ending the episode.

    Spec: "Terminal: submit_recovery_plan"

    The plan is stored in the state for logging; the episode is marked done.
    The counterfactual reward is computed by the environment after this action.

    Required params: plan_summary (str)
    Optional params: risk_items (list[str]), timeline (str)
    """
    plan_summary = params.get("plan_summary", "")
    if not plan_summary:
        return ActionResult(error="Missing param: plan_summary")

    state.done = True

    return ActionResult(
        observation={
            "action_type": "submit_recovery_plan",
            "plan_summary": plan_summary,
            "risk_items": params.get("risk_items", []),
            "timeline": params.get("timeline", ""),
            "episode_ended": True,
        },
        done=True,
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

ACTION_HANDLERS = {
    "query_status":             action_query_status,
    "query_member_report":      action_query_member_report,
    "query_observable_signals": action_query_observable_signals,
    "query_ticket":             action_query_ticket,
    "reassign_task":            action_reassign_task,
    "communicate":              action_communicate,
    "cut_scope":                action_cut_scope,
    "escalate_risk":            action_escalate_risk,
    "request_resource":         action_request_resource,
    "update_timeline":          action_update_timeline,
    "consult_expert":           action_consult_expert,
    "query_peer_opinion":      action_query_peer_opinion,
    "force_truth":             action_force_truth,
    "trigger_whistleblower":   action_trigger_whistleblower,
    "resolve_blocker":          action_resolve_blocker,
    "submit_recovery_plan":     action_submit_recovery_plan,
}


def dispatch_action(
    action: Dict[str, Any], state: ProjectState
) -> ActionResult:
    """
    Validate and dispatch an action dict to the appropriate handler.

    Invalid actions return an error ActionResult without touching the budget.
    Valid actions have their cost deducted and the handler called.

    Returns ActionResult with observation, error, done, budget_decremented.
    """
    # Validate format
    err = validate_action(action)
    if err:
        return ActionResult(error=err)

    action_type = action["action_type"]
    params = action["params"]

    cost = ACTION_COSTS[action_type]

    # Budget check (not for free actions)
    if cost > 0 and state.budget_remaining < cost:
        return ActionResult(error="Insufficient budget")

    # Deduct cost
    if cost > 0:
        state.budget_remaining -= cost

    # Record action
    state.actions_used.append(action_type)

    # Call handler
    handler = ACTION_HANDLERS[action_type]
    result = handler(state, params)
    result.budget_decremented = (cost > 0)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _acknowledge_pending_drift(state: ProjectState) -> None:
    """
    Mark any unacknowledged drift event as acknowledged.

    Called by communicate, update_timeline, and escalate_risk — the three
    actions the spec allows as valid drift acknowledgements.
    """
    for event in state.active_drift_events:
        if not event.acknowledged:
            event.acknowledged = True
    if state.pending_drift_event and not state.pending_drift_event.acknowledged:
        state.pending_drift_event.acknowledged = True


def _find_ally(member: TeamMember, state: ProjectState) -> Optional[TeamMember]:
    """
    Return the first other member sharing the same alliance_id, or None.

    Only returns a result if member.alliance_id is not None.
    Used by alibi generation in action_query_member_report.
    """
    if not member.alliance_id:
        return None
    for m in state.team_members:
        if m.member_id != member.member_id and m.alliance_id == member.alliance_id:
            return m
    return None


def check_crisis_resolution(crisis: "Crisis", state: ProjectState) -> None:
    """
    Mark a crisis as resolved if all its affected tasks are done or
    have actual_progress >= CRISIS_RESOLUTION_COMPLETION_THRESHOLD.
    """
    if crisis.is_resolved:
        return
    if not crisis.affected_task_ids:
        return
    for tid in crisis.affected_task_ids:
        task = state.get_task(tid)
        if task is None:
            continue
        if task.status != "done" and task.actual_progress < CRISIS_RESOLUTION_COMPLETION_THRESHOLD:
            return
    crisis.is_resolved = True


def _check_crisis_resolution(crisis: "Crisis", state: ProjectState) -> None:
    """Backward-compatible wrapper for internal call sites."""
    check_crisis_resolution(crisis, state)


def _expert_advisor(state: ProjectState) -> Dict[str, Any]:
    """
    Rule-based senior PM advisor — has access to TRUE state.

    Returns structured guidance covering:
    - Which crises to prioritise (by actual severity and resolution difficulty)
    - Which team members are under-performing (low actual_velocity vs. candor level)
    - Whether a drift event needs immediate acknowledgement
    - Budget spending recommendation

    This is entirely deterministic and rule-based.  No LLM involved.
    """
    advice: Dict[str, Any] = {
        "priority_crises": [],
        "suspicious_members": [],
        "drift_warning": None,
        "budget_warning": None,
        "recommended_actions": [],
    }

    # Priority crises: unresolved, sorted by severity desc
    unresolved = [c for c in state.crises if not c.is_resolved]
    unresolved.sort(key=lambda c: c.severity, reverse=True)
    advice["priority_crises"] = [
        {
            "crisis_id": c.crisis_id,
            "crisis_type": c.crisis_type,
            "severity": c.severity,
            "is_critical": c.severity >= EXPERT_CRITICAL_SEVERITY,
        }
        for c in unresolved[:3]
    ]

    # Suspicious members: low actual_velocity but possibly high reported_completion
    for m in state.team_members:
        if m.actual_velocity < 0.2 and m.reported_completion > 0.6:
            advice["suspicious_members"].append({
                "member_id": m.member_id,
                "name": m.name,
                "actual_velocity": round(m.actual_velocity, 3),
                "reported_completion": round(m.reported_completion, 3),
                "recommendation": "Cross-check with observable signals; consider reassignment",
            })

    # Drift warning
    unacked = [
        e for e in state.active_drift_events if not e.acknowledged
    ]
    if unacked:
        nearest = min(unacked, key=lambda e: e.acknowledgement_deadline)
        steps_left = nearest.acknowledgement_deadline - state.current_step
        advice["drift_warning"] = {
            "event_type": nearest.event_type,
            "steps_remaining_to_acknowledge": steps_left,
            "recommendation": "Call update_timeline or communicate immediately",
        }

    # Budget warning
    if state.budget_remaining <= 3:
        advice["budget_warning"] = {
            "budget_remaining": state.budget_remaining,
            "recommendation": "Submit recovery plan soon; prioritise resolve_blocker on critical tasks",
        }

    # Top recommended action
    if unresolved:
        critical = [c for c in unresolved if c.severity >= EXPERT_CRITICAL_SEVERITY]
        if critical:
            advice["recommended_actions"].append(
                f"resolve_blocker on tasks for crisis {critical[0].crisis_id}"
            )
    if advice["suspicious_members"]:
        advice["recommended_actions"].append(
            "reassign_task from low-velocity members to higher-availability members"
        )

    return advice
