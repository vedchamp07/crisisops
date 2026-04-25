# Action catalog (authoritative to code)

This document describes the action system implemented in `env/actions.py`.

## Action types and costs

`env/actions.py` defines `ACTION_COSTS` (action_type -> integer cost):

### Free (cost 0)

- `query_status`
- `query_member_report`
- `query_observable_signals`
- `query_ticket`

### Cost-1 (cost 1)

- `reassign_task`
- `communicate`
- `cut_scope`
- `escalate_risk`
- `request_resource`
- `update_timeline`
- `consult_expert`

### Cost-2 (cost 2)

- `resolve_blocker`

### Terminal (ends episode; cost 1)

- `submit_recovery_plan`

Note: terminal-ness is a property of the handler (`action_submit_recovery_plan` sets `state.done=True`); cost is still deducted according to `ACTION_COSTS`.

## Validation and dispatch

Entry point: `dispatch_action(action: dict, state: ProjectState) -> ActionResult`

Validation rules (`validate_action`):

- action must be a dict with `action_type` and `params`
- `action_type` must be one of `VALID_ACTION_TYPES`
- `params` must be a dict

Budget rules:

- invalid actions return an error and **do not decrement budget**
- valid paid actions decrement budget before running the handler
- if `budget_remaining < cost`, returns error `Insufficient budget`

## Per-action schema + behavior

Below: required params, optional params, and notable side effects.

### `query_status` (free)

Params: `{}`
Returns high-level project summary:

- active + resolved crises (reported/structural only)
- team summary with `reported_completion`, `reported_availability`, `assigned_task_ids`

### `query_member_report` (free)

Required params:

- `member_id: str`
  Side effects:
- increments `state.total_member_query_calls` (for cross-verify metrics)
  Returns:
- the member’s _reported_ completion and availability

### `query_observable_signals` (free)

Required params:

- `member_id: str`
  Side effects:
- returns objective signals derived from true state (via `env/candor.py`)
- increments `state.cross_verify_calls` and `state.total_member_query_calls`
  Returns:
- `signals: { ticket_age_days, commits_last_72h, peer_mentions }`

### `query_ticket` (free)

Required params:

- `task_id: str`
  Returns task metadata:
- status, assignment, critical-path flag
- drift-related flags (`is_deprioritized`, `is_compliance_blocked`)

### `reassign_task` (cost 1)

Required params:

- `task_id: str`
- `to_member_id: str`
  Side effects:
- moves assignment between members
- updates `ticket_last_changed_step` (resets ticket age)
- morale boost for receiving member (`REASSIGN_MORALE_BOOST = 0.3`)
- if task was `blocked`, it is set to `in_progress`

### `communicate` (cost 1)

Required params:

- `message_type: str`
- `content: str`
  Optional params:
- `target: "client" | "exec" | "both"` (default `both`)
  Side effects:
- resets client “last communicated step” (prevents decay penalty)
- if `message_type == "proactive_escalation_with_plan"`: increases client satisfaction by `CLIENT_GAIN_PROACTIVE`
- if `message_type == "risk_communication"`: increases exec support by `EXEC_GAIN_RISK_COMM`
- acknowledges any pending drift event

### `cut_scope` (cost 1)

Required params:

- `task_id: str`
- `justification: str` (documented as required)
  Constraints:
- cannot cut scope on critical-path tasks
  Side effects:
- sets `task.is_deprioritized=True`, `task.status="backlog"`
- morale penalty to assigned member (`CUT_SCOPE_MORALE_PENALTY = 0.5`)

### `escalate_risk` (cost 1)

Required params:

- `crisis_id: str`
- `risk_description: str` (documented as required)
  Side effects:
- increases exec support by `EXEC_GAIN_RISK_COMM`
- acknowledges any pending drift event

### `request_resource` (cost 1)

Required params (per docstring):

- `resource_type: str`
- `target_member_id: str`
  Behavior:
- if `exec_support < EXEC_SUPPORT_BUDGET_THRESHOLD` (5.0): returns `success=False` _without raising error_
- if last budget request did not include a timeline update: exec support decays by `EXEC_DECAY_BUDGET_NO_TIMELINE`
- on success, boosts target member’s `actual_availability` and `actual_velocity`

### `update_timeline` (cost 1)

Required params:

- `new_completion_date: str` (ISO date string)
  Optional params:
- `task_estimates: { task_id: float }`
  Side effects:
- sets stakeholder flags used by `request_resource`
- acknowledges drift

### `consult_expert` (cost 1)

Params: `{}`
Returns deterministic “senior PM advisor” guidance based on TRUE state.
Output includes:

- priority crises
- suspicious members (low true velocity but high reported completion)
- drift warning
- budget warning

### `resolve_blocker` (cost 2)

Required params:

- `task_id: str`
- `resolution_notes: str`
  Side effects:
- sets task to `in_progress`
- boosts `actual_progress` by `RESOLVE_BLOCKER_PROGRESS_BOOST = 0.25`
- updates ticket change step for signals

### `submit_recovery_plan` (terminal; cost 1)

Required params:

- `plan_summary: str`
  Optional params:
- `risk_items: list[str]`
- `timeline: str`
  Side effects:
- sets `state.done=True`
- ends episode immediately

## Drift acknowledgement (important)

`_acknowledge_pending_drift` is called by:

- `communicate`
- `update_timeline`
- `escalate_risk`

Those actions are explicitly treated as “acknowledging” schema drift.
