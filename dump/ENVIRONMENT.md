# Environment reference: `env/`

This document describes the CrisisOps environment as implemented in:

- `env/environment.py`
- `env/state.py`
- `env/candor.py`
- `env/schema_drift.py`, `env/stakeholders.py`

It intentionally focuses on the **actual code behavior and schemas**.

## OpenEnv-style interface

`env/environment.py` defines `CrisisOpsEnv` with the following public interface:

- `reset(seed: Optional[int]) -> dict`
- `step(action: dict) -> (observation: dict, reward: float, done: bool, info: dict)`
- `state() -> dict` (debug/server only; includes hidden truth)

Episode max length:

- `MAX_STEPS = 30`

## Core state objects (truth lives here)

`env/state.py` defines dataclasses:

### TeamMember (truth + reported)

Hidden/true fields (must NOT appear in observations):

- `candor`, `candor_level`
- `actual_completion`, `actual_availability`, `actual_velocity`
- `inflation_bias`

Agent-visible (reported) fields:

- `reported_completion`
- `reported_availability`

Assignments + signals support:

- `assigned_task_ids`
- `ticket_last_changed_step`, `peer_mention_count`

Morale:

- `morale` (0–10)

### Task

- `status` is one of: `backlog | in_progress | blocked | done`
- agent can inspect a task via `query_ticket`
- truth for progress is `actual_progress` (0–1)

### Crisis

- `severity` is a float on roughly 0–10
- `affected_task_ids` ties crises to tasks
- `is_resolved` is the termination-relevant boolean

### StakeholderState

Tracks client + exec dynamics:

- `client_satisfaction` starts at `7.0`
- `exec_support` starts at `8.0`

### DriftEvent (schema drift)

- only fires for curriculum level >= 2
- drift fire step window is `DRIFT_STEP_MIN=6` to `DRIFT_STEP_MAX=12`
- must acknowledge within `DRIFT_ACK_WINDOW=3` steps

## Candor + deception mechanics

Implemented in `env/candor.py`.

### Candor levels and ranges

From `env/state.py`:

- `honest`: candor range 0.85–1.0
- `optimism_bias`: 0.50–0.70
- `self_preservation`: 0.10–0.40

### Reported completion formula

`reported_completion` is recomputed from truth each step:

```
reported = actual + (1 - candor) * inflation_bias
```

### Observable signals (always derived from truth)

The agent obtains signals only via the free action `query_observable_signals`.
Signals computed in `env/candor.py`:

- `ticket_age_days`: computed from `ticket_last_changed_step` and `STEPS_PER_DAY=2`
- `commits_last_72h`: derived from `actual_velocity` scaled by `COMMITS_FULL_VELOCITY=6` (0 if velocity <= `VELOCITY_STALL_THRESHOLD=0.01`)
- `peer_mentions`: computed from assignment overlap (capped at 10)

## Episode “physics” (what changes each step)

Key environment constants in `env/environment.py`:

- `COMPLETION_ADVANCE_PER_STEP = 0.065`
- `PROGRESS_CHANGE_THRESHOLD = 0.02` (used to decide if a ticket “changed” for signal purposes)
- `MORALE_DECAY_PER_STEP = 0.05`
- `BUDGET_EXHAUSTION_PENALTY = 0.30`

High-level step flow (simplified):

1. Validate and dispatch action (`env/actions.py`)
2. Advance the simulation one step (task progress, morale decay, etc.)
3. Potentially fire schema drift (level >= 2)
4. Step stakeholder state machines
5. Check episode termination (submit, budget exhaustion, max steps)
6. Compute reward **only when episode ends**

## Observation vs full state

### Observation (what agents see)

Built in `CrisisOpsEnv._build_observation()`.
Keys:

- `current_step`
- `budget_remaining`
- `team_members`: list of dicts with only _reported_ values + assignments
- `crises`: list of dicts (includes `description`, `is_resolved`, `affected_task_ids`)
- `stakeholder`: from `get_stakeholder_observation(s)`
- `done`

Important non-leak invariant:

- observations must NOT include `candor`, `actual_completion`, or `actual_velocity` for members

This is enforced by tests in `tests/test_env.py`.

### Full state (debug/server only)

`CrisisOpsEnv.state()` returns a full serializable dict and **does include**:

- member candor fields
- true progress/velocity fields

This exists for the MCP server and debugging.

## Info dict

Built in `CrisisOpsEnv._build_info()`.
Keys:

- `cross_verify_rate` (computed from counters on state)
- `actions_used`
- `budget_remaining`
- `greedy_pm_score_so_far` (set to `None` until termination)
- `active_crises`
- `drift_events_fired`

## Reward computation (counterfactual)

The counterfactual reward is computed only at termination in `CrisisOpsEnv._compute_reward()`:

- The environment deep-copies the initial state.
- It runs `GreedyPMBaseline` in a clone env to termination.
- It returns: `project_score(agent_final) - project_score(greedy_final)`.

Budget exhaustion penalty:

- If the agent hits budget=0 without submitting, `project_score` is reduced by `0.30` before counterfactual subtraction.

## Consistency checks / gotchas

- `submit_recovery_plan` is implemented as a terminal action in `env/actions.py`, but still has a cost of 1 in `ACTION_COSTS` (budget is decremented and the episode ends).
- `state()` includes candor and true progress by design; tests assert that _observations_ do not.
