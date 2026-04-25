# Tests reference

This document summarizes the intent of each test module under `tests/`.

## Running tests

Typical:

- `pytest tests/ -v`

## `tests/test_env.py`

Focus: environment mechanics, schemas, and invariants.

### Budget accounting

Asserts:

- free actions never decrement budget
- cost-1 decrements by 1
- cost-2 decrements by 2
- invalid actions do not decrement budget (error returned)

### Termination

Asserts:

- `submit_recovery_plan` ends the episode (`done=True`)
- draining budget to 0 ends the episode
- `MAX_STEPS` ends the episode
- reward is `0.0` before termination

### Observation format and non-leak

Asserts:

- initial obs has keys: `current_step`, `budget_remaining`, `team_members`, `crises`, `stakeholder`, `done`
- obs `team_members` dicts do NOT include `candor` or true fields (`actual_completion`, `actual_velocity`)
- `env.state()` DOES include candor (debug-only)

### Info dict

Asserts info includes:

- `cross_verify_rate`, `actions_used`, `budget_remaining`, `greedy_pm_score_so_far`, `active_crises`, `drift_events_fired`

## `tests/test_candor.py`

Focus: candor deception properties and signal truthfulness.

Asserts:

- for `candor=0.1`, reported completion exceeds actual by > 0.3 in >= 90% of samples
- for `candor=0.9`, reported is within 0.15 of actual in >= 90% of samples
- signals are derived from true state:
  - zero `actual_velocity` -> `commits_last_72h == 0`
  - ticket age increases when stalled
- signals dict contains the required keys and does not include candor/reported/actual values

## `tests/test_reward.py`

Focus: reward math and counterfactual reward sign.

Asserts:

- `RECOVERY_WEIGHT + CLIENT_WEIGHT + MORALE_WEIGHT == 1.0`
- `project_score` improves with higher recovery/satisfaction/morale
- counterfactual reward is positive when agent final state is clearly better than greedy
- counterfactual reward is negative when agent final state is worse
- budget exhaustion reduces the agent’s effective score (penalized)

## `tests/test_curriculum.py`

Focus: curriculum unlock behavior (training-side).

Asserts:

- unlock thresholds are correct over a sliding window (`CURRICULUM_WINDOW`)
- level is monotonically non-decreasing
- max level is capped
- unlock log records from/to levels and window mean

## How these tests help “LLM inference debugging”

If LLM inference fails, you may still see episodes run (fallback policy). Tests help ensure:

- you’re not accidentally leaking candor/truth into observations
- action costs and episode termination are behaving as expected
- reward math is stable (so strange score shifts likely come from policy changes, not reward bugs)
