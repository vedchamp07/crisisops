# Baselines + agents reference

This document summarizes the non-training agents in the repo:

- Greedy baseline (`reward/baseline.py`)
- Oracle agent (calibration ceiling; `calibration/calibrate.py`)
- Random agent (`baselines/random_agent.py`)
- LLM evaluation agent (`baselines/llm_agent.py`)

## Greedy baseline (`GreedyPMBaseline`)

Source: `reward/baseline.py`.

Core idea:

- deterministic, trusts **reported** values
- never calls `query_observable_signals`

Decision procedure (`act(state)`), in priority order:

1. If `state.budget_remaining <= 2`: submit recovery plan.
2. If communication interval reached: communicate.
3. If any crisis unresolved: pick highest-severity unresolved crisis and reassign one task in it to highest **reported** availability member.
4. Else: `query_status`.

Implementation details that matter:

- Communication cadence is controlled by `GREEDY_COMM_INTERVAL` (currently `7`).
- Reassign “ping-pong” is prevented with a cooldown (`_reassign_cooldown = 3` steps per task).

### Consistency note

The docstring/spec comments mention “communicates once every 5 steps”, but the constant in code is `GREEDY_COMM_INTERVAL = 7`.

## Oracle agent (`OracleAgent`)

Source: `calibration/calibrate.py`.

Purpose:

- rule-based ceiling for calibration targets
- explicitly allowed to use TRUE state (not just what the agent sees)

Decision procedure (simplified):

1. Query observable signals for any member with `actual_velocity < 0.2` (once per member).
2. Acknowledge any unacknowledged drift immediately via `update_timeline`.
3. Resolve blockers on critical-path tasks (`resolve_blocker`) if budget allows.
4. Reassign tasks away from members with `actual_velocity < 0.3` to the member with highest `actual_availability` (with a per-task “once” guard).
5. Communicate every 5 steps.
6. Submit if all crises resolved.
7. If budget > 5, consult expert.
8. Else fallback to `query_status`.

## Random agent (`RandomAgent`)

Source: `baselines/random_agent.py`.

Behavior:

- uniformly samples a valid action type each step
- generates syntactically valid params from current state
- avoids early terminal submit: will not choose `submit_recovery_plan` until at least 5 budget points have been spent

This is meant for reward-range sanity checks.

## LLM evaluation agent (`LLMAgent`)

Source: `baselines/llm_agent.py`.

This agent wraps an external chat LLM and runs it against the environment.
Key pieces (high level):

- Provider auto-detection from environment variables.
- A large `SYSTEM_PROMPT` that instructs cross-verification and structured JSON-only output.
- Python-side memory (`signals`, `member_reports`, `deceptive`, `actions_taken`, `step`).

### Python-side enforcement

Even when using an LLM, the agent enforces several rules in Python:

- Forced gather: it will call `query_observable_signals` until every team member has been signal-verified.
- Forced one-time `communicate` after gather (after a step buffer).
- Anti-loop override: if it detects repeated free-query loops, it injects a more useful action.
- Premature-submit blocker: if the LLM tries to `submit_recovery_plan` while crises are unresolved and budget is healthy, it overrides to keep acting.

### Fallback mode

If the LLM call errors (HTTP errors, etc.), the agent switches to `_fallback_action`.
See `LLM_INFERENCE_DEBUG.md` for details.
