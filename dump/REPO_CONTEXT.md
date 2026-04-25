# CrisisOps repo context (generated)

This folder contains reference notes that are intended to be **strictly consistent with the current code in this repository**.

## Index

- `ENVIRONMENT.md` — OpenEnv interface, state/observation schema, episode dynamics.
- `ACTIONS.md` — Full action catalog (action types, costs, params, side effects).
- `BASELINES.md` — Greedy baseline, oracle agent (calibration), random baseline, and the LLM eval agent at a high level.
- `LLM_INFERENCE_DEBUG.md` — How `baselines/llm_agent.py` selects providers and calls APIs; common failure modes; what fallback does.
- `TESTS.md` — What each test asserts; how to interpret failures.

## Why these docs exist

When LLM inference fails (bad key / quota / 403 / network), `baselines/llm_agent.py` falls back to a rule-based policy. That can mask the root cause if you only look at episode scores.

Use these notes to:

- confirm what the environment _should_ expose vs what is hidden (candor/actual state)
- confirm what actions are valid (and their exact schemas)
- confirm what the baselines are doing (so you can tell whether you’re seeing LLM behavior or fallback)
- quickly narrow down “LLM not working” vs “LLM working but behaving badly”
