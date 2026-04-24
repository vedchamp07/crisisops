# CrisisOps

Read prompt_v3.txt first. It is the authoritative spec.
Do not invent anything not in that file.
Do not simplify reward components or the candor system.

## Workflow

Run `python -m py_compile <file>` after each file to verify it compiles.
After calibration/calibrate.py is built, run it and paste results before continuing.
Grep for `candor` in agent observation code before moving past environment.py — it must never appear there.

## Status

Repo is built and calibrated. Calibration (2026-04-24):
- Greedy: 0.549 ✓ (target 0.45–0.55)
- Oracle: 0.887 (above target but gap is in spec)
- Gap: 0.338 ✓ (target 0.20–0.35)

## Known issue — fallback submits too early (FIX THIS NEXT)

`_fallback_action` in `baselines/llm_agent.py` exits in only 3 LLM steps:
  1. step 5: reassign crisis task to best-available member (cost 1)
  2. step 6: escalate_risk once (cost 1)
  3. step 7: submit (nothing left to do → exits immediately)

This is triggered whenever the LLM API is unavailable (HTTP 403/429).
The episode ends at step 7 out of 30. Tasks never complete → recovery_pct ≈ 0
→ 0.5 weight wasted → LLM agent scores ~0.31 vs greedy ~0.45.

Greedy beats the fallback because it runs all 30 steps continuously reassigning
tasks, giving honest members time to complete (tasks finish in ~23 steps).

### Root cause

The `"escalate_risk" not in self._memory["actions_taken"]` guard correctly prevents
repeated escalation, but then immediately falls through to `submit_recovery_plan`
even when there are 23 steps and 17 budget remaining.

### Fix required

Replace the submit fallthrough with a loop that keeps reassigning tasks
from low-availability members to high-availability ones until budget ≤ _SUBMIT_BUDGET_THRESHOLD.
Track which (task_id, member_id) pairs have been tried to avoid identical no-op reassigns,
but allow re-reassignment after members change.

Separately: both Gemini free tier (quota exhausted) and Groq (HTTP 403 error code 1010,
Cloudflare block) have been failing. A working API key is required to test actual LLM
performance. The fallback improvements are the floor; LLM performance is the ceiling.

## LLM API key status (2026-04-24)

- GOOGLE_API_KEY (AIzaSy...): free tier exhausted, limit=0
- GROQ_API_KEY (gsk_np...): HTTP 403 error code 1010 (Cloudflare block — key may need
  to be regenerated or Groq account verified)
- Recommend: get a fresh Groq key at console.groq.com or enable Gemini billing
