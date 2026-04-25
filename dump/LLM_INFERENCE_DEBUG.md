# LLM inference + debugging guide (repo-specific)

This document is for debugging “LLM inference isn’t working” in this repo.
Source of truth: `baselines/llm_agent.py`.

## 1) Provider selection (auto-detected)

Provider detection is done by `_detect_provider()`.

It scans environment variables in **this exact order** (first match wins):

1. `LLM_BASE_URL` (uses `LLM_API_KEY`)
2. `OPENAI_API_KEY`
3. `ANTHROPIC_API_KEY`
4. `GOOGLE_API_KEY`
5. `OPENROUTER_API_KEY`
6. `TOGETHER_API_KEY`
7. `GROQ_API_KEY`
8. `OLLAMA_MODEL`

If none are set, the script prints an error and exits.

### Practical gotcha

If you have an old `LLM_BASE_URL` exported in your shell, it will override everything else.

Also: if `LLM_BASE_URL` is set but `LLM_API_KEY` is not, the code uses the string `"no-key"` — which will fail against endpoints that require auth.

## 2) How API calls are made

No SDKs are required; calls are done with `urllib.request`.

### OpenAI-compatible (`/chat/completions`)

Used for:

- OpenAI
- Groq
- Together
- OpenRouter
- custom endpoints via `LLM_BASE_URL`
- local Ollama (`base_url=http://localhost:11434/v1`)

Request:

- `POST {base_url}/chat/completions`
- JSON body includes: `model`, `messages`, `temperature`, `max_tokens`
- Header includes `Authorization: Bearer {api_key}` (even for some local endpoints)

### Anthropic

- Calls `https://api.anthropic.com/v1/messages`
- Splits system prompt into the `system` field
- Sends other turns via `messages`

### Google Gemini

- Calls `.../v1beta/models/{model}:generateContent?key=...`
- Uses `systemInstruction` for system text and Gemini’s `contents` format

## 3) What happens on failure

`LLMAgent.act()` wraps the API call in `try/except`.

If any exception occurs (HTTP errors included), it switches to:

- `_fallback_action(observation)`

So: the evaluation can “look like it ran” even if the LLM never produced output.

### How to tell you’re in fallback

Run with `--verbose`.
On LLM error, it prints something like:

- `[LLM error: ...] → fallback`

## 4) What fallback does (important for debugging)

Fallback is a deterministic policy designed to keep the episode going without LLM.
Priority order:

1. `query_observable_signals` for any unverified member
2. `reassign_task` away from detected deceptive members (if any)
3. If crisis unresolved and budget healthy: reassign crisis task to most available member; otherwise `escalate_risk` once
4. Keep reassigning unresolved crisis tasks to the most available member (with a memory of attempted (task,member) pairs)
5. Communicate occasionally, then `submit_recovery_plan` when truly nothing is left

This means poor performance in LLM mode can actually be “fallback mode performance”.

## 5) Submit thresholds and why they matter

There are multiple submit-related thresholds in `baselines/llm_agent.py`:

- `_SUBMIT_BUDGET_THRESHOLD = 3` is used for forced-submit behavior.
- The system prompt text (the instructions shown to the LLM) says “Only submit when crises resolved OR budget <= 5”.
- `_block_premature_submit()` intercepts LLM submits unless `budget <= 5` or no unresolved crises.

When debugging, be clear whether:

- the **LLM** is attempting early submit
- or Python-side forced submit is firing

## 6) Prompt/action-list consistency gotcha

The environment supports 13 action types (see `env/actions.py`).
The LLM’s `SYSTEM_PROMPT` includes an “ACTION REFERENCE” section that lists only a subset of cost-1 actions.

That doesn’t break the environment, but it can reduce the chance the LLM uses omitted actions like `cut_scope` or `request_resource`.

## 7) Fast triage checklist

1. Confirm provider selection:
   - print your relevant env vars (`OPENAI_API_KEY`, `LLM_BASE_URL`, etc.)
2. Run a tiny eval with verbose:
   - `python -m baselines.llm_agent --episodes 1 --verbose`
3. If you see fallback:
   - capture the HTTP error string (it’s included in the RuntimeError)
4. If you don’t see fallback but behavior is bad:
   - check whether forced-gather / anti-loop overrides are dominating actions

## 8) Related tooling

- `baselines/replay.py` prints a narrative trace for a single episode (useful to see whether actions look LLM-driven or heuristic-driven).
