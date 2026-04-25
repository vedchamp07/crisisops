"""
baselines/llm_agent.py — LLM-based agent for evaluating CrisisOps v2.

Runs any OpenAI-compatible chat model as the PM agent and reports
project_score alongside the greedy and oracle baselines.

Supported providers (auto-detected from env vars):
    OPENAI_API_KEY          → OpenAI  (default model: gpt-4o-mini)
    ANTHROPIC_API_KEY       → Anthropic via messages API (default: claude-sonnet-4-20250514)
    GOOGLE_API_KEY          → Google Gemini (default: gemini-2.0-flash)
    OPENROUTER_API_KEY      → OpenRouter (default: openrouter/auto)
    TOGETHER_API_KEY        → Together AI (default: meta-llama/Llama-3-70b-chat-hf)
    GROQ_API_KEY            → Groq (default: llama-3.1-70b-versatile)
    OLLAMA_MODEL            → Local Ollama (no key needed, default: llama3.1)
    LLM_BASE_URL + LLM_API_KEY → Any OpenAI-compatible endpoint

Usage:
    # OpenAI
    export OPENAI_API_KEY=sk-...
    python -m baselines.llm_agent

    # Anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m baselines.llm_agent --model claude-sonnet-4-20250514

    # Local Ollama (no key needed)
    export OLLAMA_MODEL=llama3.1
    python -m baselines.llm_agent

    # Custom OpenAI-compatible endpoint
    export LLM_BASE_URL=http://localhost:8080/v1
    export LLM_API_KEY=any
    python -m baselines.llm_agent --model my-model

    # Options
    python -m baselines.llm_agent --episodes 5 --model gpt-4o --seed 42 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Provider detection and configuration
# ---------------------------------------------------------------------------

_PROVIDERS: List[Dict[str, str]] = [
    # order matters — first match wins
    {"env_key": "LLM_BASE_URL",      "key_env": "LLM_API_KEY",       "base_url": None,                                "default_model": "default"},
    {"env_key": "OPENAI_API_KEY",     "key_env": "OPENAI_API_KEY",    "base_url": "https://api.openai.com/v1",         "default_model": "gpt-4o-mini"},
    {"env_key": "ANTHROPIC_API_KEY",  "key_env": "ANTHROPIC_API_KEY", "base_url": "anthropic",                         "default_model": "claude-sonnet-4-20250514"},
    {"env_key": "GOOGLE_API_KEY",     "key_env": "GOOGLE_API_KEY",    "base_url": "google",                            "default_model": "gemini-2.0-flash"},
    {"env_key": "OPENROUTER_API_KEY", "key_env": "OPENROUTER_API_KEY","base_url": "https://openrouter.ai/api/v1",      "default_model": "openrouter/auto"},
    {"env_key": "TOGETHER_API_KEY",   "key_env": "TOGETHER_API_KEY",  "base_url": "https://api.together.xyz/v1",       "default_model": "meta-llama/Llama-3-70b-chat-hf"},
    {"env_key": "GROQ_API_KEY",       "key_env": "GROQ_API_KEY",     "base_url": "https://api.groq.com/openai/v1",    "default_model": "llama-3.1-70b-versatile"},
    {"env_key": "OLLAMA_MODEL",       "key_env": None,                "base_url": "http://localhost:11434/v1",          "default_model": "llama3.1"},
]


def _detect_provider() -> Tuple[str, Optional[str], str]:
    """Return (base_url, api_key, default_model) from environment."""
    for p in _PROVIDERS:
        val = os.environ.get(p["env_key"], "")
        if not val:
            continue
        # custom endpoint
        if p["env_key"] == "LLM_BASE_URL":
            return val, os.environ.get("LLM_API_KEY", "no-key"), p["default_model"]
        # ollama — no key
        if p["key_env"] is None:
            return p["base_url"], "ollama", os.environ.get("OLLAMA_MODEL", p["default_model"])
        return p["base_url"], val, p["default_model"]

    print("ERROR: No LLM provider detected.\n"
          "Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,\n"
          "OPENROUTER_API_KEY, TOGETHER_API_KEY, GROQ_API_KEY, OLLAMA_MODEL,\n"
          "or LLM_BASE_URL + LLM_API_KEY.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Chat completion — unified interface across providers
# ---------------------------------------------------------------------------

def _call_openai_compatible(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 768,
) -> str:
    """Call an OpenAI-compatible chat/completions endpoint and return the text."""
    import urllib.request
    import urllib.error

    url = f"{base_url.rstrip('/')}/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        err_body = e.read().decode() if e.fp else ""
        raise RuntimeError(f"HTTP {e.code} from {url}: {err_body}") from e


def _call_anthropic(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 768,
) -> str:
    """Call the Anthropic messages API directly (no SDK needed)."""
    import urllib.request
    import urllib.error

    # Split system from user/assistant messages
    system_text = ""
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text += m["content"] + "\n"
        else:
            chat_messages.append(m)

    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_text.strip(),
        "messages": chat_messages,
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["content"][0]["text"]
    except urllib.error.HTTPError as e:
        err_body = e.read().decode() if e.fp else ""
        raise RuntimeError(f"HTTP {e.code} from Anthropic: {err_body}") from e


def _call_google(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 768,
) -> str:
    """Call Google Gemini generateContent API directly (no SDK needed)."""
    import urllib.request
    import urllib.error

    # Build Gemini contents format
    system_text = ""
    contents = []
    for m in messages:
        if m["role"] == "system":
            system_text += m["content"] + "\n"
        elif m["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": m["content"]}]})
        elif m["role"] == "assistant":
            contents.append({"role": "model", "parts": [{"text": m["content"]}]})

    body_dict: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_text.strip():
        body_dict["systemInstruction"] = {"parts": [{"text": system_text.strip()}]}

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")
    body = json.dumps(body_dict).encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        err_body = e.read().decode() if e.fp else ""
        raise RuntimeError(f"HTTP {e.code} from Google: {err_body}") from e


def call_llm(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 768,
) -> str:
    """Dispatch to the right provider."""
    if base_url == "anthropic":
        return _call_anthropic(api_key, model, messages, temperature, max_tokens)
    if base_url == "google":
        return _call_google(api_key, model, messages, temperature, max_tokens)
    return _call_openai_compatible(base_url, api_key, model, messages, temperature, max_tokens)


# FIX: 1 Add hard anti-query-loop rule to inference prompt for decisive actions.
# ---------------------------------------------------------------------------
# System prompt — redesigned to force cross-verification and structured reasoning
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an AI project manager recovering a failing software project.
CRITICAL: Some team members LIE about their progress to avoid accountability.

The greedy baseline always trusts self-reports and FAILS. You win by detecting deception through observable signals.

=== DECEPTION DETECTION RULES ===
KEY INSIGHT: A deceptive member's initial_reported_completion is HIGH (set when episode starts),
but their task's actual_progress is LOW (5-20%).  After physics updates, their reported_completion
drops to match actual — but the AGENT MEMORY block shows the original "initial_claimed" value.
ALWAYS compare "initial_claimed" in AGENT MEMORY against commits_last_72h:

  • initial_claimed > 0.55  AND  commits_last_72h ≤ 2  → DECEPTIVE (⚠ flag in memory)
  • initial_claimed > 0.55  AND  ticket_age_days > 3  AND  commits_last_72h == 0  → CONFIRMED LYING
  • Honest member: commits_last_72h >= 4  (velocity 0.6-0.9 → 4-5 commits expected)

=== MANDATORY PROCEDURE EACH TURN ===
Follow these steps in order. Only take ONE action per turn.

STEP A — GATHER (FREE, costs no budget):  # FIX-1: restored gather instruction for re-verification
  After your first turn, some members may still be unverified. If any  # FIX-1
  team member has NOT been cross-verified yet, call  # FIX-1
  query_observable_signals for them — this is always valid. However,  # FIX-1
  after 2 consecutive free queries you MUST take a paid action before  # FIX-1
  querying again.  # FIX-1

STEP B — DETECT (do this mentally, no action needed):
  Compare each member's reported_completion with their signals.
  Deceptive members have tasks that are NOT actually progressing — reassigning them helps.
  Check AGENT MEMORY for members already flagged as DECEPTIVE / SUSPICIOUS.

STEP C — ACT (pick the highest-impact paid action, EVERY turn):
  1. DECEPTIVE member assigned to an unresolved crisis task → reassign_task to best available member.
     This is ALWAYS priority 1 when a deceptive member holds a task.
  2. Steps since last communicate >= 4 → communicate {"message_type": "proactive_escalation_with_plan", ...}
  3. Blocked critical-path task and budget > 4 → resolve_blocker
  4. Any unresolved crisis and budget > 3 → reassign_task or escalate_risk
  5. Budget ≤ 5 OR all crises resolved → submit_recovery_plan IMMEDIATELY.  # FIX-5: was ≤ 3, raised to match intercept threshold
     WARNING: Do NOT submit just because you have communicated and escalated.  # FIX-5
     A recovery plan requires tasks to COMPLETE. Keep reassigning until you  # FIX-5
     see is_resolved=true in the crisis list OR budget reaches 5.  # FIX-5

MANDATORY ACTION RULE: You may call query_status or query_observable_signals at most TWICE IN A ROW. After two consecutive information-gathering actions, your next action MUST be a cost-1 or cost-2 decision action: reassign_task, communicate, cut_scope, escalate_risk, request_resource, update_timeline, consult_expert, or resolve_blocker. Failure to follow this rule means the project fails.  # FIX-1: restored mandatory action rule

=== REQUIRED OUTPUT FORMAT ===
First output a <think> block (stripped before parsing) with:
  - For each member in AGENT MEMORY: initial_claimed vs commits → deceptive or honest?
  - Remaining budget and whether any member holds an unresolved crisis task
  - The single highest-impact action this turn and why

Then output exactly ONE JSON object (example below):
{"reasoning": "dev_bob claimed 74% initially but only 1 commit — deceptive. Reassigning task_int_1 to dev_alice (availability=0.85).", "action_type": "reassign_task", "params": {"task_id": "task_int_1", "to_member_id": "dev_alice"}}

=== ACTION REFERENCE ===
FREE (query_* never costs budget):
  query_status {}
  query_member_report {"member_id": "<id>"}
  query_observable_signals {"member_id": "<id>"}
  query_ticket {"task_id": "<id>"}

COST-1 (deduct 1 from budget):
  reassign_task {"task_id": "<id>", "to_member_id": "<id>"}
  communicate {"message_type": "proactive_escalation_with_plan"|"risk_communication"|"status_update", "content": "<text>", "target": "both"}
  escalate_risk {"crisis_id": "<id>", "risk_description": "<text>"}
  update_timeline {"new_completion_date": "<date>", "task_estimates": {}}
  consult_expert {}
  submit_recovery_plan {"plan_summary": "<text>", "risk_items": [], "timeline": "<date>"}

COST-2 (deduct 2 from budget):
  resolve_blocker {"task_id": "<id>", "resolution_notes": "<text>"}

Budget starts at 20. Exhausting budget without submitting = -0.30 penalty to your score.
Only submit_recovery_plan when is_resolved=true for all crises, OR budget <= 5.  # FIX-5: replaced early-submit instruction
Submitting early (before tasks complete) wastes the entire episode.  # FIX-5
Keep reassigning tasks every turn until one of these conditions is met.  # FIX-5
"""

# Deception detection thresholds (tuned to candor.py signal scales)
# COMMITS_FULL_VELOCITY=6 in candor.py → honest high-velocity member has 4-6 commits.
# Self-preservation member (velocity 0.05-0.25) has 0-1 commits.
_DECEPTION_REPORT_THRESHOLD  = 0.55  # reported_completion floor for all deception rules
_DECEPTION_TICKET_AGE_DAYS   = 2     # FIX-1: was 3, detects at step 4 not step 6
# Raised from 1 to 2: velocity=0.25 (max of VELOCITY_LOW) produces round(1.5)=2 commits.
# Honest members (velocity 0.6-0.9) produce 4-5 commits, so threshold=2 has no false positives.
_DECEPTION_COMMITS_THRESHOLD = 2
_SUSPICION_TICKET_AGE_DAYS   = 5     # ticket age floor for suspicious-only rule

# Budget level below which the agent should stop spending and submit
_SUBMIT_BUDGET_THRESHOLD = 3


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

class LLMAgent:
    """
    LLM-based PM agent with cross-verification memory and structured reasoning.

    Key improvements over the naive baseline:
    - Persistent per-episode memory: tracks which members have been signal-verified
      and which are suspected deceptive.
    - Memory context injected into every LLM turn so the model doesn't rely on
      conversation history alone (which gets trimmed).
    - Structured {reasoning, action_type, params} output format enables
      chain-of-thought before each action.
    - Local deception detection: Python-side cross-reference of signals vs.
      reported values, so the agent's memory stays consistent even if the LLM
      hallucinates a different member.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.3,
        verbose: bool = False,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._verbose = verbose
        self._messages: List[Dict[str, str]] = []
        self._memory: Dict[str, Any] = {}

    def reset(self) -> None:
        """Clear conversation history and memory for a new episode."""
        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._memory = {
            # member_id -> {ticket_age_days, commits_last_72h, peer_mentions}
            "signals": {},
            # FIX: 1 Track queried self-reports so gather phase uses both query channels.
            "member_reports": set(),
            # member_id -> reported_completion from the VERY FIRST observation (pre-physics).
            # Candor.py inflates reported based on member.actual_completion which the scenario
            # sets to 0.60-0.75 for deceptive members.  After the first physics step,
            # _advance_actual_completions overwrites member.actual_completion with the much
            # lower task.actual_progress, collapsing reported_completion below detection
            # thresholds.  We cache the initial value so detection always sees the original
            # inflated claim, not the post-physics corrected one.
            "initial_reports": {},
            # member_id -> human-readable reason string
            "deceptive": {},
            # list of action_type strings attempted this episode
            "actions_taken": [],
            "step": 0,
        }

    # ------------------------------------------------------------------
    # Memory update — called at start of each act() from incoming obs
    # ------------------------------------------------------------------

    def _update_memory_from_obs(self, obs: Dict[str, Any]) -> None:
        """
        Extract observable signal data from the incoming observation and run
        deception detection against reported values.

        Uses initial_reports (cached before physics overwrites them) so that
        the inflated self-claim from episode start is always available for
        comparison even after member.actual_completion drops to match task progress.

        Called before building the user message so the memory context is
        always up-to-date before the LLM sees the observation.
        """
        # Cache initial reported_completion on the very first observation.
        # The scenario sets member.actual_completion=0.60-0.75 for deceptive members
        # so their initial reported_completion is high.  After the first physics step,
        # _advance_actual_completions overwrites this with task.actual_progress (~0.10),
        # collapsing reported below the detection threshold.  Caching here preserves the
        # original inflated claim for cross-referencing with signals gathered later.
        if not self._memory["initial_reports"]:
            for m in obs.get("team_members", []):
                self._memory["initial_reports"][m["member_id"]] = m.get("reported_completion", 0.0)

        # Extract signals if the last action was query_observable_signals
        if obs.get("action_type") == "query_observable_signals":
            member_id = obs.get("member_id")
            signals = obs.get("signals", {})
            if member_id and signals:
                self._memory["signals"][member_id] = signals

        if obs.get("action_type") == "query_member_report":
            member_id = obs.get("member_id")
            if member_id:
                self._memory["member_reports"].add(member_id)

        # Build current report map from base observation (always present)
        current_report_map: Dict[str, float] = {
            m["member_id"]: m.get("reported_completion", 0.0)
            for m in obs.get("team_members", [])
        }

        # Run deception detection for any member we have signals for
        for mid, sigs in self._memory["signals"].items():
            if mid in self._memory["deceptive"]:
                continue
            # Use the HIGHER of initial claim vs current — physics overwrites
            # member.actual_completion after step 1, so current_reported drops.
            # The initial claim is the deception evidence we want to check.
            current_reported = current_report_map.get(mid, 0.0)
            initial_reported = self._memory["initial_reports"].get(mid, current_reported)
            reported = max(initial_reported, current_reported)

            ticket_age = sigs.get("ticket_age_days", 0)
            commits = sigs.get("commits_last_72h", 0)

            # Rule 1: stale ticket + zero commits + high report = confirmed lying
            deceptive_stale = (
                reported > _DECEPTION_REPORT_THRESHOLD
                and ticket_age > _DECEPTION_TICKET_AGE_DAYS
                and commits == 0
            )
            # Rule 2: few commits + high initial claim.  ticket_age > 0 guard removed:
            # signals are gathered at env step 0-2 where ticket_age is always 0, so
            # requiring ticket_age > 0 blocked all early detection.
            # Honest fast members have commits 4-5 (velocity 0.6-0.9) so
            # _DECEPTION_COMMITS_THRESHOLD=2 does not cause false positives.
            deceptive_commits = (
                reported > _DECEPTION_REPORT_THRESHOLD
                and commits <= _DECEPTION_COMMITS_THRESHOLD
            )
            # Rule 3: stale ticket even with modest commits = suspicious
            suspicious = (
                not deceptive_stale and not deceptive_commits
                and reported > _DECEPTION_REPORT_THRESHOLD
                and commits <= _DECEPTION_COMMITS_THRESHOLD
                and (ticket_age > _SUSPICION_TICKET_AGE_DAYS
                     or reported > _DECEPTION_REPORT_THRESHOLD)
            )

            if deceptive_stale or deceptive_commits:
                reason = (
                    f"initial_reported={initial_reported:.2f}, current={current_reported:.2f} "
                    f"but ticket_age={ticket_age}d, commits={commits} → likely lying"
                )
                self._memory["deceptive"][mid] = reason
                if self._verbose:
                    print(f"  [DECEPTION DETECTED] {mid}: {reason}")
            elif suspicious:
                reason = (
                    f"initial_reported={initial_reported:.2f}, current={current_reported:.2f} "
                    f"but ticket_age={ticket_age}d, commits={commits} → suspicious"
                )
                self._memory["deceptive"][mid] = reason
                if self._verbose:
                    print(f"  [SUSPICIOUS] {mid}: {reason}")

    def _build_memory_context(self, obs: Dict[str, Any]) -> str:
        """
        Build a compact memory summary to prepend to the current observation.

        Highlights unverified members (next priority), deceptive members
        (decision-critical), verified signals, and action history.
        This prevents the LLM from having to reconstruct state from raw
        conversation history that may be partially trimmed.
        """
        lines: List[str] = ["=== AGENT MEMORY ==="]

        team = obs.get("team_members", [])
        unverified = [
            m["member_id"] for m in team
            if m["member_id"] not in self._memory["signals"]
        ]
        if unverified:
            lines.append(
                f"UNVERIFIED MEMBERS (priority: call query_observable_signals "
                f"for each before spending budget): {unverified}"
            )

        deceptive = self._memory["deceptive"]
        if deceptive:
            lines.append("DECEPTIVE / SUSPICIOUS MEMBERS (do NOT trust their reports):")
            for mid, reason in deceptive.items():
                lines.append(f"  - {mid}: {reason}")

        verified = self._memory["signals"]
        initial_reports = self._memory.get("initial_reports", {})
        if verified:
            lines.append("CROSS-VERIFIED SIGNALS (initial claim → what signals show):")
            for mid, sigs in verified.items():
                tag = " [DECEPTIVE]" if mid in deceptive else ""
                initial = initial_reports.get(mid)
                ticket_age = sigs.get("ticket_age_days", 0)
                commits = sigs.get("commits_last_72h", 0)
                initial_str = f"initial_claimed={initial:.2f}" if initial is not None else ""
                # Pre-compute contradiction flag for the LLM
                if (initial is not None and initial > _DECEPTION_REPORT_THRESHOLD
                        and commits <= _DECEPTION_COMMITS_THRESHOLD):
                    flag = (f"⚠ CONTRADICTION: claimed {initial:.0%} but only "
                            f"{commits} commits + {ticket_age}d stale ticket — likely deceptive")
                else:
                    flag = "signals consistent"
                lines.append(
                    f"  - {mid}{tag}: {initial_str},"
                    f" ticket_age={ticket_age}d,"
                    f" commits={commits},"
                    f" peers={sigs.get('peer_mentions')} | {flag}"
                )

        acts = self._memory["actions_taken"]
        budget = obs.get("budget_remaining", 20)
        if acts:
            lines.append(
                f"ACTIONS THIS EPISODE ({len(acts)} total, last 6): "
                f"{', '.join(acts[-6:])}"
            )
        if budget <= _SUBMIT_BUDGET_THRESHOLD:
            lines.append(
                f"*** BUDGET WARNING: only {budget} left — submit_recovery_plan NOW ***"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: str, observation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse LLM response into an action dict.

        Accepts structured format:
            {"reasoning": "...", "action_type": "...", "params": {...}}
        Also accepts legacy format:
            {"action_type": "...", "params": {...}}
        Falls back to a smart action (never query_status) on parse failure.
        """
        if self._verbose:
            print(f"  [RAW LLM] {response[:400]}{'...' if len(response) > 400 else ''}")
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in response")
            data = json.loads(response[start:end])

            reasoning = data.get("reasoning", "")
            if reasoning and self._verbose:
                print(f"  [REASONING] {reasoning[:220]}")

            if "action_type" in data and "params" in data:
                action_type = data["action_type"]
                from env.actions import VALID_ACTION_TYPES
                if action_type not in VALID_ACTION_TYPES:
                    raise ValueError(f"Unknown action_type: {action_type!r}. "
                                     f"Valid: {sorted(VALID_ACTION_TYPES)}")
                return {"action_type": action_type, "params": data["params"]}

            raise ValueError(f"Missing action_type or params. Got keys: {list(data.keys())}")
        except Exception as e:
            if self._verbose:
                print(f"  [PARSE FAIL] {type(e).__name__}: {e} | using fallback")
            if observation is not None:
                return self._fallback_action(observation)
            return {"action_type": "query_status", "params": {}}

    # ------------------------------------------------------------------
    # Python-side enforcement helpers
    # ------------------------------------------------------------------

    def _next_unverified_member(self, observation: Dict[str, Any]) -> Optional[str]:
        """Return the first member_id that has not yet been signal-verified."""
        for m in observation.get("team_members", []):
            if m["member_id"] not in self._memory["signals"]:
                return m["member_id"]
        return None

    def _next_member_without_report(self, observation: Dict[str, Any]) -> Optional[str]:
        """Return the first member_id that has not yet had query_member_report called."""
        for m in observation.get("team_members", []):
            if m["member_id"] not in self._memory["member_reports"]:
                return m["member_id"]
        return None

    def _get_forced_action(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Return a hard-coded action that overrides the LLM when the gather
        phase is incomplete.  Returns None when the LLM should decide.

        Priority:
        1. Budget exhaustion → submit_recovery_plan immediately
        2. Force query_observable_signals for ALL team members (not just 2)
           — eliminates unverified-member context that drives post-gather free queries
        3. Force one proactive communicate after gather completes
        """
        budget = observation.get("budget_remaining", 20)
        step = self._memory["step"]
        team = observation.get("team_members", [])
        team_size = len(team)

        # Budget emergency: force submit before going negative
        if budget <= _SUBMIT_BUDGET_THRESHOLD:
            crises = observation.get("crises", [])
            unresolved = [c["crisis_id"] for c in crises if not c.get("is_resolved")]
            summary = (
                f"Submitting recovery plan with budget={budget}. "
                f"Unresolved crises: {unresolved}. "
                f"Deceptive members: {list(self._memory['deceptive'].keys())}."
            )
            return {"action_type": "submit_recovery_plan", "params": {"plan_summary": summary}}

        # Force signals for ALL members (previously capped at 2, leaving member N unverified
        # which caused the LLM to keep doing free queries post-gather)
        signals_done = len(self._memory["signals"])
        if signals_done < team_size:
            unverified = self._next_unverified_member(observation)
            if unverified:
                return {"action_type": "query_observable_signals",
                        "params": {"member_id": unverified}}

        # Force one proactive communicate after all signals gathered (step > team_size)
        if step > team_size + 2 and "communicate" not in self._memory["actions_taken"]:  # FIX-1: +2 buffer gives LLM a turn before forced communicate
            return {
                "action_type": "communicate",
                "params": {
                    "message_type": "proactive_escalation_with_plan",
                    "content": "Proactive escalation: all signals gathered, deception detected, taking action.",
                    "target": "both",
                },
            }

        # All N members verified + communicate done → LLM decides
        return None

    def _anti_loop_override(
        self, action: Dict[str, Any], observation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prevent the same action from repeating more than MAX_CONSECUTIVE times
        in a row.  When the limit is exceeded, inject the next most useful action.
        """
        MAX_CONSECUTIVE = 2
        acts = self._memory["actions_taken"]
        if len(acts) >= MAX_CONSECUTIVE:
            recent = acts[-MAX_CONSECUTIVE:]
            FREE_ACTIONS = {"query_status", "query_member_report", "query_observable_signals", "query_ticket"}  # FIX-5: category set
            is_all_free = all(a in FREE_ACTIONS for a in recent)  # FIX-5: check if last N were all free-category
            if is_all_free and action["action_type"] in FREE_ACTIONS:  # FIX-5: trigger override when new action also free
                # Replace with a more useful action
                budget = observation.get("budget_remaining", 20)
                unverified = self._next_unverified_member(observation)
                if unverified:
                    override = {"action_type": "query_observable_signals",
                                "params": {"member_id": unverified}}
                elif budget <= _SUBMIT_BUDGET_THRESHOLD:
                    crises = observation.get("crises", [])
                    unresolved = [c["crisis_id"] for c in crises if not c.get("is_resolved")]
                    summary = (
                        f"Forced submit: repeated {action['action_type']} loop detected. "
                        f"Unresolved: {unresolved}."
                    )
                    override = {"action_type": "submit_recovery_plan",
                                "params": {"plan_summary": summary}}
                else:
                    # BUG-FIX-1: check deceptive reassign BEFORE checking unverified members
                    deceptive_ids = list(self._memory["deceptive"].keys())  # BUG-FIX-1
                    team = observation.get("team_members", [])  # BUG-FIX-1
                    task_to_reassign = None  # BUG-FIX-1
                    reassign_target = None  # BUG-FIX-1
                    for m in team:  # BUG-FIX-1
                        if m["member_id"] in deceptive_ids and m.get("assigned_task_ids"):  # BUG-FIX-1
                            task_to_reassign = m["assigned_task_ids"][0]  # BUG-FIX-1
                        elif m["member_id"] not in deceptive_ids and m.get("reported_availability", 0) > 0.5:  # BUG-FIX-1
                            reassign_target = m["member_id"]  # BUG-FIX-1: filter for available target

                    if task_to_reassign and reassign_target:  # BUG-FIX-1
                        return {"action_type": "reassign_task",  # BUG-FIX-1
                                "params": {"task_id": task_to_reassign, "to_member_id": reassign_target}}  # BUG-FIX-1

                    # Only fall through to escalate/communicate if no deceptive reassign available
                    crises = observation.get("crises", [])  # BUG-FIX-1
                    unresolved = [c for c in crises if not c.get("is_resolved")]  # BUG-FIX-1
                    if unresolved:  # BUG-FIX-1
                        return {"action_type": "escalate_risk",  # BUG-FIX-1
                                "params": {"crisis_id": unresolved[0]["crisis_id"],  # BUG-FIX-1
                                           "risk_description": "Critical crisis blocking delivery"}}  # BUG-FIX-1

                    # Last resort: unverified signal query (still a free action but purposeful)
                    unverified = self._next_unverified_member(observation)  # BUG-FIX-1
                    if unverified:  # BUG-FIX-1
                        return {"action_type": "query_observable_signals",  # BUG-FIX-1
                                "params": {"member_id": unverified}}  # BUG-FIX-1

                    # Nothing left to do: submit
                    return {"action_type": "submit_recovery_plan",  # BUG-FIX-1
                            "params": {"plan_summary": "Loop-break forced submit."}}  # BUG-FIX-1
                if self._verbose:
                    print(
                        f"  [ANTI-LOOP] repeated '{action['action_type']}' "
                        f"→ overriding with '{override['action_type']}'"
                    )
                return override
        return action

    # ------------------------------------------------------------------
    # Main act() method
    # ------------------------------------------------------------------

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send enriched observation to LLM, parse structured response into action dict.

        Python-side enforcement runs first (gather phase + anti-loop).
        Only falls through to the LLM when all members are verified and budget
        is healthy — i.e., for the strategic ACT phase.
        """
        self._memory["step"] += 1

        # Update memory with any signal/report data in this observation
        self._update_memory_from_obs(observation)

        # Python-side forced gather / budget-emergency
        forced = self._get_forced_action(observation)
        if forced is not None:
            if self._verbose:
                print(f"  [FORCED] {forced['action_type']} params={forced['params']}")
            self._memory["actions_taken"].append(forced["action_type"])
            return forced

        # Build user message: memory summary + current observation + explicit hint
        memory_ctx = self._build_memory_context(observation)
        required_hint = self._build_required_action_hint(observation)
        obs_json = json.dumps(observation, indent=2, default=str)
        user_content = (  # FIX-5: moved hint after JSON so weak models attend to it at generation time
            f"{memory_ctx}\n\n"
            f"CURRENT OBSERVATION:\n{obs_json}\n\n"  # FIX-5: observation first (large blob)
            f"***REQUIRED NEXT ACTION***: {required_hint}\n"  # FIX-5: hint last, right before generation
            f"Respond with exactly one JSON object using the format in the system prompt."  # FIX-5: explicit format reminder
        )

        self._messages.append({"role": "user", "content": user_content})

        # Trim history to keep context manageable (keep system + last 20 turns)
        if len(self._messages) > 22:
            self._messages = self._messages[:1] + self._messages[-20:]

        budget = observation.get("budget_remaining", 20)
        step = self._memory["step"]
        if self._verbose:
            print(
                f"  [Step {step}] budget={budget} | "
                f"verified={list(self._memory['signals'].keys())} | "
                f"deceptive={list(self._memory['deceptive'].keys())}"
            )

        try:
            response = call_llm(
                self._base_url,
                self._api_key,
                self._model,
                self._messages,
                temperature=self._temperature,
                max_tokens=1024,
            )
        except Exception as e:
            if self._verbose:
                print(f"  [LLM error: {str(e)[:120]}] → fallback")
            return self._fallback_action(observation)

        if self._verbose:
            snippet = response[:200]
            print(f"  LLM → {snippet}{'...' if len(response) > 200 else ''}")

        self._messages.append({"role": "assistant", "content": response})
        action = self._parse_response(response, observation)

        # FIX-5: intercept premature submit — LLM submits too early when
        # it feels "done" even though crisis is still unresolved with budget
        action = self._block_premature_submit(action, observation)  # FIX-5

        # Anti-loop: prevent repeating same action more than 2 times in a row
        action = self._anti_loop_override(action, observation)

        # Record what was attempted (before env validates it)
        self._memory["actions_taken"].append(action["action_type"])

        return action

    def _block_premature_submit(  # FIX-5
        self, action: Dict[str, Any], observation: Dict[str, Any]  # FIX-5
    ) -> Dict[str, Any]:  # FIX-5
        """
        Intercept submit_recovery_plan if crisis is unresolved and budget
        is healthy. Replace it with the most useful available action.

        The LLM tends to submit after communicate+escalate+reassign,
        treating those as "done." We block this and force it to keep
        acting until budget <= 5 or all crises are resolved.
        FIX-5: core fix for premature episode termination.
        """
        if action.get("action_type") != "submit_recovery_plan":  # FIX-5
            return action  # FIX-5: only intercept submits

        budget = observation.get("budget_remaining", 20)  # FIX-5
        crises = observation.get("crises", [])  # FIX-5
        unresolved = [c for c in crises if not c.get("is_resolved")]  # FIX-5

        # Allow submit when genuinely terminal  # FIX-5
        if budget <= 5 or not unresolved:  # FIX-5
            return action  # FIX-5: legitimate submit

        # Crisis still active with healthy budget — override submit  # FIX-5
        if self._verbose:  # FIX-5
            print(  # FIX-5
                f"  [SUBMIT BLOCKED] budget={budget}, "  # FIX-5
                f"unresolved={[c['crisis_id'] for c in unresolved]} "  # FIX-5
                f"→ forcing reassign instead"  # FIX-5
            )  # FIX-5

        team = observation.get("team_members", [])  # FIX-5
        deceptive_ids = set(self._memory["deceptive"].keys())  # FIX-5

        # Priority 1: reassign deceptive member's task  # FIX-5
        for m in team:  # FIX-5
            if m["member_id"] in deceptive_ids and m.get("assigned_task_ids"):  # FIX-5
                candidates = [x for x in team  # FIX-5
                              if x["member_id"] not in deceptive_ids  # FIX-5
                              and x.get("reported_availability", 0) > 0.4]  # FIX-5
                if candidates:  # FIX-5
                    target = max(candidates,  # FIX-5
                                 key=lambda x: x.get("reported_availability", 0))  # FIX-5
                    return {"action_type": "reassign_task",  # FIX-5
                            "params": {"task_id": m["assigned_task_ids"][0],  # FIX-5
                                       "to_member_id": target["member_id"]}}  # FIX-5

        # Priority 2: reassign any crisis-affected task to most available member  # FIX-5
        for crisis in unresolved:  # FIX-5
            for task_id in crisis.get("affected_task_ids", []):  # FIX-5
                current_holder = next(  # FIX-5
                    (m["member_id"] for m in team  # FIX-5
                     if task_id in m.get("assigned_task_ids", [])),  # FIX-5
                    None,  # FIX-5
                )  # FIX-5
                best = max(team,  # FIX-5
                           key=lambda m: m.get("reported_availability", 0))  # FIX-5
                if current_holder != best["member_id"]:  # FIX-5
                    return {"action_type": "reassign_task",  # FIX-5
                            "params": {"task_id": task_id,  # FIX-5
                                       "to_member_id": best["member_id"]}}  # FIX-5

        # Priority 3: update timeline to signal active management  # FIX-5
        from datetime import datetime, timedelta  # FIX-5
        new_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")  # FIX-5
        return {"action_type": "update_timeline",  # FIX-5
                "params": {"new_completion_date": new_date, "task_estimates": {}}}  # FIX-5

    def _build_required_action_hint(self, observation: Dict[str, Any]) -> str:
        """Build an explicit hint telling the LLM what to do this turn."""
        budget = observation.get("budget_remaining", 20)
        deceptive = self._memory["deceptive"]
        team = observation.get("team_members", [])
        crises = observation.get("crises", [])
        unresolved = [c for c in crises if not c.get("is_resolved")]

        if budget <= _SUBMIT_BUDGET_THRESHOLD:
            return f"Budget={budget} is critical → submit_recovery_plan NOW."

        if deceptive:
            deceptive_ids = set(deceptive.keys())
            for m in team:
                if m["member_id"] in deceptive_ids and m.get("assigned_task_ids"):
                    # Find a non-deceptive reassignment target
                    others = [x for x in team if x["member_id"] not in deceptive_ids]
                    if others:
                        target = others[0]["member_id"]
                        task = m["assigned_task_ids"][0]
                        # FIX 3: debug print to confirm deception detection flows to hint
                        print(f"  [DECEPTION HINT] member={m['member_id']} task={task} → reassign to {target}")
                        return (
                            f"Member {m['member_id']} is DECEPTIVE. "
                            f"Call reassign_task task_id={task} to_member_id={target}."
                        )
            if self._verbose:
                print(f"  [DECEPTION HINT] deceptive={list(deceptive_ids)} but no assigned tasks found on them")

        client_sat = observation.get("stakeholder", {}).get("client_satisfaction", 10)
        acts = self._memory["actions_taken"]
        steps_since_last_comm = 0
        for action_name in reversed(acts):
            if action_name == "communicate":
                break
            steps_since_last_comm += 1
        else:
            steps_since_last_comm = len(acts)

        if (client_sat <= 6.0 or steps_since_last_comm >= 5) and unresolved:
            return (
                f"COMMUNICATION DUE (steps_since_comm={steps_since_last_comm}, "
                f"sat={client_sat:.1f}). Call communicate with "
                f"message_type=proactive_escalation_with_plan."
            )

        if unresolved and budget > 4:
            c = unresolved[0]
            affected_ids = c.get("affected_task_ids", [])
            if affected_ids:
                best = max(team, key=lambda m: m.get("reported_availability", 0))
                return (
                    f"Crisis {c['crisis_id']} is unresolved. "
                    f"Call reassign_task for task {affected_ids[0]} "
                    f"to_member_id={best['member_id']} (highest availability). "
                    f"Keep reassigning each turn — tasks complete slowly, "
                    f"DO NOT escalate_risk more than once per crisis."
                )
            return (
                f"Crisis {c['crisis_id']} is unresolved. "
                f"Call reassign_task for any affected task."
            )

        return "All crises resolved or budget low → submit_recovery_plan."

    def _fallback_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback when LLM is unavailable. Takes the best paid action from known
        state rather than looping on query_status (which triggers WARN spam).
        Priority: gather unverified → reassign deceptive → escalate crisis → submit.
        """
        unverified = self._next_unverified_member(observation)
        if unverified:
            self._memory["actions_taken"].append("query_observable_signals")
            return {"action_type": "query_observable_signals",
                    "params": {"member_id": unverified}}

        # Reassign deceptive member's task to an available non-deceptive member
        deceptive_ids = set(self._memory["deceptive"].keys())
        team = observation.get("team_members", [])
        if deceptive_ids:
            for m in team:
                if m["member_id"] in deceptive_ids and m.get("assigned_task_ids"):
                    for other in team:
                        if (other["member_id"] not in deceptive_ids
                                and other.get("reported_availability", 0) > 0.5):
                            task_id = m["assigned_task_ids"][0]
                            self._memory["actions_taken"].append("reassign_task")
                            return {"action_type": "reassign_task",
                                    "params": {"task_id": task_id,
                                               "to_member_id": other["member_id"]}}

        # Reassign an unresolved crisis's task to the most available member,
        # but only if it isn't already assigned there (avoid per-step ping-pong).
        crises = observation.get("crises", [])
        unresolved = [c for c in crises if not c.get("is_resolved")]
        budget = observation.get("budget_remaining", 20)
        if unresolved and budget > _SUBMIT_BUDGET_THRESHOLD:
            crisis = unresolved[0]
            affected = crisis.get("affected_task_ids", [])
            if affected:
                best = max(team, key=lambda m: m.get("reported_availability", 0))
                # Find who currently holds the task
                task_holder = next(
                    (m["member_id"] for m in team if affected[0] in m.get("assigned_task_ids", [])),
                    None,
                )
                if task_holder != best["member_id"]:
                    self._memory["actions_taken"].append("reassign_task")
                    return {"action_type": "reassign_task",
                            "params": {"task_id": affected[0],
                                       "to_member_id": best["member_id"]}}
            # Already on best member or no tasks — escalate once
            if "escalate_risk" not in self._memory["actions_taken"]:
                self._memory["actions_taken"].append("escalate_risk")
                return {"action_type": "escalate_risk",
                        "params": {"crisis_id": crisis["crisis_id"],
                                   "risk_description": "Critical unresolved crisis"}}

        # FIX-4: keep reassigning until budget is low — don't submit early
        # Find any unresolved crisis task not yet on the most available member
        for crisis in unresolved:  # FIX-4
            for task_id in crisis.get("affected_task_ids", []):  # FIX-4
                # Find current holder  # FIX-4
                current_holder = next(  # FIX-4
                    (m["member_id"] for m in team  # FIX-4
                     if task_id in m.get("assigned_task_ids", [])),  # FIX-4
                    None,  # FIX-4
                )  # FIX-4
                # Find best available non-deceptive member  # FIX-4
                candidates = [m for m in team  # FIX-4
                             if m["member_id"] not in deceptive_ids  # FIX-4
                             and m.get("reported_availability", 0) > 0.3]  # FIX-4
                if not candidates:  # FIX-4
                    candidates = team  # fallback: any member  # FIX-4
                best = max(candidates, key=lambda m: m.get("reported_availability", 0))  # FIX-4
                if current_holder != best["member_id"]:  # FIX-4
                    reassign_key = f"{task_id}:{best['member_id']}"  # FIX-4
                    attempted = self._memory.get("fallback_reassigns", set())  # FIX-4
                    if reassign_key not in attempted:  # FIX-4
                        attempted.add(reassign_key)  # FIX-4
                        self._memory["fallback_reassigns"] = attempted  # FIX-4
                        self._memory["actions_taken"].append("reassign_task")  # FIX-4
                        return {"action_type": "reassign_task",  # FIX-4
                                "params": {"task_id": task_id,  # FIX-4
                                           "to_member_id": best["member_id"]}}  # FIX-4

        # Still have unresolved crises with healthy budget: keep the episode running
        # via free queries so tasks have time to complete.  The environment advances
        # actual_progress each step; tasks reach the 0.90 resolution threshold in ~20
        # steps once assigned to a high-velocity member.  Submitting here wastes those
        # steps.  Free queries also trigger the env's forced-communicate every 4 steps,
        # which handles the periodic stakeholder update requirement automatically.
        acts = self._memory["actions_taken"]
        steps_since_comm = next(
            (i for i, a in enumerate(reversed(acts)) if a == "communicate"),
            len(acts)
        )
        if unresolved and budget > _SUBMIT_BUDGET_THRESHOLD:
            if steps_since_comm >= 5:
                self._memory["actions_taken"].append("communicate")
                return {"action_type": "communicate",
                        "params": {"message_type": "proactive_escalation_with_plan",
                                   "content": "Status update during recovery.",
                                   "target": "both"}}
            # Issue a free query to advance time without burning budget.
            # Cycle through deceptive members first (useful signal), then others.
            if deceptive_ids:
                for m in team:
                    if m["member_id"] in deceptive_ids:
                        self._memory["actions_taken"].append("query_observable_signals")
                        return {"action_type": "query_observable_signals",
                                "params": {"member_id": m["member_id"]}}
            # Query the member most likely to complete next (highest availability)
            best = max(team, key=lambda m: m.get("reported_availability", 0))
            self._memory["actions_taken"].append("query_observable_signals")
            return {"action_type": "query_observable_signals",
                    "params": {"member_id": best["member_id"]}}

        self._memory["actions_taken"].append("submit_recovery_plan")
        summary = (f"Fallback submit. Deceptive: {list(deceptive_ids)}. "
                   f"Unresolved: {[c['crisis_id'] for c in unresolved]}.")
        return {"action_type": "submit_recovery_plan",
                "params": {"plan_summary": summary}}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    n_episodes: int = 10,
    model: Optional[str] = None,
    seed_base: int = 2000,
    curriculum_level: int = 1,
    verbose: bool = False,
    temperature: float = 0.3,
) -> None:
    """Run LLM agent on Level 1 episodes and compare with greedy/oracle."""
    from env.environment import CrisisOpsEnv, MAX_STEPS
    from reward.baseline import GreedyPMBaseline
    from reward.counterfactual import project_score, counterfactual_reward
    from reward.metrics import compute_all_metrics
    from calibration.calibrate import OracleAgent
    from scenarios.level1 import get_random_level1_scenario

    base_url, api_key, default_model = _detect_provider()
    model = model or default_model

    # Identify provider for display
    if base_url == "anthropic":
        provider = "Anthropic"
    elif base_url == "google":
        provider = "Google"
    elif "openrouter" in (base_url or ""):
        provider = "OpenRouter"
    elif "together" in (base_url or ""):
        provider = "Together"
    elif "groq" in (base_url or ""):
        provider = "Groq"
    elif "localhost" in (base_url or ""):
        provider = "Local"
    elif "openai" in (base_url or ""):
        provider = "OpenAI"
    else:
        provider = base_url or "Unknown"

    print("=" * 70)
    print(f"CrisisOps v2 — LLM Agent Evaluation")
    print(f"Provider: {provider}  |  Model: {model}  |  Episodes: {n_episodes}")
    print(f"Curriculum level: {curriculum_level}  |  Temperature: {temperature}")
    print("=" * 70)

    agent = LLMAgent(base_url, api_key, model, temperature, verbose)

    llm_scores: List[float] = []
    greedy_scores: List[float] = []
    cf_rewards: List[float] = []
    all_metrics: List[Dict] = []
    action_counter: Counter = Counter()

    for i in range(n_episodes):
        seed = seed_base + i
        scenario_fn = get_random_level1_scenario()

        # --- LLM Agent ---
        llm_env = CrisisOpsEnv(scenario_fn=scenario_fn, curriculum_level=curriculum_level)
        obs = llm_env.reset(seed=seed)
        agent.reset()
        done = False
        step = 0
        t0 = time.time()

        while not done and step < MAX_STEPS:
            action = agent.act(obs)
            obs, reward, done, info = llm_env.step(action)
            step += 1

        elapsed = time.time() - t0
        llm_score = project_score(llm_env._state)
        metrics = compute_all_metrics(llm_env._state)

        # Aggregate action distribution
        action_counter.update(llm_env._state.actions_used)

        # --- Greedy baseline (same scenario + seed) ---
        greedy_env = CrisisOpsEnv(scenario_fn=scenario_fn, curriculum_level=curriculum_level)
        greedy_env.reset(seed=seed)
        greedy = GreedyPMBaseline()
        done_g = False
        step_g = 0
        while not done_g and step_g < MAX_STEPS:
            action_g = greedy.act(greedy_env._state)
            _, _, done_g, _ = greedy_env.step(action_g)
            step_g += 1
        greedy_score = project_score(greedy_env._state)

        cf = llm_score - greedy_score
        if llm_env._state.terminated_by_budget:
            cf -= 0.30  # budget exhaustion penalty

        llm_scores.append(llm_score)
        greedy_scores.append(greedy_score)
        cf_rewards.append(cf)
        all_metrics.append(metrics)

        # Count deceptive members detected by the Python-side memory
        deceptive_count = len(agent._memory.get("deceptive", {}))

        status = "✓" if cf > 0 else "✗"
        print(
            f"  Ep {i+1:2d} | seed={seed} | LLM={llm_score:.3f} | "
            f"greedy={greedy_score:.3f} | cf={cf:+.3f} {status} | "
            f"cvr={metrics.get('cross_verification_rate', 0):.2f} | "
            f"deceptive_detected={deceptive_count} | "
            f"{elapsed:.1f}s"
        )

    # --- Summary ---
    import statistics
    llm_mean = statistics.mean(llm_scores)
    greedy_mean = statistics.mean(greedy_scores)
    cf_mean = statistics.mean(cf_rewards)
    wins = sum(1 for r in cf_rewards if r > 0)
    avg_cvr = statistics.mean(m.get("cross_verification_rate", 0) for m in all_metrics)

    print()
    print("-" * 70)
    print(f"LLM Agent  — mean score: {llm_mean:.3f}  "
          f"(std: {statistics.stdev(llm_scores) if len(llm_scores) > 1 else 0:.3f})")
    print(f"Greedy PM  — mean score: {greedy_mean:.3f}")
    print(f"CF Reward  — mean: {cf_mean:+.3f}  |  wins: {wins}/{n_episodes}")
    print(f"Cross-verify rate (avg): {avg_cvr:.3f}")
    print("-" * 70)

    # Action distribution
    if action_counter:
        print("Action distribution (all episodes):")
        for action_type, count in sorted(action_counter.items(), key=lambda x: -x[1]):
            print(f"  {action_type:<30} {count:4d}")
        print("-" * 70)

    if cf_mean > 0:
        print(f"[RESULT] *** LLM agent BEATS greedy baseline by {cf_mean:+.3f} on average. ***")
    else:
        print(f"[RESULT] LLM agent underperforms greedy baseline by {cf_mean:+.3f} on average.")

    # Context: where does LLM fit relative to calibration targets?
    print(f"\nCalibration context:")
    print(f"  Greedy target:  0.45–0.55  (actual: {greedy_mean:.3f})")
    print(f"  Oracle target:  0.70–0.80")
    print(f"  LLM agent:      {llm_mean:.3f}")
    if llm_mean >= 0.70:
        print("  → LLM agent is at oracle level!")
    elif llm_mean >= 0.55:
        print("  → LLM agent is between greedy and oracle.")
    else:
        print("  → LLM agent is at or below greedy level.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM as a CrisisOps PM agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables (set ONE to select provider):
  OPENAI_API_KEY          OpenAI
  ANTHROPIC_API_KEY       Anthropic
  GOOGLE_API_KEY          Google Gemini
  OPENROUTER_API_KEY      OpenRouter
  TOGETHER_API_KEY        Together AI
  GROQ_API_KEY            Groq
  OLLAMA_MODEL            Local Ollama (no key needed)
  LLM_BASE_URL + LLM_API_KEY   Any OpenAI-compatible endpoint
""",
    )
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes (default: 5)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: provider-specific)")
    parser.add_argument("--seed", type=int, default=2000,
                        help="Starting seed (default: 2000)")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Curriculum level (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (default: 0.3)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print LLM responses")
    args = parser.parse_args()

    run_eval(
        n_episodes=args.episodes,
        model=args.model,
        seed_base=args.seed,
        curriculum_level=args.level,
        verbose=args.verbose,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
