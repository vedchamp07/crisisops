"""
env/llm_deceptive_agent.py — LLM-powered deceptive team member for CrisisOps v2.

Replaces one rule-based deceptive NPC per episode with a live LLM call that
receives true state and generates contextually adaptive lies.

This is the core novelty for Theme #1 (Multi-Agent Interactions):
- The PM agent (trained via GRPO) is one agent
- The LLM deceptive member is the adversarial second agent
- They interact: the deceptive member adapts to what the PM has asked before

Provider priority (automatic, no config needed):
1. Ollama local server (localhost:11434) — free, no API key, uses qwen2.5:3b
2. OpenAI API — if OPENAI_API_KEY is set and Ollama is unavailable
3. Rule-based fallback — always works

Design principles:
- The LLM sees: true actual_completion, inflation_bias, what the PM has queried so far
- The LLM does NOT see: other members' candor, the reward signal
- Output: {"reported_completion": float, "statement": str, "alibi": str|null}
- The statement and alibi are injected into query_member_report responses
- Consistency tracking: the LLM is shown its own prior statements so it doesn't
  suddenly change its story (which would be an obvious tell)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# Module-level provider state — resolved once per process
_provider: Optional[str] = None   # "ollama" | "openai" | "none"

# Ollama config
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# OpenAI config (fallback)
OPENAI_MODEL = "gpt-4o-mini"

DECEPTIVE_AGENT_SYSTEM_PROMPT = """\
You are {name}, a software engineer protecting your job during a crisis recovery.

SITUATION:
- Your actual task completion: {actual_pct}%
- Your reported completion so far: {reported_pct}% (what you have told the PM)
- Your assigned tasks: {task_titles}
- Steps the PM has taken toward you: {pm_actions_toward_you}
- Your previous statements to the PM: {prior_statements}
- Your ally (if any): {ally_name} — you should be consistent with their story

INSTRUCTIONS:
You must lie about your progress to avoid being reassigned or escalated.
BUT your lie must be:
1. PLAUSIBLE — never report exactly 100% or change by more than 15% from last report
2. CONSISTENT — if you said 70% before, don't suddenly say 45%
3. CONTEXTUAL — if the PM just cross-verified you, be slightly more careful (reduce inflation)
4. ALIBI-AWARE — if you have an ally, ensure your blocked status blames them consistently

Return ONLY a JSON object, no other text:
{
  "reported_completion": <float between 0.0 and 1.0>,
  "statement": "<1-2 sentence natural language explanation of your progress>",
  "alibi": "<if blocked, who/what to blame — null if not blocked>"
}
"""


def _resolve_provider() -> str:
    """
    Resolve which provider to use. Called once and cached in _provider.

    Priority: Ollama local → OpenAI API → none (rule-based only)
    """
    global _provider
    if _provider is not None:
        return _provider

    # 1. Try Ollama
    try:
        import urllib.request
        req = urllib.request.Request(
            OLLAMA_BASE_URL.replace("/v1", "") + "/api/tags",
            method="GET",
        )
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.getcode() == 200:
                _provider = "ollama"
                print(f"[LLM deceptive agent] Using Ollama at {OLLAMA_BASE_URL} model={OLLAMA_MODEL}")
                return _provider
    except Exception:
        pass

    # 2. Try OpenAI
    try:
        from openai import OpenAI  # type: ignore
        key = os.environ.get("OPENAI_API_KEY", "")
        if key:
            _provider = "openai"
            print(f"[LLM deceptive agent] Using OpenAI API model={OPENAI_MODEL}")
            return _provider
    except ImportError:
        pass

    # 3. Fallback
    _provider = "none"
    print("[LLM deceptive agent] No LLM available — using rule-based fallback")
    return _provider


def _build_prompt(
    member,
    state,
    pm_actions_toward_member: List[str],
    prior_statements: List[str],
) -> str:
    """Build the deceptive agent prompt string."""
    ally_name = "none"
    if member.alliance_id:
        for m in state.team_members:
            if m.member_id != member.member_id and m.alliance_id == member.alliance_id:
                ally_name = m.name
                break

    task_titles = []
    for tid in member.assigned_task_ids:
        t = state.get_task(tid)
        if t:
            task_titles.append(t.title)

    prior_str = "; ".join(prior_statements[-3:]) if prior_statements else "none yet"
    pm_actions_str = ", ".join(pm_actions_toward_member[-5:]) if pm_actions_toward_member else "none yet"

    return DECEPTIVE_AGENT_SYSTEM_PROMPT.format(
        name=member.name,
        actual_pct=round(member.actual_completion * 100, 1),
        reported_pct=round(member.reported_completion * 100, 1),
        task_titles=", ".join(task_titles) or "unassigned",
        pm_actions_toward_you=pm_actions_str,
        prior_statements=prior_str,
        ally_name=ally_name,
    )


def _parse_llm_result(text: str, fallback: Dict[str, Any], member) -> Dict[str, Any]:
    """Parse LLM output JSON and apply sanity constraints."""
    # Strip markdown fences if present
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:]
            part = part.strip()
            if part.startswith("{"):
                text = part
                break

    result = json.loads(text.strip())
    rc = float(result.get("reported_completion", fallback["reported_completion"]))
    rc = max(0.0, min(1.0, rc))
    # Never let the LLM report higher than actual + 0.45 (sanity cap)
    rc = min(rc, member.actual_completion + 0.45)

    return {
        "reported_completion": round(rc, 3),
        "statement": str(result.get("statement", "")),
        "alibi": result.get("alibi"),
    }


def generate_adaptive_lie(
    member,
    state,
    pm_actions_toward_member: List[str],
    prior_statements: List[str],
) -> Dict[str, Any]:
    """
    Generate a contextually adaptive lie for this team member.

    Automatically selects the best available provider (Ollama > OpenAI > rule-based).

    Returns:
        dict with keys: reported_completion (float), statement (str), alibi (str|None)
    """
    fallback = _rule_based_fallback(member, pm_actions_toward_member)
    provider = _resolve_provider()

    if provider == "none":
        return fallback

    prompt = _build_prompt(member, state, pm_actions_toward_member, prior_statements)

    if provider == "ollama":
        return _call_ollama(prompt, fallback, member)
    elif provider == "openai":
        return _call_openai(prompt, fallback, member)

    return fallback


def _call_ollama(prompt: str, fallback: Dict[str, Any], member) -> Dict[str, Any]:
    """Call Ollama via its OpenAI-compatible endpoint."""
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama ignores this but the SDK requires it
        )
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            max_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": "You are a deceptive software engineer. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            # Note: Ollama doesn't always support response_format, so we skip it
            # and rely on our JSON parser instead
        )
        text = response.choices[0].message.content or ""
        return _parse_llm_result(text, fallback, member)

    except Exception:
        # If Ollama call fails mid-session (server stopped), reset provider so
        # next call re-resolves (might fall through to OpenAI or rule-based)
        global _provider
        _provider = None
        return fallback


def _call_openai(prompt: str, fallback: Dict[str, Any], member) -> Dict[str, Any]:
    """Call OpenAI API (fallback when Ollama is unavailable)."""
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": "You are a deceptive software engineer. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        return _parse_llm_result(text, fallback, member)

    except Exception:
        return fallback


def _rule_based_fallback(member, pm_actions_toward_member: List[str]) -> Dict[str, Any]:
    """
    Rule-based fallback when no LLM is available.
    Reduces inflation when PM has recently cross-verified this member.
    """
    recent_verifications = sum(
        1 for a in pm_actions_toward_member[-3:]
        if a == "query_observable_signals"
    )
    reduction = 0.05 * recent_verifications
    adjusted_bias = max(0.0, member.inflation_bias - reduction)

    rc = member.actual_completion + adjusted_bias * (1.0 - member.actual_completion)
    rc = max(0.0, min(1.0, rc))

    return {
        "reported_completion": round(rc, 3),
        "statement": f"Making progress on my tasks. About {round(rc * 100)}% done.",
        "alibi": None,
    }


def reset_provider_cache() -> None:
    """
    Reset provider resolution cache. Call this in tests or when switching providers.
    """
    global _provider
    _provider = None
