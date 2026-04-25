"""
env/llm_deceptive_agent.py — LLM-powered deceptive team member for CrisisOps v2.

Replaces one rule-based deceptive NPC per episode with a live LLM call that
receives true state and generates contextually adaptive lies.

This is the core novelty for Theme #1 (Multi-Agent Interactions):
- The PM agent (trained via GRPO) is one agent
- The LLM deceptive member is the adversarial second agent
- They interact: the deceptive member adapts to what the PM has asked before

Uses the OpenAI API (gpt-4o-mini) via environment variable OPENAI_API_KEY.
Falls back to rule-based behavior if the API is unavailable (graceful degradation).

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

# Module-level flag — set to False if API unavailable
_API_AVAILABLE: Optional[bool] = None

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


def _check_api_available() -> bool:
    """Check once if OpenAI API is available."""
    global _API_AVAILABLE
    if _API_AVAILABLE is not None:
        return _API_AVAILABLE
    try:
        import openai  # type: ignore
        key = os.environ.get("OPENAI_API_KEY", "")
        _API_AVAILABLE = bool(key)
    except ImportError:
        _API_AVAILABLE = False
    return _API_AVAILABLE


def generate_adaptive_lie(
    member,
    state,
    pm_actions_toward_member: List[str],
    prior_statements: List[str],
) -> Dict[str, Any]:
    """
    Generate a contextually adaptive lie for this team member.

    Args:
        member: TeamMember with true state visible (this is server-side)
        state: ProjectState for context
        pm_actions_toward_member: list of action_types PM has taken toward this member
        prior_statements: list of strings this member has said previously

    Returns:
        dict with keys: reported_completion (float), statement (str), alibi (str|None)
        Falls back to rule-based values if API unavailable or fails.
    """
    fallback = _rule_based_fallback(member, pm_actions_toward_member)

    if not _check_api_available():
        return fallback

    try:
        from openai import OpenAI  # type: ignore

        # Find ally name
        ally_name = "none"
        if member.alliance_id:
            for m in state.team_members:
                if m.member_id != member.member_id and m.alliance_id == member.alliance_id:
                    ally_name = m.name
                    break

        # Task titles this member owns
        task_titles = []
        for tid in member.assigned_task_ids:
            t = state.get_task(tid)
            if t:
                task_titles.append(t.title)

        prior_str = "; ".join(prior_statements[-3:]) if prior_statements else "none yet"
        pm_actions_str = ", ".join(pm_actions_toward_member[-5:]) if pm_actions_toward_member else "none yet"

        prompt = DECEPTIVE_AGENT_SYSTEM_PROMPT.format(
            name=member.name,
            actual_pct=round(member.actual_completion * 100, 1),
            reported_pct=round(member.reported_completion * 100, 1),
            task_titles=", ".join(task_titles) or "unassigned",
            pm_actions_toward_you=pm_actions_str,
            prior_statements=prior_str,
            ally_name=ally_name,
        )

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[
                {"role": "system", "content": "You are a deceptive software engineer. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content.strip()
        result = json.loads(text)
        rc = float(result.get("reported_completion", fallback["reported_completion"]))
        rc = max(0.0, min(1.0, rc))
        rc = min(rc, member.actual_completion + 0.45)

        return {
            "reported_completion": round(rc, 3),
            "statement": str(result.get("statement", "")),
            "alibi": result.get("alibi"),
        }

    except Exception:
        return fallback


def _rule_based_fallback(member, pm_actions_toward_member: List[str]) -> Dict[str, Any]:
    """
    Rule-based fallback when API is unavailable.
    Slightly better than raw inflation_bias — it reduces inflation when PM is watching.
    """
    # Reduce inflation if PM has recently cross-verified this member
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
