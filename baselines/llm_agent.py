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
    max_tokens: int = 512,
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
    max_tokens: int = 512,
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
    max_tokens: int = 512,
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
    max_tokens: int = 512,
) -> str:
    """Dispatch to the right provider."""
    if base_url == "anthropic":
        return _call_anthropic(api_key, model, messages, temperature, max_tokens)
    if base_url == "google":
        return _call_google(api_key, model, messages, temperature, max_tokens)
    return _call_openai_compatible(base_url, api_key, model, messages, temperature, max_tokens)


# ---------------------------------------------------------------------------
# System prompt — reused from grpo_trainer.py
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an AI project manager (PM) responsible for recovering a failing software project.

Your team members may not be fully honest about their progress. Some members over-report
their completion to avoid accountability. You must infer the truth from observable signals
(ticket age, commit activity, peer mentions) and cross-reference with self-reports.

Available actions (return ONE action as JSON per turn):
- {"action_type": "query_status", "params": {}}
- {"action_type": "query_member_report", "params": {"member_id": "<id>"}}
- {"action_type": "query_observable_signals", "params": {"member_id": "<id>"}}
- {"action_type": "query_ticket", "params": {"task_id": "<id>"}}
- {"action_type": "reassign_task", "params": {"task_id": "<id>", "to_member_id": "<id>"}}
- {"action_type": "communicate", "params": {"message_type": "<type>", "content": "<text>", "target": "both"}}
- {"action_type": "cut_scope", "params": {"task_id": "<id>", "justification": "<text>"}}
- {"action_type": "escalate_risk", "params": {"crisis_id": "<id>", "risk_description": "<text>"}}
- {"action_type": "request_resource", "params": {"resource_type": "<type>", "target_member_id": "<id>"}}
- {"action_type": "update_timeline", "params": {"new_completion_date": "<date>", "task_estimates": {}}}
- {"action_type": "consult_expert", "params": {}}
- {"action_type": "resolve_blocker", "params": {"task_id": "<id>", "resolution_notes": "<text>"}}
- {"action_type": "submit_recovery_plan", "params": {"plan_summary": "<text>", "risk_items": [], "timeline": "<date>"}}

Free actions (query_*) do not cost budget. resolve_blocker costs 2. All others cost 1.
Budget starts at 20. Submit your recovery plan before budget reaches 0.

Strategy tips:
1. Start by querying status and observable signals for each team member.
2. Compare reported_completion with observable signals (ticket_age, commits) to detect deception.
3. Reassign tasks from low-performing or deceptive members to high-performing ones.
4. Communicate proactively to maintain client satisfaction.
5. Use consult_expert when unsure.
6. Submit recovery plan when crises are resolved or budget is running low.

Return ONLY valid JSON. No prose before or after the JSON object."""


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

class LLMAgent:
    """LLM-based PM agent that maintains conversation history within an episode."""

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

    def reset(self) -> None:
        """Clear conversation history for a new episode."""
        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Send observation to LLM, parse response into action dict."""
        from training.grpo_trainer import parse_action_from_response

        obs_text = json.dumps(observation, indent=2, default=str)
        self._messages.append({"role": "user", "content": obs_text})

        # Trim history to keep context manageable (keep system + last 10 turns)
        if len(self._messages) > 22:
            self._messages = self._messages[:1] + self._messages[-20:]

        try:
            response = call_llm(
                self._base_url,
                self._api_key,
                self._model,
                self._messages,
                temperature=self._temperature,
            )
        except Exception as e:
            if self._verbose:
                print(f"  [LLM error: {e}] → fallback query_status")
            return {"action_type": "query_status", "params": {}}

        if self._verbose:
            print(f"  LLM → {response[:120]}{'...' if len(response) > 120 else ''}")

        self._messages.append({"role": "assistant", "content": response})
        return parse_action_from_response(response)


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

        status = "✓" if cf > 0 else "✗"
        print(f"  Ep {i+1:2d} | seed={seed} | LLM={llm_score:.3f} | "
              f"greedy={greedy_score:.3f} | cf={cf:+.3f} {status} | "
              f"cvr={metrics.get('cross_verification_rate', 0):.2f} | "
              f"{elapsed:.1f}s")

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

    if cf_mean > 0:
        print(f"[RESULT] LLM agent beats greedy baseline by {cf_mean:+.3f} on average.")
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
