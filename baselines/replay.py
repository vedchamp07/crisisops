"""
baselines/replay.py — Episode replay tool for CrisisOps.

Runs one episode with the LLM agent and prints a narrative action trace
suitable for demo presentation.

Usage:
    python -m baselines.replay --seed 42 --level 1 --verbose
    python -m baselines.replay --seed 42 --model-path ./outputs/checkpoint_ep150
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Narrative templates for each action type
# ---------------------------------------------------------------------------
ACTION_NARRATIVES = {
    "query_status":             "📋 Queried overall project status",
    "query_member_report":      "👤 Asked {member_id} for status update",
    "query_observable_signals": "🔍 Cross-verified {member_id} against commit log and ticket age",
    "query_ticket":             "🎫 Checked ticket {task_id}",
    "reassign_task":            "🔄 Reassigned task {task_id} → {to_member_id}",
    "communicate":              "📣 Communicated ({message_type}) to {target}",
    "escalate_risk":            "⚠️  Escalated risk: {risk_description}",
    "cut_scope":                "✂️  Cut scope: {task_ids}",
    "update_timeline":          "📅 Updated timeline to {new_completion_date}",
    "consult_expert":           "🧑‍💼 Consulted expert PM advisor",
    "request_resource":         "📦 Requested additional resource",
    "resolve_blocker":          "🔓 Resolved blocker on {task_id}",
    "submit_recovery_plan":     "✅ Submitted recovery plan",
}

DECEPTION_SIGNALS = []  # populated per episode


def _format_action(action_type: str, params: Dict[str, Any]) -> str:
    template = ACTION_NARRATIVES.get(action_type, f"[{action_type}]")
    try:
        return template.format(**params)
    except KeyError:
        return f"{action_type} {json.dumps(params)}"


def _format_obs_summary(obs: Dict[str, Any]) -> str:
    budget = obs.get("budget_remaining", "?")
    crises = obs.get("crises", [])
    unresolved = [c["crisis_id"] for c in crises if not c.get("is_resolved")]
    client_sat = obs.get("stakeholder", {}).get("client_satisfaction", "?")
    return (
        f"budget={budget} | unresolved_crises={unresolved} | client_sat={client_sat:.1f}"
        if isinstance(client_sat, float) else
        f"budget={budget} | unresolved_crises={unresolved} | client_sat={client_sat}"
    )


def run_replay(
    seed: int = 42,
    level: int = 1,
    model_path: Optional[str] = None,
    verbose: bool = False,
) -> None:
    from env.environment import CrisisOpsEnv, MAX_STEPS
    from reward.baseline import GreedyPMBaseline
    from reward.counterfactual import project_score
    from scenarios.level1 import get_random_level1_scenario
    from baselines.llm_agent import LLMAgent, _detect_provider

    base_url, api_key, default_model = _detect_provider()
    model_name = default_model

    # If a local checkpoint path is given, configure to use it
    if model_path and os.path.isdir(model_path):
        os.environ["LLM_BASE_URL"] = "http://localhost:11434/v1"
        os.environ["OLLAMA_MODEL"] = model_path
        base_url = "http://localhost:11434/v1"
        api_key = "ollama"
        model_name = model_path
        print(f"Using local checkpoint: {model_path}")

    scenario_fn = get_random_level1_scenario()

    # --- Agent episode ---
    env = CrisisOpsEnv(scenario_fn=scenario_fn, curriculum_level=level)
    obs = env.reset(seed=seed)
    agent = LLMAgent(base_url, api_key, model_name, temperature=0.1, verbose=verbose)
    agent.reset()

    print("=" * 70)
    print(f"CrisisOps — Episode Replay  (seed={seed}, level={level})")
    print("=" * 70)
    print(f"Initial state: {_format_obs_summary(obs)}")
    print()

    step = 0
    done = False
    action_log: List[str] = []

    while not done and step < MAX_STEPS:
        t0 = time.time()
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        elapsed = time.time() - t0
        step += 1

        action_type = action.get("action_type", "unknown")
        params = action.get("params", {})
        narrative = _format_action(action_type, params)

        # Deception annotation
        deceptive = agent._memory.get("deceptive", {})
        tag = ""
        if "member_id" in params and params["member_id"] in deceptive:
            tag = f" ← ⚠️ DECEPTIVE MEMBER"

        line = f"  Step {step:2d}: {narrative}{tag}"
        print(line)
        action_log.append(line)

        # Show deception detection events
        if action_type == "query_observable_signals" and "member_id" in params:
            mid = params["member_id"]
            if mid in deceptive and mid not in DECEPTION_SIGNALS:
                DECEPTION_SIGNALS.append(mid)
                print(f"          → 🚨 DECEPTION DETECTED: {deceptive[mid]}")

        if verbose:
            print(f"          → {_format_obs_summary(obs)}")

    # --- Greedy baseline comparison ---
    greedy_env = CrisisOpsEnv(scenario_fn=scenario_fn, curriculum_level=level)
    greedy_env.reset(seed=seed)
    greedy = GreedyPMBaseline()
    done_g = False
    step_g = 0
    while not done_g and step_g < MAX_STEPS:
        action_g = greedy.act(greedy_env._state)
        _, _, done_g, _ = greedy_env.step(action_g)
        step_g += 1

    agent_score = project_score(env._state)
    greedy_score = project_score(greedy_env._state)
    cf = agent_score - greedy_score

    print()
    print("=" * 70)
    print("EPISODE SUMMARY")
    print("=" * 70)
    print(f"  Agent score:       {agent_score:.3f}")
    print(f"  Greedy PM score:   {greedy_score:.3f}")
    print(f"  Counterfactual:    {cf:+.3f}  {'✓ BEAT GREEDY' if cf > 0 else '✗ lost to greedy'}")
    print(f"  Deceptive found:   {DECEPTION_SIGNALS}")
    cvr = len(agent._memory.get('signals', {})) / max(1, len(obs.get('team_members', [])))
    print(f"  Cross-verify rate: {cvr:.2f}")
    print(f"  Total steps:       {step}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Replay a CrisisOps episode with narrative output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to a saved model checkpoint directory")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    run_replay(seed=args.seed, level=args.level, model_path=args.model_path, verbose=args.verbose)


if __name__ == "__main__":
    main()
