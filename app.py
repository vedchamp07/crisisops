"""
app.py — CrisisOps v2 Gradio demo for HuggingFace Spaces.

Lets a user manually play one episode of CrisisOps and see the
counterfactual reward at the end compared to the greedy PM baseline.

Also supports running a random baseline agent for comparison.
"""

import json
import sys
import os

# Ensure the package root is on the path when run from HF Spaces
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr

from env.environment import CrisisOpsEnv
from env.state import ProjectState
from reward.counterfactual import project_score
from scenarios.level1 import get_random_level1_scenario
from scenarios.level2 import get_random_level2_scenario

# ---------------------------------------------------------------------------
# Global episode state (single-user demo)
# ---------------------------------------------------------------------------

_env: CrisisOpsEnv = None
_obs: dict = None
_history: list = []

ACTION_CHOICES = [
    "query_status",
    "query_member_report",
    "query_observable_signals",
    "query_ticket",
    "query_peer_opinion",
    "reassign_task",
    "communicate",
    "cut_scope",
    "escalate_risk",
    "request_resource",
    "update_timeline",
    "consult_expert",
    "force_truth",
    "trigger_whistleblower",
    "resolve_blocker",
    "submit_recovery_plan",
]

PARAM_HINT = {
    "query_status": "{}",
    "query_member_report": '{"member_id": "dev_1"}',
    "query_observable_signals": '{"member_id": "dev_1"}',
    "query_ticket": '{"task_id": "task_1"}',
    "query_peer_opinion": '{"asked_member_id": "dev_1", "about_member_id": "dev_2"}',
    "reassign_task": '{"task_id": "task_1", "to_member_id": "dev_2"}',
    "communicate": '{"message_type": "proactive_escalation_with_plan", "content": "Update", "target": "both"}',
    "cut_scope": '{"task_id": "task_1", "justification": "low priority"}',
    "escalate_risk": '{"crisis_id": "crisis_1", "risk_description": "high severity"}',
    "request_resource": '{"resource_type": "budget", "target_member_id": "dev_1"}',
    "update_timeline": '{"new_completion_date": "2026-05-15", "task_estimates": {}}',
    "consult_expert": "{}",
    "force_truth": '{"member_id": "dev_1"}',
    "trigger_whistleblower": "{}",
    "resolve_blocker": '{"task_id": "task_1", "resolution_notes": "Fixed"}',
    "submit_recovery_plan": '{"plan_summary": "Recovery complete", "risk_items": [], "timeline": "2 weeks"}',
}


def _format_obs(obs: dict) -> str:
    return json.dumps(obs, indent=2)


def _make_env(level: int = 1) -> CrisisOpsEnv:
    if level == 1:
        scenario_fn = get_random_level1_scenario()
    else:
        scenario_fn = get_random_level2_scenario()
    from reward.counterfactual import counterfactual_reward
    return CrisisOpsEnv(
        scenario_fn=scenario_fn,
        reward_fn=counterfactual_reward,
        curriculum_level=level,
    )


def reset_episode(level: int):
    global _env, _obs, _history
    _env = _make_env(int(level))
    _obs = _env.reset(seed=None)
    _history = []
    status = (
        f"Episode started — Level {level} | "
        f"Budget: {_obs.get('budget_remaining', '?')} | "
        f"PC: {_obs.get('political_capital', '?')} | "
        f"Memory: {'ready' if _obs.get('agent_memory') else 'empty'} | "
        f"Step: 0"
    )
    return _format_obs(_obs), status, "—"


def take_action(action_type: str, params_json: str):
    global _env, _obs, _history

    if _env is None:
        return "Run 'Reset Episode' first.", "Not started", "—"

    try:
        params = json.loads(params_json) if params_json.strip() else {}
    except json.JSONDecodeError as e:
        return _format_obs(_obs), f"Invalid JSON in params: {e}", "—"

    action = {"action_type": action_type, "params": params}
    _history.append(action_type)

    try:
        obs, reward, done, info = _env.step(action)
    except Exception as e:
        return _format_obs(_obs), f"Error: {e}", "—"

    _obs = obs
    # Show agent_memory prominently when present
    if obs.get('agent_memory'):
        obs_str = f"[AGENT MEMORY]\n{obs['agent_memory']}\n\n" + _format_obs(obs)
    else:
        obs_str = _format_obs(obs)

    if done:
        reward_str = f"{reward:+.3f} vs greedy PM"
        verdict = "✓ Agent beat greedy PM" if reward > 0 else "✗ Greedy PM did better"
        status = (
            f"EPISODE DONE — Counterfactual reward: {reward_str} | {verdict}\n"
            f"Actions used: {len(_history)}"
        )
    else:
        reward_str = "—"
        status = (
            f"Step {obs.get('current_step', '?')} | "
            f"Budget: {obs.get('budget_remaining', '?')} | "
            f"PC: {obs.get('political_capital', '?')} | "
            f"Memory: {'set' if obs.get('agent_memory') else 'empty'} | "
            f"Last: {action_type}"
        )

    return obs_str, status, reward_str


def update_param_hint(action_type: str):
    return PARAM_HINT.get(action_type, "{}")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="CrisisOps v2 — Deception Detection RL Environment") as demo:

    gr.Markdown("""
# CrisisOps v2

**Train LLMs to recover failing projects while detecting human deception.**

Team members misreport progress to avoid accountability. The PM agent must 
triangulate observable signals (commits, ticket age, peer testimony) against 
self-reports to identify liars and recover the project.

> *"In Kube SRE Gym, the agent reads machine logs — logs don't lie. 
> In CrisisOps, the agent asks engineers — engineers do lie."*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            level_slider = gr.Slider(minimum=1, maximum=2, step=1, value=1, label="Curriculum level")
            reset_btn = gr.Button("Reset Episode", variant="primary")

            gr.Markdown("### Take an action")
            action_dd = gr.Dropdown(
                choices=ACTION_CHOICES,
                value="query_status",
                label="Action type",
            )
            params_box = gr.Textbox(
                value="{}",
                label="Params (JSON)",
                lines=3,
                placeholder='{"member_id": "dev_1"}',
            )
            step_btn = gr.Button("Take Action", variant="secondary")

            gr.Markdown("""
**Quick reference:**
- Free: query_status, query_member_report, query_observable_signals, query_ticket
- Cost 1: reassign_task, communicate, escalate_risk, query_peer_opinion, force_truth, trigger_whistleblower, resolve_blocker
- Terminal: submit_recovery_plan

**Earning PC:** proactive_escalation_with_plan (+2), catching a liar (+3), update_timeline (+1)
            """)
            gr.Markdown("""
**New this episode:**
- One team member uses LLM to generate adaptive lies (if OPENAI_API_KEY set)
- Agent memory compresses every 8 steps — shown above the raw obs when available
- Political capital (PC) shown in status bar — earn by catching liars, spend on force_truth
""")

        with gr.Column(scale=2):
            obs_display = gr.Code(language="json", label="Current observation", lines=30)
            status_display = gr.Textbox(label="Status", lines=2)
            reward_display = gr.Textbox(label="Counterfactual reward (shown at episode end)")

    # Wire events
    reset_btn.click(
        reset_episode,
        inputs=[level_slider],
        outputs=[obs_display, status_display, reward_display],
    )
    step_btn.click(
        take_action,
        inputs=[action_dd, params_box],
        outputs=[obs_display, status_display, reward_display],
    )
    action_dd.change(
        update_param_hint,
        inputs=[action_dd],
        outputs=[params_box],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
