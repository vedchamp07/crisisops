"""
app.py — CrisisOps Gradio demo for HuggingFace Spaces.

War-room dashboard: manual play, scripted watch demo, and technical footers.
"""

from __future__ import annotations

import html
import json
import os
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr

from env.environment import CrisisOpsEnv
from env.state import ProjectState
from reward.baseline import GreedyPMBaseline
from reward.counterfactual import project_score
from scenarios.level1 import get_random_level1_scenario
from scenarios.level2 import get_random_level2_scenario
from scenarios.level3 import get_random_level3_scenario

# ---------------------------------------------------------------------------
# KPI placeholders — TODO: update from final eval (Aryan)
# ---------------------------------------------------------------------------
TRAINED_SCORE = 0.42  # TODO: update from final eval
GREEDY_SCORE = 0.18  # TODO: update from final eval

GITHUB_URL = "https://github.com/vedchamp07/crisisops"
BLOG_URL = "https://huggingface.co/posts"

# ---------------------------------------------------------------------------
# Deception rules (verbatim from training/grpo_trainer.py SYSTEM_PROMPT)
# ---------------------------------------------------------------------------
DECEPTION_RULES_VERBATIM = """=== DECEPTION DETECTION RULES ===
A member is DECEPTIVE when signals contradict their self-report:
    - reported_completion > 0.5  AND  ticket_age_days > 3  AND  commits_last_72h == 0  -> LYING
    - reported_completion > 0.6  AND  ticket_age_days > 5                               -> SUSPICIOUS
    - Honest member: ticket_age_days is low (0-2), commits_last_72h >= 2"""

DECEPTION_RULES_TOOLTIP = DECEPTION_RULES_VERBATIM.replace("\n", " ").strip()

# ---------------------------------------------------------------------------
# Global single-user state
# ---------------------------------------------------------------------------
_env: Optional[CrisisOpsEnv] = None
_greedy_env: Optional[CrisisOpsEnv] = None
_greedy_policy: Optional[GreedyPMBaseline] = None
_obs: Optional[dict] = None
_history: List[str] = []
_signal_cache: Dict[str, Dict[str, int]] = {}
_action_log: List[str] = []
_pending_action: str = ""
_curriculum_level: int = 1
_last_cf_delta: float = 0.0

PRESET_CHOICES = [
    "Easy demo (1 liar, obvious)",
    "Realistic (2 liars, ambiguous)",
    "Hardcore (3+ liars, alliance)",
    "Random",
]

PRESET_LEVEL_SEED: Dict[str, Optional[Tuple[int, int]]] = {
    "Easy demo (1 liar, obvious)": (1, 42),
    "Realistic (2 liars, ambiguous)": (2, 7),
    "Hardcore (3+ liars, alliance)": (3, 99),
    "Random": None,
}

ACTION_REFERENCE_HTML = """
<table class="cops-table">
<thead><tr><th>Action</th><th>Cost</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>query_status</td><td>0</td><td>Project snapshot</td></tr>
<tr><td>query_member_report</td><td>0</td><td>Self-report</td></tr>
<tr><td>query_observable_signals</td><td>0</td><td>ticket_age_days, commits_last_72h, peer_mentions</td></tr>
<tr><td>query_ticket</td><td>0</td><td>Task detail</td></tr>
<tr><td>reassign_task</td><td>1</td><td>task_id, to_member_id</td></tr>
<tr><td>communicate</td><td>1</td><td>message_type, content, target</td></tr>
<tr><td>cut_scope</td><td>1</td><td>task_id, justification</td></tr>
<tr><td>escalate_risk</td><td>1</td><td>crisis_id, risk_description</td></tr>
<tr><td>request_resource</td><td>1</td><td>resource_type, target_member_id</td></tr>
<tr><td>update_timeline</td><td>1</td><td>new_completion_date, task_estimates</td></tr>
<tr><td>consult_expert</td><td>1</td><td>—</td></tr>
<tr><td>query_peer_opinion</td><td>1</td><td>asked_member_id, about_member_id</td></tr>
<tr><td>force_truth</td><td>1 + 3 PC</td><td>member_id</td></tr>
<tr><td>trigger_whistleblower</td><td>1 + 6 PC</td><td>—</td></tr>
<tr><td>resolve_blocker</td><td>2</td><td>task_id, resolution_notes</td></tr>
<tr><td>submit_recovery_plan</td><td>terminal</td><td>plan_summary, risk_items, timeline</td></tr>
</tbody>
</table>
<p style="color:#6b7280;font-size:0.8rem;margin-top:8px;">PC starts at 5. Earn: proactive_escalation_with_plan (+2), catching a liar (+3), update_timeline (+1). Spend: force_truth (−3), whistleblower (−6).</p>
"""


def _load_css() -> str:
    p = os.path.join(os.path.dirname(__file__), "static", "styles.css")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return f.read()
    return ""


def _scenario_factory(level: int) -> Callable[..., ProjectState]:
    if level == 1:
        return get_random_level1_scenario()
    if level == 2:
        return get_random_level2_scenario()
    return get_random_level3_scenario()


def _make_env(scenario_fn: Callable[..., ProjectState], curriculum_level: int) -> CrisisOpsEnv:
    from reward.counterfactual import counterfactual_reward

    return CrisisOpsEnv(
        scenario_fn=scenario_fn,
        reward_fn=counterfactual_reward,
        curriculum_level=curriculum_level,
    )


def _format_obs(obs: dict) -> str:
    return json.dumps(obs, indent=2)


def _escape(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _initials(name: str, member_id: str) -> str:
    parts = name.split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    if name:
        return name[:2].upper()
    return member_id.replace("dev_", "")[:2].upper() or "?"


def _merge_signals_from_obs(obs: dict) -> None:
    global _signal_cache
    if obs.get("action_type") == "query_observable_signals" and "signals" in obs:
        mid = obs.get("member_id")
        if mid and isinstance(obs["signals"], dict):
            _signal_cache[str(mid)] = dict(obs["signals"])


def _suspicion(
    reported: float,
    signals: Optional[Dict[str, int]],
) -> Tuple[str, str]:
    """Return (ring_class, label)."""
    if signals is None:
        return "ring-unknown", "unknown"
    ta = int(signals.get("ticket_age_days", 0))
    cm = int(signals.get("commits_last_72h", 0))
    if reported > 0.5 and ta > 3 and cm == 0:
        return "ring-red", "lying"
    if reported > 0.6 and ta > 5:
        return "ring-amber", "suspicious"
    return "ring-green", "clean"


def _sev_class(sev: float) -> str:
    if sev >= 7.5:
        return "cops-sev-high"
    if sev >= 5.0:
        return "cops-sev-mid"
    return "cops-sev-low"


def _hero_kpi_block() -> str:
    imp = 0.0
    if abs(GREEDY_SCORE) > 1e-9:
        imp = (TRAINED_SCORE - GREEDY_SCORE) / abs(GREEDY_SCORE) * 100.0
    return f"""
<div class="cops-hero">
  <h1>CrisisOps</h1>
  <p class="cops-tagline">Train AI agents to recover failing projects when humans lie about progress</p>
  <div class="cops-kpi-row">
    <div class="cops-kpi"><div class="label">Trained Agent Score</div><div class="value">+{TRAINED_SCORE:.2f}</div></div>
    <div class="cops-kpi"><div class="label">Greedy Baseline</div><div class="value" style="color:#f59e0b">+{GREEDY_SCORE:.2f}</div></div>
    <div class="cops-kpi"><div class="label">Improvement</div><div class="value">+{imp:.0f}%</div></div>
  </div>
  <p style="margin:12px 0 0 0;font-size:0.8rem;color:#6b7280;">
    <a href="{GITHUB_URL}" target="_blank" rel="noopener" style="color:#3b82f6;">GitHub</a>
    <span style="margin:0 8px;color:#374151;">|</span>
    <span style="color:#6b7280;">Use the footer accordion for &ldquo;How it works&rdquo; and raw JSON.</span>
  </p>
</div>
"""


def _render_war_room(
    obs: Optional[dict],
    highlight_member_id: Optional[str] = None,
) -> str:
    if not obs:
        return '<div class="cops-panel"><div class="cops-col-title">War room</div><p style="color:#6b7280;">Start an episode.</p></div>'

    members = obs.get("team_members") or []
    crises = sorted(
        obs.get("crises") or [],
        key=lambda c: float(c.get("severity", 0)),
        reverse=True,
    )
    budget = int(obs.get("budget_remaining", 0))
    pc = float(obs.get("political_capital", 0))
    step = int(obs.get("current_step", 0))
    max_steps = 30

    used_budget = max(0, 20 - budget)
    budget_pct = min(100.0, used_budget / 20.0 * 100.0)
    budget_color = "#3b82f6"
    if budget <= 5:
        budget_color = "#f59e0b"
    if budget <= 2:
        budget_color = "#ef4444"

    pc_pct = min(100.0, max(0.0, pc / 10.0 * 100.0))
    global _last_cf_delta
    cf = _last_cf_delta
    cf_cap = 0.25
    cf_clamped = max(-cf_cap, min(cf_cap, cf))
    if cf_clamped >= 0:
        pos_w = abs(cf_clamped) / cf_cap * 50.0
        neg_w = 0.0
    else:
        pos_w = 0.0
        neg_w = abs(cf_clamped) / cf_cap * 50.0

    col_a = ['<div class="cops-col-title">Team members</div>']
    for m in members:
        mid = str(m.get("member_id", ""))
        name = str(m.get("name", mid))
        role = str(m.get("role", ""))
        rc = float(m.get("reported_completion", 0))
        sig = _signal_cache.get(mid)
        ring, susp_lbl = _suspicion(rc, sig)
        if sig is None:
            tip = _escape("Run query_observable_signals to investigate. " + DECEPTION_RULES_TOOLTIP[:400])
            ring = "ring-unknown"
        else:
            tip = _escape(DECEPTION_RULES_TOOLTIP[:500])
        av_extra = " ring-highlight" if highlight_member_id == mid else ""
        initials = _initials(name, mid)
        bar_w = min(100.0, rc * 100.0)

        chips = []
        if sig is None:
            chips.append('<span class="cops-chip">ticket_age: ?</span>')
            chips.append('<span class="cops-chip">commits_72h: ?</span>')
            chips.append('<span class="cops-chip">peer: ?</span>')
        else:
            chips.append(f'<span class="cops-chip">ticket_age: {sig.get("ticket_age_days", "?")}</span>')
            chips.append(f'<span class="cops-chip">commits_72h: {sig.get("commits_last_72h", "?")}</span>')
            chips.append(f'<span class="cops-chip">peer: {sig.get("peer_mentions", "?")}</span>')

        col_a.append(
            f"""
<div class="cops-card" title="{tip}">
  <div class="cops-card-head">
    <div class="cops-avatar {ring}{av_extra}" title="{tip}">{_escape(initials)}</div>
    <div>
      <div class="cops-name">{_escape(name)}</div>
      <div class="cops-role">{_escape(role)} · {_escape(susp_lbl)}</div>
    </div>
  </div>
  <div class="cops-bar-label">reported_completion</div>
  <div class="cops-bar-track"><div class="cops-bar-fill" style="width:{bar_w:.1f}%"></div></div>
  <div style="font-family:var(--mono);font-size:0.75rem;color:#e5e7eb;margin-top:4px;">{rc*100:.1f}%</div>
  <div class="cops-chips">{"".join(chips)}</div>
</div>
"""
        )

    col_b = ['<div class="cops-col-title">Crises</div>']
    if not crises:
        col_b.append('<div class="cops-card"><span style="color:#6b7280;">No active crises</span></div>')
    for c in crises:
        sev = float(c.get("severity", 0))
        resolved = bool(c.get("is_resolved"))
        title = str(c.get("crisis_type", "crisis"))
        desc = str(c.get("description", ""))[:120]
        aids = c.get("affected_task_ids") or []
        sc = _sev_class(sev)
        stamp = '<div class="cops-resolved">RESOLVED</div>' if resolved else ""
        col_b.append(
            f"""
<div class="cops-card">
  <strong>{_escape(title)}</strong>
  <span class="cops-crisis-sev {sc}">SEV {sev:.1f}</span>
  <div style="font-size:0.75rem;color:#6b7280;margin-top:6px;">{_escape(desc)}</div>
  <div style="font-family:var(--mono);font-size:0.7rem;color:#9ca3af;margin-top:8px;">tasks: {len(aids)}</div>
  {stamp}
</div>
"""
        )

    col_c = [
        '<div class="cops-col-title">Resources &amp; timeline</div>',
        f'<div class="cops-card"><div class="cops-meter-label">Budget consumed (remaining {budget})</div>'
        f'<div class="cops-meter-track"><div class="cops-meter-fill" style="width:{budget_pct:.1f}%;background:{budget_color}"></div></div>',
        f'<div class="cops-meter-label">Political capital ({pc:.1f} / 10)</div>'
        f'<div class="cops-meter-track">'
        f'<div class="cops-meter-marker" style="left:30%"></div>'
        f'<div class="cops-meter-marker" style="left:60%"></div>'
        f'<div class="cops-meter-fill" style="width:{pc_pct:.1f}%;background:#10b981"></div></div>'
        f'<div style="font-size:0.65rem;color:#6b7280;font-family:var(--mono);">Markers: force_truth ≥3 · whistleblower ≥6</div>'
        f'<div class="cops-meter-label" style="margin-top:12px;">Step {step} / {max_steps}</div>'
        f'<div class="cops-meter-label">Live: project score vs greedy (parallel play)</div>'
        f'<div class="cops-score-track">'
        f'<div class="cops-score-mid"></div>'
        f'<div class="cops-score-pos" style="width:{pos_w:.1f}%;left:50%;"></div>'
        f'<div class="cops-score-neg" style="width:{neg_w:.1f}%;right:50%;"></div></div>'
        f'<div style="font-family:var(--mono);font-size:0.75rem;color:#e5e7eb;margin-top:6px;">Δ vs greedy: {cf:+.3f}</div></div>',
    ]

    return f"""
<div class="cops-panel">
  <div class="cops-war-wrap">
    <div>{"".join(col_a)}</div>
    <div>{"".join(col_b)}</div>
    <div>{"".join(col_c)}</div>
  </div>
</div>
"""


def _sync_greedy_one_step() -> None:
    global _greedy_env, _greedy_policy
    if _greedy_env is None or _greedy_policy is None:
        return
    if _greedy_env._state.done:
        return
    try:
        act = _greedy_policy.act(_greedy_env._state)
        _greedy_env.step(act)
    except Exception:
        pass


def _update_cf_delta() -> None:
    global _last_cf_delta, _env, _greedy_env
    if _env is None or _greedy_env is None or _env._state is None or _greedy_env._state is None:
        _last_cf_delta = 0.0
        return
    try:
        pa = project_score(_env._state)
        pg = project_score(_greedy_env._state)
        _last_cf_delta = pa - pg
    except Exception:
        _last_cf_delta = 0.0


def _log_action_line(action_type: str, ps_delta: float) -> None:
    global _action_log
    line = f"{action_type:24s}  Δscore {ps_delta:+.4f}"
    _action_log.append(line)
    _action_log = _action_log[-5:]


def _preset_to_level_seed(preset: str) -> Tuple[int, int]:
    spec = PRESET_LEVEL_SEED.get(preset)
    if spec is None:
        lvl = random.choice([1, 2, 3])
        return lvl, random.randint(1, 2**30)
    return spec[0], spec[1]


def reset_episode(preset: str) -> Tuple:
    global _env, _greedy_env, _greedy_policy, _obs, _history, _signal_cache, _action_log
    global _pending_action, _curriculum_level, _last_cf_delta

    level, seed = _preset_to_level_seed(preset)
    _curriculum_level = level
    scenario_fn = _scenario_factory(level)
    _env = _make_env(scenario_fn, level)
    _greedy_env = _make_env(scenario_fn, level)
    _greedy_policy = GreedyPMBaseline()

    _obs = _env.reset(seed=seed)
    _greedy_env.reset(seed=seed)
    _history = []
    _signal_cache = {}
    _action_log = []
    _pending_action = ""
    _merge_signals_from_obs(_obs)
    _update_cf_delta()

    status = (
        f"Episode started | preset={preset} | level={level} | seed={seed} | "
        f"budget={_obs.get('budget_remaining')} | step=0"
    )
    war = _render_war_room(_obs)
    raw = _format_obs(_obs)
    reward_disp = "—"
    log_txt = "(no actions yet)"
    banner = ""
    mc = _member_choices(_obs)
    tc = _task_choices(_obs)
    return (
        war,
        status,
        raw,
        reward_disp,
        mc,
        tc,
        mc,
        mc,
        mc,
        mc,
        log_txt,
        banner,
        _crisis_choices(_obs),
        mc,
    )


def _member_choices(obs: Optional[dict]) -> gr.update:
    if not obs:
        return gr.update(choices=[], value=None)
    mids = [str(m["member_id"]) for m in (obs.get("team_members") or [])]
    return gr.update(choices=mids, value=mids[0] if mids else None)


def _task_choices(obs: Optional[dict]) -> gr.update:
    if not obs or _env is None or _env._state is None:
        return gr.update(choices=[], value=None)
    tids = [t.task_id for t in _env._state.tasks]
    return gr.update(choices=tids, value=tids[0] if tids else None)


def _crisis_choices(obs: Optional[dict]) -> gr.update:
    if not obs:
        return gr.update(choices=[], value=None)
    ids = [str(c.get("crisis_id")) for c in (obs.get("crises") or [])]
    return gr.update(choices=ids, value=ids[0] if ids else None)


def take_action(
    action_type: str,
    params_json: str,
) -> Tuple:
    """Backward-compatible core step; params_json may be legacy."""
    global _env, _obs, _history

    if _env is None:
        empty = _render_war_room(None)
        return (
            empty,
            "Run Start Episode first.",
            "{}",
            "—",
            "—",
            "",
            gr.update(value=""),
        )

    try:
        params = json.loads(params_json) if params_json.strip() else {}
    except json.JSONDecodeError as e:
        _update_cf_delta()
        return (
            _render_war_room(_obs),
            f"Invalid JSON: {e}",
            _format_obs(_obs),
            "—",
            "\n".join(_action_log),
            "",
            gr.update(),
        )

    action = {"action_type": action_type, "params": params}
    ps_before = project_score(_env._state) if _env._state else 0.0

    try:
        obs, reward, done, info = _env.step(action)
    except Exception as e:
        _update_cf_delta()
        return (
            _render_war_room(_obs),
            f"Error: {e}",
            _format_obs(_obs or {}),
            "—",
            "\n".join(_action_log),
            "",
            gr.update(),
        )

    _obs = obs
    _history.append(action_type)
    _merge_signals_from_obs(obs)
    _sync_greedy_one_step()
    _update_cf_delta()

    ps_after = project_score(_env._state) if _env._state else 0.0
    _log_action_line(action_type, ps_after - ps_before)

    raw = _format_obs(obs)
    if obs.get("agent_memory"):
        raw = f"/* agent_memory */\n{obs['agent_memory']}\n\n" + raw

    if done:
        reward_str = f"{reward:+.3f} vs greedy PM"
        status = f"DONE | CF reward: {reward_str} | actions: {len(_history)}"
        banner = (
            f'<div class="cops-banner-win">EPISODE END — Counterfactual reward: {reward:+.3f} vs greedy</div>'
            if reward > 0
            else f'<div class="cops-banner-win" style="border-color:#ef4444;color:#ef4444;">EPISODE END — {reward:+.3f} vs greedy</div>'
        )
    else:
        reward_str = "—"
        status = (
            f"Step {obs.get('current_step')} | budget {obs.get('budget_remaining')} | "
            f"PC {obs.get('political_capital')} | last: {action_type}"
        )
        banner = ""

    return (
        _render_war_room(_obs),
        status,
        raw,
        reward_str,
        "\n".join(_action_log) if _action_log else "—",
        banner,
        gr.update(value=""),
    )


def _params_for_console(
    action: str,
    m_obs: Optional[str],
    m_force: Optional[str],
    m_ask: Optional[str],
    m_about: Optional[str],
    task_id: Optional[str],
    task_to: Optional[str],
    comm_type: str,
    comm_body: str,
    esc_crisis: str,
    esc_txt: str,
    cut_task: str,
    cut_why: str,
    res_notes: str,
    plan_body: str,
    timeline: str,
) -> str:
    if action == "query_status":
        return "{}"
    if action == "query_member_report":
        return json.dumps({"member_id": m_obs or ""})
    if action == "query_observable_signals":
        return json.dumps({"member_id": m_obs or ""})
    if action == "query_ticket":
        return json.dumps({"task_id": task_id or ""})
    if action == "query_peer_opinion":
        return json.dumps({"asked_member_id": m_ask or "", "about_member_id": m_about or ""})
    if action == "reassign_task":
        return json.dumps({"task_id": task_id or "", "to_member_id": task_to or ""})
    if action == "communicate":
        return json.dumps(
            {"message_type": comm_type, "content": comm_body, "target": "both"}
        )
    if action == "escalate_risk":
        return json.dumps({"crisis_id": esc_crisis, "risk_description": esc_txt})
    if action == "cut_scope":
        return json.dumps({"task_id": cut_task, "justification": cut_why})
    if action == "request_resource":
        return json.dumps({"resource_type": "budget", "target_member_id": m_obs or ""})
    if action == "update_timeline":
        return json.dumps({"new_completion_date": timeline, "task_estimates": {}})
    if action == "consult_expert":
        return "{}"
    if action == "force_truth":
        return json.dumps({"member_id": m_force or ""})
    if action == "trigger_whistleblower":
        return "{}"
    if action == "resolve_blocker":
        return json.dumps({"task_id": task_id or "", "resolution_notes": res_notes})
    if action == "submit_recovery_plan":
        return json.dumps(
            {"plan_summary": plan_body, "risk_items": [], "timeline": timeline or "TBD"}
        )
    return "{}"


def console_take_action(
    action: str,
    m_obs, m_force, m_ask, m_about,
    task_id, task_to,
    comm_type, comm_body,
    esc_crisis, esc_txt,
    cut_task, cut_why,
    res_notes,
    plan_body, timeline,
) -> Tuple:
    pj = _params_for_console(
        action, m_obs, m_force, m_ask, m_about, task_id, task_to,
        comm_type, comm_body, esc_crisis, esc_txt, cut_task, cut_why,
        res_notes, plan_body, timeline,
    )
    a, b, c, d, e, f, g = take_action(action, pj)
    mc = _member_choices(_obs)
    tc = _task_choices(_obs)
    crisis_ids = []
    if _obs:
        crisis_ids = [str(x.get("crisis_id")) for x in (_obs.get("crises") or [])]
    criz = _crisis_choices(_obs)
    return a, b, c, d, e, f, g, mc, mc, mc, mc, tc, mc, criz


def queue_investigate(member_id: Optional[str]) -> gr.update:
    global _pending_action
    _pending_action = "query_observable_signals"
    return gr.update(value="query_observable_signals"), gr.update(value=member_id)


def _build_demo_trajectory(initial_obs: dict) -> List[Tuple[dict, str]]:
    """Heuristic demo trace from initial observation (watch mode)."""
    tm = list(initial_obs.get("team_members") or [])
    if not tm:
        return [({"action_type": "query_status", "params": {}}, "Baseline snapshot")]
    sorted_m = sorted(tm, key=lambda x: -float(x.get("reported_completion", 0)))
    trace: List[Tuple[dict, str]] = [
        ({"action_type": "query_status", "params": {}}, "Snapshot global status — map the crisis"),
    ]
    for m in sorted_m[: min(3, len(sorted_m))]:
        mid = m["member_id"]
        trace.append(
            (
                {"action_type": "query_observable_signals", "params": {"member_id": mid}},
                f"Cross-verify {m.get('name')} — signals vs self-report",
            )
        )
    top = sorted_m[0]
    mid_hi = top["member_id"]
    tids = top.get("assigned_task_ids") or []
    others = [x for x in tm if x["member_id"] != mid_hi]
    if tids and others:
        low = min(others, key=lambda x: float(x.get("reported_completion", 1)))
        trace.append(
            (
                {
                    "action_type": "reassign_task",
                    "params": {"task_id": tids[0], "to_member_id": low["member_id"]},
                },
                "Reassign work away from the inflated report — cut liar leverage",
            )
        )
    trace.append(
        (
            {
                "action_type": "communicate",
                "params": {
                    "message_type": "proactive_escalation_with_plan",
                    "content": "Escalating with recovery plan and verified assignments.",
                    "target": "both",
                },
            },
            "Stakeholder comms — proactive_escalation_with_plan",
        )
    )
    trace.append(
        (
            {
                "action_type": "submit_recovery_plan",
                "params": {
                    "plan_summary": "Deception surfaced via observable signals; tasks reassigned; risks enumerated.",
                    "risk_items": [],
                    "timeline": "2026-06-01",
                },
            },
            "Submit recovery plan — end episode",
        )
    )
    return trace


def play_watch_demo(preset: str):
    """Generator: replay demo with delay (watch mode)."""
    global _env, _greedy_env, _greedy_policy, _obs, _history, _signal_cache, _action_log
    global _curriculum_level, _last_cf_delta

    level, seed = _preset_to_level_seed(preset)
    scenario_fn = _scenario_factory(level)
    _env = _make_env(scenario_fn, level)
    _greedy_env = _make_env(scenario_fn, level)
    _greedy_policy = GreedyPMBaseline()
    _obs = _env.reset(seed=seed)
    _greedy_env.reset(seed=seed)
    _history = []
    _signal_cache = {}
    _action_log = []
    _merge_signals_from_obs(_obs)
    _update_cf_delta()

    trace = _build_demo_trajectory(_obs)
    final_reward = 0.0
    banner = ""
    cf_computed = False

    yield (
        _render_war_room(_obs),
        f"Expert trace demo | level={level} seed={seed} | {len(trace)} steps",
        _format_obs(_obs),
        "—",
        "\n".join(_action_log) if _action_log else "—",
        banner,
    )

    for i, (act, narr) in enumerate(trace):
        time.sleep(1.2)
        ps_before = project_score(_env._state) if _env._state else 0.0
        try:
            obs, reward, done, _ = _env.step(act)
        except Exception as e:
            banner = f'<div class="cops-banner-win" style="border-color:#ef4444;color:#ef4444;">Demo error: {e}</div>'
            yield (
                _render_war_room(_obs),
                f"Error at step {i}",
                _format_obs(_obs or {}),
                "—",
                "\n".join(_action_log),
                banner,
            )
            return
        _obs = obs
        _history.append(act["action_type"])
        _merge_signals_from_obs(obs)
        _sync_greedy_one_step()
        _update_cf_delta()
        ps_after = project_score(_env._state) if _env._state else 0.0
        _log_action_line(act["action_type"], ps_after - ps_before)
        _action_log[-1] = _action_log[-1] + f"  | {narr}"

        mid = act.get("params", {}).get("member_id")
        hl = str(mid) if act["action_type"] == "query_observable_signals" else None

        if done:
            final_reward = reward
            cf_computed = True
            if final_reward > 0:
                verdict = "AGENT BEAT GREEDY"
                color = "#10b981"
            else:
                verdict = "AGENT UNDERPERFORMED GREEDY"
                color = "#ef4444"
            banner = (
                f'<div class="cops-banner-win" style="border-color:{color};color:{color};">'
                f"EPISODE END: {final_reward:+.3f} vs greedy — {verdict}</div>"
            )
        else:
            banner = ""

        yield (
            _render_war_room(_obs, highlight_member_id=hl),
            f"Demo step {i+1}/{len(trace)} | {narr}",
            _format_obs(obs),
            f"{reward:+.3f}" if done else "—",
            "\n".join(_action_log),
            banner,
        )
        if done:
            break

    if not cf_computed:
        banner = (
            '<div class="cops-banner-win" style="border-color:#6b7280;color:#6b7280;">'
            "Demo trajectory finished without submitting recovery plan — no episode reward computed</div>"
        )
    yield (
        _render_war_room(_obs),
        "Expert trace demo complete",
        _format_obs(_obs),
        f"{final_reward:+.3f}" if cf_computed else "—",
        "\n".join(_action_log),
        banner,
    )


def _toggle_mode(mode: str):
    w, p, c = False, False, False
    if mode.startswith("Watch"):
        w = True
    elif mode.startswith("Play"):
        p = True
    else:
        c = True
    return (
        gr.update(visible=w),
        gr.update(visible=p),
        gr.update(visible=c),
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
_custom_css = _load_css()

with gr.Blocks(
    title="CrisisOps — Deception Detection RL Environment",
    css=_custom_css,
) as demo:
    gr.HTML(_hero_kpi_block())

    mode_radio = gr.Radio(
        choices=[
            "Watch Expert Trace",
            "Play Yourself",
            "Compare: Trained vs Greedy",
        ],
        value="Play Yourself",
        label="Mode",
        elem_classes=["cops-mode-tabs"],
    )

    with gr.Row():
        preset_dd = gr.Dropdown(choices=PRESET_CHOICES, value=PRESET_CHOICES[0], label="Scenario preset")
        start_btn = gr.Button("Start Episode", variant="primary")
        reset_btn = gr.Button("Reset", variant="secondary")

    war_html = gr.HTML(_render_war_room(None))

    with gr.Group(visible=False) as watch_group:
        gr.Markdown(
            "**Watch Expert Trace** — scripted expert-style playback for illustration only "
            "(not the GRPO-trained policy; no Space-hosted LoRA)."
        )
        watch_go = gr.Button("Play expert trace (timed)", variant="primary")

    with gr.Group(visible=True) as play_group:
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Action console")
                with gr.Tabs():
                    with gr.Tab("Investigate (0)"):
                        b_qs = gr.Button("query_status")
                        b_qmr = gr.Button("query_member_report")
                        b_qsig = gr.Button("query_observable_signals")
                        b_qt = gr.Button("query_ticket")
                    with gr.Tab("Decide (1)"):
                        b_re = gr.Button("reassign_task")
                        b_co = gr.Button("communicate")
                        b_es = gr.Button("escalate_risk")
                        b_cut = gr.Button("cut_scope")
                        b_rr = gr.Button("request_resource")
                        b_ut = gr.Button("update_timeline")
                        b_ce = gr.Button("consult_expert")
                        b_qp = gr.Button("query_peer_opinion")
                    with gr.Tab("Heavy (2 / PC)"):
                        b_ft = gr.Button("force_truth")
                        b_tw = gr.Button("trigger_whistleblower")
                        b_rb = gr.Button("resolve_blocker")
                    with gr.Tab("Submit"):
                        b_sub = gr.Button("submit_recovery_plan")

                action_state = gr.Textbox(visible=False, value="query_status")
                m_obs = gr.Dropdown(label="member_id (reports / signals / request_resource)", choices=[])
                m_force = gr.Dropdown(label="force_truth member", choices=[])
                m_ask = gr.Dropdown(label="peer: asked_member_id", choices=[])
                m_about = gr.Dropdown(label="peer: about_member_id", choices=[])
                task_id = gr.Dropdown(label="task_id", choices=[])
                task_to = gr.Dropdown(label="reassign to_member_id", choices=[])
                comm_type = gr.Dropdown(
                    choices=["proactive_escalation_with_plan", "risk_communication", "status_update"],
                    value="proactive_escalation_with_plan",
                    label="communicate message_type",
                )
                comm_body = gr.Textbox(label="communicate content", lines=2)
                esc_crisis = gr.Dropdown(label="escalate crisis_id", choices=[])
                esc_txt = gr.Textbox(label="risk_description", value="Schedule risk")
                cut_task = gr.Textbox(label="cut_scope task_id", value="")
                cut_why = gr.Textbox(label="justification", value="deprioritize")
                res_notes = gr.Textbox(label="resolve_blocker resolution_notes", value="Unblocked")
                plan_body = gr.Textbox(label="plan_summary", lines=3, value="Recovery plan")
                timeline = gr.Textbox(label="timeline / update_timeline date", value="2026-06-01")

                take_btn = gr.Button("Take Action", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Quick investigate")
                inv_member = gr.Dropdown(label="Member", choices=[])
                inv_btn = gr.Button("Queue query_observable_signals", variant="secondary")
                action_log = gr.Textbox(label="Action log (last 5)", lines=10, elem_classes=["cops-action-log"])
                legacy_params = gr.Textbox(label="Params JSON (advanced)", visible=False, value="{}")

    with gr.Group(visible=False) as compare_group:
        gr.Markdown(
            "### Compare: Trained vs Greedy\n\n**Coming soon** — side-by-side war rooms with a shared Step control "
            "will advance the trained trace and greedy baseline in lockstep."
        )

    status_display = gr.Textbox(label="Status", lines=2)
    reward_display = gr.Textbox(label="Counterfactual reward (episode end)")
    win_banner = gr.HTML("")

    with gr.Accordion("Technical footer", open=False):
        with gr.Tabs():
            with gr.Tab("How it works"):
                gr.Markdown(
                    f"""
**Counterfactual reward** = `project_score(your final state) − project_score(greedy PM final state)` on the same initial scenario. `project_score` weights recovery, client satisfaction, and team morale (see `reward/counterfactual.py`). **GRPO** trains the policy with group-relative advantages on generated actions.

Blog / writeups: [{BLOG_URL}]({BLOG_URL})
"""
                )
            with gr.Tab("Deception rules (verbatim)"):
                gr.Code(
                    value=DECEPTION_RULES_VERBATIM,
                    language=None,
                    label="From training policy prompt",
                )
            with gr.Tab("Action reference"):
                gr.HTML(ACTION_REFERENCE_HTML)
            with gr.Tab("Raw observation JSON"):
                obs_code = gr.Code(language="json", label="Current observation")

    # --- Wire: mode visibility ---
    mode_radio.change(
        _toggle_mode,
        inputs=[mode_radio],
        outputs=[watch_group, play_group, compare_group],
    )

    # --- Wire: reset / start ---
    _start_outputs = (
        war_html,
        status_display,
        obs_code,
        reward_display,
        m_obs,
        task_id,
        task_to,
        m_force,
        m_ask,
        m_about,
        action_log,
        win_banner,
        esc_crisis,
        inv_member,
    )

    start_btn.click(
        reset_episode,
        inputs=[preset_dd],
        outputs=list(_start_outputs),
    )
    reset_btn.click(
        reset_episode,
        inputs=[preset_dd],
        outputs=list(_start_outputs),
    )

    # --- Expert trace demo (generator) ---
    watch_go.click(
        play_watch_demo,
        inputs=[preset_dd],
        outputs=[war_html, status_display, obs_code, reward_display, action_log, win_banner],
    )

    # --- Console action wiring ---
    console_inputs = [
        action_state,
        m_obs,
        m_force,
        m_ask,
        m_about,
        task_id,
        task_to,
        comm_type,
        comm_body,
        esc_crisis,
        esc_txt,
        cut_task,
        cut_why,
        res_notes,
        plan_body,
        timeline,
    ]
    out_console = (
        war_html,
        status_display,
        obs_code,
        reward_display,
        action_log,
        win_banner,
        legacy_params,
        m_obs,
        m_force,
        m_ask,
        m_about,
        task_id,
        task_to,
        esc_crisis,
    )

    def bind_btn(btn, name):
        def _set_name():
            return gr.update(value=name)

        btn.click(_set_name, outputs=[action_state]).then(
            console_take_action,
            inputs=console_inputs,
            outputs=list(out_console),
        )

    bind_btn(b_qs, "query_status")
    bind_btn(b_qmr, "query_member_report")
    bind_btn(b_qsig, "query_observable_signals")
    bind_btn(b_qt, "query_ticket")
    bind_btn(b_re, "reassign_task")
    bind_btn(b_co, "communicate")
    bind_btn(b_es, "escalate_risk")
    bind_btn(b_cut, "cut_scope")
    bind_btn(b_rr, "request_resource")
    bind_btn(b_ut, "update_timeline")
    bind_btn(b_ce, "consult_expert")
    bind_btn(b_qp, "query_peer_opinion")
    bind_btn(b_ft, "force_truth")
    bind_btn(b_tw, "trigger_whistleblower")
    bind_btn(b_rb, "resolve_blocker")
    bind_btn(b_sub, "submit_recovery_plan")

    inv_btn.click(
        lambda m: (gr.update(value="query_observable_signals"), gr.update(value=m)),
        inputs=[inv_member],
        outputs=[action_state, m_obs],
    ).then(
        console_take_action,
        inputs=console_inputs,
        outputs=list(out_console),
    )

    take_btn.click(
        console_take_action,
        inputs=console_inputs,
        outputs=list(out_console),
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    inbrowser=False,
    share=False,
    show_error=True,
)
