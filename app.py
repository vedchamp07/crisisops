"""
app.py — CrisisOps Gradio demo for HuggingFace Spaces.

War-room dashboard for the CrisisOps deception-detection RL environment.

Two modes:
  - Watch Expert Trace: scripted heuristic playback (illustrative, NOT the
    GRPO-trained policy; LoRA weights are not hosted in the Space).
  - Play Yourself: interactive console with contextual params.

All CSS is inlined to keep deploys self-contained. Backend modules
(env/, reward/, scenarios/) are imported but never modified.
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
from reward.counterfactual import counterfactual_reward, project_score
from scenarios.level1 import get_random_level1_scenario
from scenarios.level2 import get_random_level2_scenario
from scenarios.level3 import get_random_level3_scenario

GITHUB_URL = os.environ.get("CRISISOPS_GITHUB_URL", "https://github.com/aryannzzz/CrisisOps")
BLOG_URL = os.environ.get("CRISISOPS_BLOG_URL", "").strip()

DECEPTION_RULES_VERBATIM = """=== DECEPTION DETECTION RULES ===
A member is DECEPTIVE when signals contradict their self-report:
    - reported_completion > 0.5  AND  ticket_age_days > 3  AND  commits_last_72h == 0  -> LYING
    - reported_completion > 0.6  AND  ticket_age_days > 5                               -> SUSPICIOUS
    - Honest member: ticket_age_days is low (0-2), commits_last_72h >= 2"""

# ---------------------------------------------------------------------------
# Inlined CSS — keeps the Space self-contained (no external file load)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;

    /* Light-first palette (HF Spaces default is light) */
    --bg: #f6f7fb;
    --panel: #ffffff;
    --panel-2: #f1f3fa;
    --border: #d6d9e6;
    --text: #0f172a;
    --muted: #5b637a;

    /* Accents (shared) */
    --accent: #0891b2;       /* cyan */
    --accent-2: #7c3aed;     /* violet */
    --success: #059669;      /* emerald */
    --warning: #b45309;      /* amber */
    --danger:  #e11d48;      /* rose */
}

@media (prefers-color-scheme: dark) {
    :root {
        /* Dark palette override */
        --bg: #07091a;
        --panel: #11142b;
        --panel-2: #181c3a;
        --border: #2a2f55;
        --text: #f3f4f6;
        --muted: #8b93b3;

        --accent: #22d3ee;
        --accent-2: #a855f7;
        --success: #34d399;
        --warning: #fbbf24;
        --danger:  #f43f5e;
    }
}

.gradio-container, body { background: var(--bg) !important; color: var(--text) !important; }
.gradio-container { max-width: 1440px !important; }

/* Hero */
.cops-hero {
    padding: 26px 30px;
    background:
        radial-gradient(800px 220px at 0% 0%, rgba(124,58,237,0.14), transparent 60%),
        radial-gradient(700px 200px at 100% 100%, rgba(8,145,178,0.12), transparent 60%),
        linear-gradient(180deg, var(--panel-2) 0%, var(--bg) 100%);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 16px;
}
.cops-hero h1 {
    margin: 0; padding: 0;
    font-family: var(--mono);
    font-size: 2.1rem; font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 70%, var(--accent) 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.cops-tagline { margin: 6px 0 20px 0; color: var(--muted); font-size: 0.98rem; }
.cops-kpi-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
.cops-kpi {
    background: linear-gradient(180deg, var(--panel-2) 0%, var(--panel) 100%);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
}
.cops-kpi .label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
.cops-kpi .value { font-family: var(--mono); font-size: 1.65rem; margin-top: 4px; font-weight: 700; }
.cops-kpi.kpi-trained .value { color: var(--accent); }      /* cyan */
.cops-kpi.kpi-greedy  .value { color: var(--warning); }     /* amber */
.cops-kpi.kpi-improve .value { color: var(--accent-2); }    /* violet */

/* Panels */
.cops-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-top: 8px;
    /* Lock width so payload changes don't reflow neighbours */
    width: 100%;
    box-sizing: border-box;
}

/* War room - 3 cols, fixed grid (no flex reflow) */
.cops-war-wrap {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    min-height: 480px;
}
.cops-war-wrap > div { min-width: 0; } /* prevent grid blowout */
.cops-col-title {
    font-size: 0.75rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 0 0 10px 0; padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
}

/* Cards */
.cops-card {
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.cops-card-head { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.cops-name { font-weight: 600; font-size: 0.92rem; color: var(--text); }
.cops-role { font-size: 0.72rem; color: var(--muted); margin-top: 2px; }

/* Avatar with suspicion ring */
.cops-avatar {
    width: 38px; height: 38px;
    border-radius: 50%;
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-size: 0.78rem;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 2px solid var(--muted);
    flex-shrink: 0;
}
.ring-green   { border-color: var(--success); box-shadow: 0 0 0 3px rgba(52,211,153,0.28); }
.ring-amber   { border-color: var(--warning); box-shadow: 0 0 0 3px rgba(251,191,36,0.30); }
.ring-red     { border-color: var(--danger);  box-shadow: 0 0 0 3px rgba(244,63,94,0.36); }
.ring-unknown { border-style: dashed; border-color: var(--muted); }
.ring-highlight { outline: 2px solid var(--accent); outline-offset: 3px; box-shadow: 0 0 12px rgba(34,211,238,0.45); }

/* Bars and chips */
.cops-bar-label { font-size: 0.68rem; color: var(--muted); margin-top: 4px; font-family: var(--mono); }
.cops-bar-track {
    width: 100%; height: 6px;
    background: var(--bg); border-radius: 3px;
    margin-top: 4px; overflow: hidden;
}
.cops-bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%); transition: width 200ms ease; }
.cops-chips { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 4px; }
.cops-chip {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 2px 6px;
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--text);
}

/* Crisis severity */
.cops-crisis-sev {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 3px;
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 700;
    margin-left: 6px;
}
.cops-sev-high { background: rgba(244,63,94,0.18);  color: var(--danger);  border: 1px solid rgba(244,63,94,0.35); }
.cops-sev-mid  { background: rgba(251,191,36,0.18); color: var(--warning); border: 1px solid rgba(251,191,36,0.35); }
.cops-sev-low  { background: rgba(52,211,153,0.18); color: var(--success); border: 1px solid rgba(52,211,153,0.35); }
.cops-resolved {
    display: inline-block;
    margin-top: 6px;
    padding: 2px 8px;
    border-radius: 3px;
    background: rgba(52,211,153,0.22);
    color: var(--success);
    border: 1px solid rgba(52,211,153,0.4);
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.05em;
}

/* Resource meters */
.cops-meter-label { font-size: 0.68rem; color: var(--muted); font-family: var(--mono); margin-top: 10px; }
.cops-meter-track {
    position: relative;
    width: 100%; height: 8px;
    background: var(--bg); border-radius: 4px;
    margin-top: 4px; overflow: hidden;
}
.cops-meter-fill { height: 100%; transition: width 200ms ease; }
.cops-meter-marker {
    position: absolute; top: 0; bottom: 0;
    width: 1px; background: var(--muted);
    z-index: 2;
}
.cops-score-track {
    position: relative;
    width: 100%; height: 10px;
    background: var(--bg); border-radius: 5px;
    margin-top: 4px; overflow: hidden;
}
.cops-score-mid { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: var(--muted); z-index: 2; }
.cops-score-pos { position: absolute; top: 0; bottom: 0; background: var(--success); transition: width 200ms ease; }
.cops-score-neg { position: absolute; top: 0; bottom: 0; background: var(--danger);  transition: width 200ms ease; }

/* Banners */
.cops-banner-win {
    margin-top: 12px; padding: 14px 18px;
    border: 1px solid var(--success);
    color: var(--success);
    background: rgba(52,211,153,0.08);
    border-radius: 8px;
    font-family: var(--mono);
    font-weight: 700;
    text-align: center;
    letter-spacing: 0.02em;
}

/* Tables */
.cops-table {
    width: 100%; border-collapse: collapse;
    font-family: var(--mono); font-size: 0.78rem;
}
.cops-table th, .cops-table td {
    border-bottom: 1px solid var(--border);
    padding: 6px 10px; text-align: left;
    color: var(--text);
}
.cops-table th { color: var(--muted); font-weight: 600; }

/* Action log monospace */
.cops-action-log textarea { font-family: var(--mono) !important; font-size: 0.78rem !important; }
"""

# ---------------------------------------------------------------------------
# Action metadata: which params each action needs in the contextual form
# ---------------------------------------------------------------------------
ACTION_META: Dict[str, Dict[str, Any]] = {
    "query_status":             {"category": "Investigate", "cost": 0, "params": []},
    "query_member_report":      {"category": "Investigate", "cost": 0, "params": ["member"]},
    "query_observable_signals": {"category": "Investigate", "cost": 0, "params": ["member"]},
    "query_ticket":             {"category": "Investigate", "cost": 0, "params": ["task"]},
    "query_peer_opinion":       {"category": "Decide",      "cost": 1, "params": ["asked_member", "about_member"]},
    "reassign_task":            {"category": "Decide",      "cost": 1, "params": ["task", "to_member"]},
    "communicate":              {"category": "Decide",      "cost": 1, "params": ["comm_type", "comm_body"]},
    "cut_scope":                {"category": "Decide",      "cost": 1, "params": ["task", "justification"]},
    "escalate_risk":            {"category": "Decide",      "cost": 1, "params": ["crisis", "risk_text"]},
    "request_resource":         {"category": "Decide",      "cost": 1, "params": ["member"]},
    "update_timeline":          {"category": "Decide",      "cost": 1, "params": ["timeline"]},
    "consult_expert":           {"category": "Decide",      "cost": 1, "params": []},
    "force_truth":              {"category": "Heavy",       "cost": "1 + 3 PC", "params": ["member"]},
    "trigger_whistleblower":    {"category": "Heavy",       "cost": "1 + 6 PC", "params": []},
    "resolve_blocker":          {"category": "Heavy",       "cost": 2, "params": ["task", "resolution_notes"]},
    "submit_recovery_plan":     {"category": "Submit",      "cost": "terminal", "params": ["plan_summary", "timeline"]},
}
ACTION_LIST = list(ACTION_META.keys())

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
<p style="color:var(--muted);font-size:0.8rem;margin-top:8px;">
PC starts at 5. Earn: proactive_escalation_with_plan (+2), catching a liar (+3), update_timeline (+1).
Spend: force_truth (-3), whistleblower (-6).
</p>
"""

PRESET_CHOICES = [
    "Easy demo (1 liar, obvious)",
    "Realistic (2 liars, ambiguous)",
    "Hardcore (3+ liars, alliance)",
    "Random",
]
PRESET_LEVEL_SEED: Dict[str, Optional[Tuple[int, int]]] = {
    "Easy demo (1 liar, obvious)":      (1, 42),
    "Realistic (2 liars, ambiguous)":   (2, 7),
    "Hardcore (3+ liars, alliance)":    (3, 99),
    "Random":                           None,
}

# ---------------------------------------------------------------------------
# Single-user global state (matches HF Space sandbox model)
# ---------------------------------------------------------------------------
_env: Optional[CrisisOpsEnv] = None
_greedy_env: Optional[CrisisOpsEnv] = None
_greedy_policy: Optional[GreedyPMBaseline] = None
_obs: Optional[dict] = None
_history: List[str] = []
_signal_cache: Dict[str, Dict[str, int]] = {}
_action_log: List[str] = []
_last_cf_delta: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _scenario_factory(level: int) -> Callable[..., ProjectState]:
    if level == 1:
        return get_random_level1_scenario()
    if level == 2:
        return get_random_level2_scenario()
    return get_random_level3_scenario()


def _make_env(scenario_fn: Callable[..., ProjectState], curriculum_level: int) -> CrisisOpsEnv:
    return CrisisOpsEnv(
        scenario_fn=scenario_fn,
        reward_fn=counterfactual_reward,
        curriculum_level=curriculum_level,
    )


def _format_obs(obs: Optional[dict]) -> str:
    return json.dumps(obs or {}, indent=2)


def _escape(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _initials(name: str, member_id: str) -> str:
    parts = (name or "").split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    if name:
        return name[:2].upper()
    return (member_id or "?").replace("dev_", "")[:2].upper() or "?"


def _merge_signals_from_obs(obs: dict) -> None:
    """Cache signals when the action result observation includes them."""
    global _signal_cache
    if obs.get("action_type") == "query_observable_signals" and isinstance(obs.get("signals"), dict):
        mid = obs.get("member_id")
        if mid:
            _signal_cache[str(mid)] = dict(obs["signals"])


def _suspicion(reported: float, signals: Optional[Dict[str, int]]) -> Tuple[str, str]:
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


def _sync_greedy_one_step() -> None:
    global _greedy_env, _greedy_policy
    if _greedy_env is None or _greedy_policy is None:
        return
    if _greedy_env._state is None or _greedy_env._state.done:
        return
    try:
        act = _greedy_policy.act(_greedy_env._state)
        _greedy_env.step(act)
    except Exception:
        pass


def _update_cf_delta() -> None:
    global _last_cf_delta
    if _env is None or _greedy_env is None or _env._state is None or _greedy_env._state is None:
        _last_cf_delta = 0.0
        return
    try:
        _last_cf_delta = project_score(_env._state) - project_score(_greedy_env._state)
    except Exception:
        _last_cf_delta = 0.0


def _log_action_line(action_type: str, ps_delta: float, note: str = "") -> None:
    global _action_log
    line = f"{action_type:24s}  Δ {ps_delta:+.4f}"
    if note:
        line = f"{line}  | {note}"
    _action_log.append(line)
    _action_log = _action_log[-6:]


def _preset_to_level_seed(preset: str) -> Tuple[int, int]:
    spec = PRESET_LEVEL_SEED.get(preset)
    if spec is None:
        return random.choice([1, 2, 3]), random.randint(1, 2**30)
    return spec[0], spec[1]


# ---------------------------------------------------------------------------
# Hero KPI block
# ---------------------------------------------------------------------------
def _hero_kpi_block() -> str:
        blog = (
                f'<span style="margin:0 10px;color:var(--border);">|</span>'
                f'<a href="{BLOG_URL}" target="_blank" rel="noopener" '
                f'style="color:var(--accent);font-weight:600;">Writeups →</a>'
                if BLOG_URL
                else ""
        )
        return f"""
<div class="cops-hero">
  <h1>CrisisOps</h1>
  <p class="cops-tagline">Train AI agents to recover failing projects when humans lie about progress.</p>
    <p style="margin:14px 0 0 0;font-size:0.86rem;color:var(--muted);">
        <a href="{GITHUB_URL}" target="_blank" rel="noopener" style="color:var(--accent);font-weight:600;">GitHub →</a>
        {blog}
        <span style="margin:0 10px;color:var(--border);">|</span>
        <span>See the &ldquo;Technical footer&rdquo; for how-it-works, action reference, and raw observation JSON.</span>
  </p>
</div>
"""


# ---------------------------------------------------------------------------
# War room render — single HTML blob, 3-column fixed grid
# ---------------------------------------------------------------------------
def _render_war_room(obs: Optional[dict], highlight_member_id: Optional[str] = None) -> str:
    if not obs:
        return (
            '<div class="cops-panel"><div class="cops-col-title">War room</div>'
            '<p style="color:var(--muted);">Select a preset and click <strong>Start Episode</strong>.</p></div>'
        )

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
    if budget <= 2:
        budget_color = "var(--danger)"
    elif budget <= 5:
        budget_color = "var(--warning)"
    else:
        budget_color = "var(--accent)"

    pc_pct = min(100.0, max(0.0, pc / 10.0 * 100.0))

    cf = _last_cf_delta
    cf_cap = 0.25
    cf_clamped = max(-cf_cap, min(cf_cap, cf))
    if cf_clamped >= 0:
        pos_w = abs(cf_clamped) / cf_cap * 50.0
        neg_w = 0.0
    else:
        pos_w = 0.0
        neg_w = abs(cf_clamped) / cf_cap * 50.0

    # Column A: team members
    col_a_parts = ['<div class="cops-col-title">Team members</div>']
    for m in members:
        mid = str(m.get("member_id", ""))
        name = str(m.get("name", mid))
        role = str(m.get("role", ""))
        rc = float(m.get("reported_completion", 0))
        sig = _signal_cache.get(mid)
        ring, susp_lbl = _suspicion(rc, sig)
        av_extra = " ring-highlight" if highlight_member_id == mid else ""
        bar_w = min(100.0, max(0.0, rc * 100.0))

        if sig is None:
            chips = (
                '<span class="cops-chip">ticket_age: ?</span>'
                '<span class="cops-chip">commits_72h: ?</span>'
                '<span class="cops-chip">peer: ?</span>'
            )
        else:
            chips = (
                f'<span class="cops-chip">ticket_age: {_escape(sig.get("ticket_age_days", "?"))}</span>'
                f'<span class="cops-chip">commits_72h: {_escape(sig.get("commits_last_72h", "?"))}</span>'
                f'<span class="cops-chip">peer: {_escape(sig.get("peer_mentions", "?"))}</span>'
            )

        col_a_parts.append(f"""
<div class="cops-card">
  <div class="cops-card-head">
    <div class="cops-avatar {ring}{av_extra}">{_escape(_initials(name, mid))}</div>
    <div>
      <div class="cops-name">{_escape(name)}</div>
      <div class="cops-role">{_escape(role)} · {_escape(susp_lbl)}</div>
    </div>
  </div>
  <div class="cops-bar-label">reported_completion</div>
  <div class="cops-bar-track"><div class="cops-bar-fill" style="width:{bar_w:.1f}%"></div></div>
    <div style="font-family:var(--mono);font-size:0.75rem;color:var(--text);margin-top:4px;">{rc*100:.1f}%</div>
  <div class="cops-chips">{chips}</div>
</div>""")

    # Column B: crises
    col_b_parts = ['<div class="cops-col-title">Crises</div>']
    if not crises:
        col_b_parts.append('<div class="cops-card"><span style="color:var(--muted);">No active crises</span></div>')
    for c in crises:
        sev = float(c.get("severity", 0))
        resolved = bool(c.get("is_resolved"))
        title = str(c.get("crisis_type", "crisis"))
        desc = str(c.get("description", ""))[:140]
        aids = c.get("affected_task_ids") or []
        sc = _sev_class(sev)
        stamp = '<div class="cops-resolved">RESOLVED</div>' if resolved else ""
        col_b_parts.append(f"""
<div class="cops-card">
  <strong>{_escape(title)}</strong>
  <span class="cops-crisis-sev {sc}">SEV {sev:.1f}</span>
    <div style="font-size:0.75rem;color:var(--muted);margin-top:6px;">{_escape(desc)}</div>
    <div style="font-family:var(--mono);font-size:0.7rem;color:var(--muted);margin-top:8px;">tasks: {len(aids)}</div>
  {stamp}
</div>""")

    # Column C: resources & timeline
    col_c = f"""
<div class="cops-col-title">Resources &amp; timeline</div>
<div class="cops-card">
  <div class="cops-meter-label">Budget consumed (remaining {budget} / 20)</div>
  <div class="cops-meter-track">
    <div class="cops-meter-fill" style="width:{budget_pct:.1f}%;background:{budget_color}"></div>
  </div>
  <div class="cops-meter-label">Political capital ({pc:.1f} / 10)</div>
  <div class="cops-meter-track">
    <div class="cops-meter-marker" style="left:30%"></div>
    <div class="cops-meter-marker" style="left:60%"></div>
        <div class="cops-meter-fill" style="width:{pc_pct:.1f}%;background:var(--success)"></div>
  </div>
    <div style="font-size:0.65rem;color:var(--muted);font-family:var(--mono);margin-top:4px;">
    Markers: force_truth ≥3 · whistleblower ≥6
  </div>
  <div class="cops-meter-label" style="margin-top:14px;">Step {step} / {max_steps}</div>
  <div class="cops-meter-label">Δ project_score vs greedy (parallel rollout)</div>
  <div class="cops-score-track">
    <div class="cops-score-mid"></div>
    <div class="cops-score-pos" style="width:{pos_w:.1f}%;left:50%;"></div>
    <div class="cops-score-neg" style="width:{neg_w:.1f}%;right:50%;"></div>
  </div>
    <div style="font-family:var(--mono);font-size:0.78rem;color:var(--text);margin-top:6px;">
    Δ = {cf:+.3f}
  </div>
</div>"""

    return f"""
<div class="cops-panel">
  <div class="cops-war-wrap">
    <div>{"".join(col_a_parts)}</div>
    <div>{"".join(col_b_parts)}</div>
    <div>{col_c}</div>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# State-dependent dropdown choices
# ---------------------------------------------------------------------------
def _member_ids(obs: Optional[dict]) -> List[str]:
    if not obs:
        return []
    return [str(m.get("member_id")) for m in (obs.get("team_members") or [])]


def _task_ids() -> List[str]:
    if _env is None or _env._state is None:
        return []
    return [t.task_id for t in _env._state.tasks]


def _crisis_ids(obs: Optional[dict]) -> List[str]:
    if not obs:
        return []
    return [str(c.get("crisis_id")) for c in (obs.get("crises") or [])]


def _refresh_dropdown_updates() -> Dict[str, Any]:
    """One dict of gr.update() values, picked off later by name."""
    mids = _member_ids(_obs)
    tids = _task_ids()
    cids = _crisis_ids(_obs)
    return {
        "member": gr.update(choices=mids, value=mids[0] if mids else None),
        "to_member": gr.update(choices=mids, value=mids[1] if len(mids) > 1 else (mids[0] if mids else None)),
        "asked_member": gr.update(choices=mids, value=mids[0] if mids else None),
        "about_member": gr.update(choices=mids, value=mids[1] if len(mids) > 1 else (mids[0] if mids else None)),
        "task": gr.update(choices=tids, value=tids[0] if tids else None),
        "crisis": gr.update(choices=cids, value=cids[0] if cids else None),
    }


# ---------------------------------------------------------------------------
# Reset / start
# ---------------------------------------------------------------------------
def reset_episode(preset: str):
    global _env, _greedy_env, _greedy_policy, _obs, _history, _signal_cache, _action_log

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

    status = (
        f"Episode started · preset={preset} · level={level} · seed={seed} · "
        f"budget={_obs.get('budget_remaining')} · step=0"
    )
    drops = _refresh_dropdown_updates()
    return (
        _render_war_room(_obs),                         # war_html
        status,                                          # status_display
        _format_obs(_obs),                               # obs_code
        "—",                                             # reward_display
        "(no actions yet)",                              # action_log
        "",                                              # win_banner
        drops["member"],                                 # m_member
        drops["to_member"],                              # m_to_member
        drops["asked_member"],                           # m_asked
        drops["about_member"],                           # m_about
        drops["task"],                                   # d_task
        drops["crisis"],                                 # d_crisis
    )


# ---------------------------------------------------------------------------
# Action execution — single round-trip per click
# ---------------------------------------------------------------------------
def _build_params(
    action: str,
    member: Optional[str],
    to_member: Optional[str],
    asked: Optional[str],
    about: Optional[str],
    task: Optional[str],
    crisis: Optional[str],
    comm_type: str,
    comm_body: str,
    risk_text: str,
    justification: str,
    resolution_notes: str,
    timeline: str,
    plan_summary: str,
) -> Dict[str, Any]:
    if action == "query_status" or action == "consult_expert" or action == "trigger_whistleblower":
        return {}
    if action == "query_member_report" or action == "query_observable_signals":
        return {"member_id": member or ""}
    if action == "query_ticket":
        return {"task_id": task or ""}
    if action == "query_peer_opinion":
        return {"asked_member_id": asked or "", "about_member_id": about or ""}
    if action == "reassign_task":
        return {"task_id": task or "", "to_member_id": to_member or ""}
    if action == "communicate":
        return {"message_type": comm_type, "content": comm_body, "target": "both"}
    if action == "cut_scope":
        return {"task_id": task or "", "justification": justification}
    if action == "escalate_risk":
        return {"crisis_id": crisis or "", "risk_description": risk_text}
    if action == "request_resource":
        return {"resource_type": "budget", "target_member_id": member or ""}
    if action == "update_timeline":
        return {"new_completion_date": timeline, "task_estimates": {}}
    if action == "force_truth":
        return {"member_id": member or ""}
    if action == "resolve_blocker":
        return {"task_id": task or "", "resolution_notes": resolution_notes}
    if action == "submit_recovery_plan":
        return {"plan_summary": plan_summary, "risk_items": [], "timeline": timeline or "TBD"}
    return {}


def take_action(
    action: str,
    member, to_member, asked, about, task, crisis,
    comm_type, comm_body, risk_text, justification,
    resolution_notes, timeline, plan_summary,
):
    """Single-roundtrip action handler. Returns the full output tuple."""
    global _env, _obs, _history

    if _env is None or _obs is None:
        empty = _render_war_room(None)
        drops = _refresh_dropdown_updates()
        return (
            empty,
            "Click Start Episode first.",
            "{}",
            "—",
            "—",
            "",
            drops["member"], drops["to_member"], drops["asked_member"],
            drops["about_member"], drops["task"], drops["crisis"],
        )

    params = _build_params(
        action, member, to_member, asked, about, task, crisis,
        comm_type, comm_body, risk_text, justification,
        resolution_notes, timeline, plan_summary,
    )

    ps_before = project_score(_env._state) if _env._state else 0.0

    try:
        obs, reward, done, _ = _env.step({"action_type": action, "params": params})
    except Exception as e:
        drops = _refresh_dropdown_updates()
        return (
            _render_war_room(_obs),
            f"Error: {e}",
            _format_obs(_obs),
            "—",
            "\n".join(_action_log) or "—",
            "",
            drops["member"], drops["to_member"], drops["asked_member"],
            drops["about_member"], drops["task"], drops["crisis"],
        )

    _obs = obs
    _history.append(action)
    _merge_signals_from_obs(obs)
    _sync_greedy_one_step()
    _update_cf_delta()
    ps_after = project_score(_env._state) if _env._state else 0.0
    _log_action_line(action, ps_after - ps_before)

    raw = _format_obs(obs)
    if obs.get("agent_memory"):
        raw = f"// agent_memory:\n{obs['agent_memory']}\n\n{raw}"

    if done:
        if reward > 0:
            verdict = "AGENT BEAT GREEDY"
            color = "var(--success)"
        else:
            verdict = "AGENT UNDERPERFORMED GREEDY"
            color = "var(--danger)"
        banner = (
            f'<div class="cops-banner-win" style="border-color:{color};color:{color};">'
            f"EPISODE END · CF reward {reward:+.3f} vs greedy · {verdict}</div>"
        )
        status = f"DONE · CF reward {reward:+.3f} · actions={len(_history)}"
        reward_disp = f"{reward:+.3f} vs greedy"
    else:
        banner = ""
        status = (
            f"step {obs.get('current_step')} · budget {obs.get('budget_remaining')} · "
            f"PC {obs.get('political_capital')} · last={action}"
        )
        reward_disp = "—"

    drops = _refresh_dropdown_updates()
    return (
        _render_war_room(_obs),
        status,
        raw,
        reward_disp,
        "\n".join(_action_log) or "—",
        banner,
        drops["member"], drops["to_member"], drops["asked_member"],
        drops["about_member"], drops["task"], drops["crisis"],
    )


# ---------------------------------------------------------------------------
# Action picker -> contextual params: visibility toggles
# ---------------------------------------------------------------------------
def _params_visibility(action: str) -> Dict[str, bool]:
    """Return which contextual param controls should be visible for this action."""
    needed = set(ACTION_META.get(action, {}).get("params", []))
    return {
        "member_row":      "member" in needed,
        "to_member_row":   "to_member" in needed,
        "task_row":        "task" in needed,
        "crisis_row":      "crisis" in needed,
        "asked_row":       "asked_member" in needed,
        "about_row":       "about_member" in needed,
        "comm_row":        "comm_type" in needed or "comm_body" in needed,
        "risk_row":        "risk_text" in needed,
        "justification_row": "justification" in needed,
        "resolution_row":  "resolution_notes" in needed,
        "timeline_row":    "timeline" in needed,
        "plan_row":        "plan_summary" in needed,
    }


def on_action_change(action: str):
    v = _params_visibility(action)
    cost = ACTION_META.get(action, {}).get("cost", "?")
    cat = ACTION_META.get(action, {}).get("category", "?")
    info = f"**{action}**  ·  category: {cat}  ·  cost: {cost}"
    return (
        info,
        gr.update(visible=v["member_row"]),
        gr.update(visible=v["to_member_row"]),
        gr.update(visible=v["task_row"]),
        gr.update(visible=v["crisis_row"]),
        gr.update(visible=v["asked_row"]),
        gr.update(visible=v["about_row"]),
        gr.update(visible=v["comm_row"]),
        gr.update(visible=v["risk_row"]),
        gr.update(visible=v["justification_row"]),
        gr.update(visible=v["resolution_row"]),
        gr.update(visible=v["timeline_row"]),
        gr.update(visible=v["plan_row"]),
    )


# ---------------------------------------------------------------------------
# Watch mode: scripted expert trace (NOT the trained policy)
# ---------------------------------------------------------------------------
def _build_demo_trajectory(initial_obs: dict) -> List[Tuple[dict, str]]:
    tm = list(initial_obs.get("team_members") or [])
    if not tm:
        return [({"action_type": "query_status", "params": {}}, "Snapshot")]
    sorted_m = sorted(tm, key=lambda x: -float(x.get("reported_completion", 0)))
    trace: List[Tuple[dict, str]] = [
        ({"action_type": "query_status", "params": {}}, "Snapshot global status — map the crisis"),
    ]
    for m in sorted_m[:min(3, len(sorted_m))]:
        mid = m["member_id"]
        trace.append((
            {"action_type": "query_observable_signals", "params": {"member_id": mid}},
            f"Cross-verify {m.get('name', mid)} — signals vs self-report",
        ))
    top = sorted_m[0]
    tids = top.get("assigned_task_ids") or []
    others = [x for x in tm if x["member_id"] != top["member_id"]]
    if tids and others:
        low = min(others, key=lambda x: float(x.get("reported_completion", 1)))
        trace.append((
            {"action_type": "reassign_task",
             "params": {"task_id": tids[0], "to_member_id": low["member_id"]}},
            "Reassign work away from the inflated reporter",
        ))
    trace.append((
        {"action_type": "communicate",
         "params": {"message_type": "proactive_escalation_with_plan",
                    "content": "Escalating with recovery plan and verified assignments.",
                    "target": "both"}},
        "Stakeholder comms — proactive_escalation_with_plan",
    ))
    trace.append((
        {"action_type": "submit_recovery_plan",
         "params": {"plan_summary": "Deception surfaced; tasks reassigned; risks enumerated.",
                    "risk_items": [], "timeline": "2026-06-01"}},
        "Submit recovery plan — end episode",
    ))
    return trace


def play_watch_demo(preset: str):
    """Generator: replay scripted expert trace with delay (Watch mode)."""
    global _env, _greedy_env, _greedy_policy, _obs, _history, _signal_cache, _action_log

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
    cf_computed = False

    yield (
        _render_war_room(_obs),
        f"Expert trace · level={level} · seed={seed} · {len(trace)} steps",
        _format_obs(_obs),
        "—",
        "(starting…)",
        "",
    )

    for i, (act, narr) in enumerate(trace):
        time.sleep(1.2)
        ps_before = project_score(_env._state) if _env._state else 0.0
        try:
            obs, reward, done, _ = _env.step(act)
        except Exception as e:
            yield (
                _render_war_room(_obs),
                f"Error at step {i}: {e}",
                _format_obs(_obs),
                "—",
                "\n".join(_action_log) or "—",
                f'<div class="cops-banner-win" style="border-color:var(--danger);color:var(--danger);">Demo error: {_escape(str(e))}</div>',
            )
            return
        _obs = obs
        _history.append(act["action_type"])
        _merge_signals_from_obs(obs)
        _sync_greedy_one_step()
        _update_cf_delta()
        ps_after = project_score(_env._state) if _env._state else 0.0
        _log_action_line(act["action_type"], ps_after - ps_before, narr)

        mid = act.get("params", {}).get("member_id")
        hl = str(mid) if act["action_type"] == "query_observable_signals" else None

        if done:
            final_reward = reward
            cf_computed = True
            if final_reward > 0:
                verdict = "AGENT BEAT GREEDY"
                color = "var(--success)"
            else:
                verdict = "AGENT UNDERPERFORMED GREEDY"
                color = "var(--danger)"
            banner = (
                f'<div class="cops-banner-win" style="border-color:{color};color:{color};">'
                f"EPISODE END · {final_reward:+.3f} vs greedy · {verdict}</div>"
            )
        else:
            banner = ""

        yield (
            _render_war_room(_obs, highlight_member_id=hl),
            f"Step {i+1}/{len(trace)} · {narr}",
            _format_obs(obs),
            f"{reward:+.3f}" if done else "—",
            "\n".join(_action_log),
            banner,
        )
        if done:
            break

    if not cf_computed:
        banner = (
            '<div class="cops-banner-win" style="border-color:var(--muted);color:var(--muted);">'
            "Trajectory ended without submit_recovery_plan — no CF reward computed</div>"
        )
        yield (
            _render_war_room(_obs),
            "Expert trace finished (no submit)",
            _format_obs(_obs),
            "—",
            "\n".join(_action_log),
            banner,
        )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="CrisisOps — Deception Detection RL Environment",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(),
) as demo:
    gr.HTML(_hero_kpi_block())

    with gr.Row():
        preset_dd = gr.Dropdown(
            choices=PRESET_CHOICES,
            value=PRESET_CHOICES[0],
            label="Scenario preset",
            scale=3,
        )
        start_btn = gr.Button("Start Episode", variant="primary", scale=1)
        watch_btn = gr.Button("Watch Expert Trace", variant="secondary", scale=1)

    war_html = gr.HTML(_render_war_room(None))

    status_display = gr.Textbox(label="Status", lines=1, interactive=False)
    win_banner = gr.HTML("")

    # ------------------------------------------------------------------
    # Action console — single picker, contextual params
    # ------------------------------------------------------------------
    with gr.Accordion("Play yourself — action console", open=True):
        action_dd = gr.Dropdown(
            choices=ACTION_LIST,
            value="query_status",
            label="Action",
        )
        action_info = gr.Markdown("**query_status**  ·  category: Investigate  ·  cost: 0")

        # Contextual param rows — shown/hidden based on action choice
        with gr.Row(visible=False) as member_row:
            m_member = gr.Dropdown(label="member_id", choices=[], scale=1)

        with gr.Row(visible=False) as to_member_row:
            m_to_member = gr.Dropdown(label="to_member_id (reassign)", choices=[], scale=1)

        with gr.Row(visible=False) as task_row:
            d_task = gr.Dropdown(label="task_id", choices=[], scale=1)

        with gr.Row(visible=False) as crisis_row:
            d_crisis = gr.Dropdown(label="crisis_id", choices=[], scale=1)

        with gr.Row(visible=False) as asked_row:
            m_asked = gr.Dropdown(label="asked_member_id", choices=[], scale=1)
        with gr.Row(visible=False) as about_row:
            m_about = gr.Dropdown(label="about_member_id", choices=[], scale=1)

        with gr.Row(visible=False) as comm_row:
            comm_type = gr.Dropdown(
                choices=["proactive_escalation_with_plan", "risk_communication", "status_update"],
                value="proactive_escalation_with_plan",
                label="message_type",
                scale=1,
            )
            comm_body = gr.Textbox(label="content", value="Escalating with recovery plan.", scale=2)

        with gr.Row(visible=False) as risk_row:
            risk_text = gr.Textbox(label="risk_description", value="Schedule risk", scale=1)

        with gr.Row(visible=False) as justification_row:
            justification = gr.Textbox(label="justification", value="deprioritize", scale=1)

        with gr.Row(visible=False) as resolution_row:
            resolution_notes = gr.Textbox(label="resolution_notes", value="Unblocked", scale=1)

        with gr.Row(visible=False) as timeline_row:
            timeline = gr.Textbox(label="new_completion_date / timeline", value="2026-06-01", scale=1)

        with gr.Row(visible=False) as plan_row:
            plan_summary = gr.Textbox(label="plan_summary", value="Recovery plan complete.", scale=1, lines=2)

        take_btn = gr.Button("Take Action", variant="primary")
        action_log = gr.Textbox(
            label="Action log (last 6)",
            lines=8,
            interactive=False,
            elem_classes=["cops-action-log"],
        )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    with gr.Accordion("Technical footer", open=False):
        with gr.Tabs():
            with gr.Tab("How it works"):
                gr.Markdown(f"""
**Counterfactual reward** = `project_score(your final state) − project_score(greedy PM final state)`
on the same initial scenario seed. `project_score` weights recovery (0.5),
client satisfaction (0.3), and team morale (0.2) — all from *actual* state,
never from self-reports.

**GRPO** trains the policy with group-relative advantages. The trained
LoRA weights are not hosted in this Space (size); the Watch mode is a
scripted heuristic trace for illustration, not the trained policy.

Source: [GitHub]({GITHUB_URL}){'  ·  Writeups: [' + BLOG_URL + '](' + BLOG_URL + ')' if BLOG_URL else ''}
""")
            with gr.Tab("Deception rules (verbatim)"):
                gr.Code(value=DECEPTION_RULES_VERBATIM, language=None, label="From the trained policy's system prompt")
            with gr.Tab("Action reference"):
                gr.HTML(ACTION_REFERENCE_HTML)
            with gr.Tab("Raw observation JSON"):
                obs_code = gr.Code(language="json", label="Current observation")
            with gr.Tab("Reward / state"):
                reward_display = gr.Textbox(label="Counterfactual reward (episode end)", interactive=False)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    reset_outputs = [
        war_html, status_display, obs_code, reward_display,
        action_log, win_banner,
        m_member, m_to_member, m_asked, m_about, d_task, d_crisis,
    ]
    start_btn.click(reset_episode, inputs=[preset_dd], outputs=reset_outputs)

    # Action picker -> visibility of param rows
    action_dd.change(
        on_action_change,
        inputs=[action_dd],
        outputs=[
            action_info,
            member_row, to_member_row, task_row, crisis_row,
            asked_row, about_row, comm_row, risk_row,
            justification_row, resolution_row, timeline_row, plan_row,
        ],
    )

    take_inputs = [
        action_dd,
        m_member, m_to_member, m_asked, m_about, d_task, d_crisis,
        comm_type, comm_body, risk_text, justification,
        resolution_notes, timeline, plan_summary,
    ]
    take_outputs = [
        war_html, status_display, obs_code, reward_display,
        action_log, win_banner,
        m_member, m_to_member, m_asked, m_about, d_task, d_crisis,
    ]
    take_btn.click(take_action, inputs=take_inputs, outputs=take_outputs)

    # Watch mode (generator)
    watch_btn.click(
        play_watch_demo,
        inputs=[preset_dd],
        outputs=[war_html, status_display, obs_code, reward_display, action_log, win_banner],
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    inbrowser=False,
    share=False,
    show_error=True,
)