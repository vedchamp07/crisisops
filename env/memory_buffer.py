"""
env/memory_buffer.py — Long-horizon memory buffer for CrisisOps v2.

At every memory_compression_interval steps, compresses the current episode state
into a short natural-language summary stored in state.memory_buffer.

The buffer is injected into every subsequent observation under the key
"agent_memory". This forces the agent to:
1. Maintain compressed state tracking beyond the raw observation
2. Build accurate internal models of team member reliability over time
3. Reason about multi-step deception patterns across the episode

This is a genuine long-horizon planning challenge — the buffer degrades the
accuracy of the compressed state, requiring the agent to triangulate.
"""

from __future__ import annotations

from env.state import ProjectState


def should_compress(state: ProjectState) -> bool:
    """Return True if this step should trigger a memory compression."""
    if state.memory_compression_interval <= 0:
        return False
    steps_since = state.current_step - state.last_memory_compression_step
    return steps_since >= state.memory_compression_interval


def compress_state_to_memory(state: ProjectState) -> str:
    """
    Compress current episode state into a short memory buffer string.

    The compression is intentionally lossy — it captures:
    - Which members have been cross-verified and their signal patterns
    - Which crises are resolved vs active
    - Current stakeholder satisfaction trend
    - Budget and PC remaining
    - Deception suspicions accumulated so far

    Returns a natural language string suitable for injection into the obs.
    """
    lines = []

    # Budget and resource status
    lines.append(
        f"Step {state.current_step}: budget={state.budget_remaining}, "
        f"PC={round(state.political_capital, 1)}"
    )

    # Crisis status
    resolved = [c.crisis_id for c in state.crises if c.is_resolved]
    active = [c.crisis_id for c in state.crises if not c.is_resolved]
    if resolved:
        lines.append(f"Resolved crises: {', '.join(resolved)}")
    if active:
        lines.append(f"Active crises: {', '.join(active)}")

    # Member reliability assessments (based on cross-verify history)
    suspicious = []
    reliable = []
    for m in state.team_members:
        if m.times_cross_verified == 0:
            continue
        # Assess reliability from signal vs report gap
        signal_gap = m.reported_completion - m.actual_completion
        if signal_gap > 0.15:
            suspicious.append(f"{m.name}(gap={round(signal_gap, 2)})")
        elif m.caught_this_episode:
            suspicious.append(f"{m.name}(CAUGHT)")
        else:
            reliable.append(m.name)

    if suspicious:
        lines.append(f"SUSPICIOUS: {', '.join(suspicious)}")
    if reliable:
        lines.append(f"Reliable: {', '.join(reliable)}")

    # Unverified members
    unverified = [m.name for m in state.team_members if m.times_cross_verified == 0]
    if unverified:
        lines.append(f"Not yet verified: {', '.join(unverified)}")

    # Stakeholder status
    lines.append(
        f"Client satisfaction: {round(state.stakeholder.client_satisfaction, 1)}, "
        f"exec support: {round(state.stakeholder.exec_support, 1)}"
    )

    return " | ".join(lines)


def maybe_update_memory(state: ProjectState) -> bool:
    """
    Check if memory compression should fire and update state.memory_buffer.

    Returns True if compression happened this step.
    """
    if not should_compress(state):
        return False

    state.memory_buffer = compress_state_to_memory(state)
    state.last_memory_compression_step = state.current_step
    return True
