"""
candor.py — Candor score system, deception mechanics, and observable signal generation.

The candor system is the core novel mechanism of CrisisOps v2.  Each TeamMember
gets a hidden candor float sampled once per episode.  The reported completion
diverges from actual completion by an amount proportional to (1 - candor).
Observable signals are computed strictly from actual state so that a clever agent
can detect deception by cross-referencing reports against signals.

Spec reference: "CANDOR SYSTEM (candor.py)"
"""

from __future__ import annotations

import random
from typing import Dict, Tuple

from env.state import (
    TeamMember,
    ProjectState,
    CANDOR_LEVEL_HONEST,
    CANDOR_LEVEL_OPTIMISM_BIAS,
    CANDOR_LEVEL_SELF_PRESERVATION,
    HONEST_CANDOR_RANGE,
    OPTIMISM_BIAS_RANGE,
    SELF_PRESERVATION_RANGE,
    SIGNAL_TICKET_AGE_DAYS,
    SIGNAL_COMMITS_LAST_72H,
    SIGNAL_PEER_MENTIONS,
)

# ---------------------------------------------------------------------------
# Inflation bias distribution parameters per candor level
# (spec: "inflation_bias is sampled per episode from a distribution appropriate
#  to the candor level")
# ---------------------------------------------------------------------------

# (mean, std) for each level's inflation bias Normal distribution
# Honest members have near-zero bias; self-preservation members inflate heavily.
INFLATION_BIAS_PARAMS: Dict[str, Tuple[float, float]] = {
    CANDOR_LEVEL_HONEST:             (0.02, 0.03),   # tiny positive drift
    CANDOR_LEVEL_OPTIMISM_BIAS:      (0.20, 0.08),   # moderate positive inflation
    # self_preservation: mean=0.50 chosen so that with candor=0.1,
    # P(reported > actual + 0.30) >= 90% (spec test requirement).
    # Math: reported = actual + 0.9*bias; need 0.9*bias > 0.30 → bias > 0.333
    # With N(0.50, 0.10): P(bias > 0.333) = P(Z > -1.67) ≈ 0.952 ✓
    CANDOR_LEVEL_SELF_PRESERVATION:  (0.50, 0.10),   # heavy inflation
}

# Clamp bounds for inflation_bias to keep reported values sane
INFLATION_BIAS_MIN = 0.0
INFLATION_BIAS_MAX = 0.95

# Clamp bounds for reported values
REPORTED_COMPLETION_MIN = 0.0
REPORTED_COMPLETION_MAX = 1.0

# Availability noise std per candor level (reported_availability ≈ actual + noise)
AVAILABILITY_NOISE_STD: Dict[str, float] = {
    CANDOR_LEVEL_HONEST:            0.03,
    CANDOR_LEVEL_OPTIMISM_BIAS:     0.10,
    CANDOR_LEVEL_SELF_PRESERVATION: 0.20,
}

# ---------------------------------------------------------------------------
# Observable signal scaling constants
# (all derived from actual state — spec is explicit about this)
# ---------------------------------------------------------------------------

# Maximum ticket age in days returned as a signal (capped for bounded obs space)
MAX_TICKET_AGE_DAYS = 30

# Steps per "day" for simulated ticket age (1 step ≈ half a working day)
STEPS_PER_DAY = 2

# Scale factor: actual_velocity maps to expected commits in a 72h window
# If actual_velocity == 1.0 (full speed), expected commits = COMMITS_FULL_VELOCITY
COMMITS_FULL_VELOCITY = 6   # realistic sprint commit cadence for one dev over 3 days

# Stall threshold: actual completion must advance by at least this per step
# to count as "not stalled"
VELOCITY_STALL_THRESHOLD = 0.01


def sample_candor_level(rng: random.Random) -> str:
    """
    Pick a candor level label uniformly at random.

    Equal probability for all three levels ensures that the environment is
    challenging and diverse across episodes.  The CrisisGenerator can later
    skew this distribution toward levels the agent handles poorly.
    """
    return rng.choice([
        CANDOR_LEVEL_HONEST,
        CANDOR_LEVEL_OPTIMISM_BIAS,
        CANDOR_LEVEL_SELF_PRESERVATION,
    ])


def sample_candor_float(level: str, rng: random.Random) -> float:
    """
    Sample the hidden candor float uniformly from the range for ``level``.

    Spec ranges:
        honest:             0.85–1.0
        optimism_bias:      0.50–0.70
        self_preservation:  0.10–0.40
    """
    ranges = {
        CANDOR_LEVEL_HONEST:            HONEST_CANDOR_RANGE,
        CANDOR_LEVEL_OPTIMISM_BIAS:     OPTIMISM_BIAS_RANGE,
        CANDOR_LEVEL_SELF_PRESERVATION: SELF_PRESERVATION_RANGE,
    }
    lo, hi = ranges[level]
    return rng.uniform(lo, hi)


def sample_inflation_bias(level: str, rng: random.Random) -> float:
    """
    Sample the per-episode inflation bias from the distribution for ``level``.

    Uses a Normal distribution parameterised per level (see INFLATION_BIAS_PARAMS).
    Result is clamped to [INFLATION_BIAS_MIN, INFLATION_BIAS_MAX] so the
    reported_completion formula stays numerically sane.
    """
    mean, std = INFLATION_BIAS_PARAMS[level]
    bias = rng.gauss(mean, std)
    return max(INFLATION_BIAS_MIN, min(INFLATION_BIAS_MAX, bias))


def compute_reported_completion(actual: float, candor: float, inflation_bias: float) -> float:
    """
    Core deception formula from the spec:

        reported = actual + (1 - candor) * inflation_bias

    Clamped to [0.0, 1.0] so the agent always receives a valid fraction.
    The gap between reported and actual grows as candor drops.
    """
    reported = actual + (1.0 - candor) * inflation_bias
    return max(REPORTED_COMPLETION_MIN, min(REPORTED_COMPLETION_MAX, reported))


def compute_reported_availability(
    actual: float, level: str, rng: random.Random
) -> float:
    """
    Add candor-level noise to actual availability for the reported figure.

    Honest members report near-truth; self-preservation members may claim to
    be more available than they are (positive noise skew).
    """
    std = AVAILABILITY_NOISE_STD[level]
    # Self-preservation members tend to over-report availability
    if level == CANDOR_LEVEL_SELF_PRESERVATION:
        noise = abs(rng.gauss(0.15, std))   # positive skew
    else:
        noise = rng.gauss(0.0, std)
    reported = actual + noise
    return max(0.0, min(1.0, reported))


def initialise_member_candor(member: TeamMember, rng: random.Random) -> None:
    """
    Populate all candor-derived fields on a TeamMember in place.

    Called once per episode during reset().  After this call the member is
    ready for signal computation and reported-value queries.

    Mutates: member.candor, member.candor_level, member.inflation_bias,
             member.reported_completion, member.reported_availability.
    """
    level = sample_candor_level(rng)
    candor = sample_candor_float(level, rng)
    bias = sample_inflation_bias(level, rng)

    member.candor_level = level
    member.candor = candor
    member.inflation_bias = bias
    member.reported_completion = compute_reported_completion(
        member.actual_completion, candor, bias
    )
    member.reported_availability = compute_reported_availability(
        member.actual_availability, level, rng
    )


def refresh_reported_values(member: TeamMember) -> None:
    """
    Recompute reported_completion from current actual_completion.

    Called after the environment advances actual_completion each step.
    The candor and inflation_bias are fixed for the episode; only the
    actual progress changes.
    """
    member.reported_completion = compute_reported_completion(
        member.actual_completion, member.candor, member.inflation_bias
    )


# ---------------------------------------------------------------------------
# Observable signal computation — MUST use actual state only
# ---------------------------------------------------------------------------

def compute_ticket_age_days(member: TeamMember, current_step: int) -> int:
    """
    Compute ticket_age_days from ACTUAL state.

    Defined as the number of simulated days since the member's ticket last
    changed status, proxied by ticket_last_changed_step.  A stalled member
    (actual_velocity near zero) will have a high ticket age.

    Spec: "ticket_age_days: days since ticket last changed status — derived
    from actual completion velocity"
    """
    steps_since_change = max(0, current_step - member.ticket_last_changed_step)
    days = steps_since_change // STEPS_PER_DAY
    return min(days, MAX_TICKET_AGE_DAYS)


def compute_commits_last_72h(member: TeamMember) -> int:
    """
    Compute commits_last_72h from ACTUAL state.

    If actual_velocity is at or below the stall threshold, returns 0.
    Otherwise scales proportionally to COMMITS_FULL_VELOCITY.

    Spec: "commits_last_72h: integer count — 0 if actual completion stalled,
    proportional to actual progress"
    """
    if member.actual_velocity <= VELOCITY_STALL_THRESHOLD:
        return 0
    raw = member.actual_velocity * COMMITS_FULL_VELOCITY
    return max(0, round(raw))


def compute_peer_mentions(member: TeamMember, state: ProjectState) -> int:
    """
    Count how often this member appears in other members' assigned task lists
    as a dependency proxy.

    Spec: "peer_mentions: count of how often this member appears in other
    members' dependency chains"

    Implementation: for each task assigned to OTHER members, count how many
    of those tasks are also in this member's assigned_task_ids (shared tasks
    indicate dependency coupling).  Capped at 10 to bound the obs space.
    """
    own_task_set = set(member.assigned_task_ids)
    count = 0
    for other in state.team_members:
        if other.member_id == member.member_id:
            continue
        for tid in other.assigned_task_ids:
            if tid in own_task_set:
                count += 1
    return min(count, 10)


def get_observable_signals(member: TeamMember, state: ProjectState) -> Dict[str, int]:
    """
    Return the full observable signal dict for a TeamMember.

    This is the ONLY function that should be called to build the signals
    returned by query_observable_signals().  It never touches reported state
    or the candor float — only actual_velocity, ticket_last_changed_step, and
    task assignments.

    Returns a dict with keys: ticket_age_days, commits_last_72h, peer_mentions.
    """
    return {
        SIGNAL_TICKET_AGE_DAYS:    compute_ticket_age_days(member, state.current_step),
        SIGNAL_COMMITS_LAST_72H:   compute_commits_last_72h(member),
        SIGNAL_PEER_MENTIONS:      compute_peer_mentions(member, state),
    }


def update_ticket_change_step(member: TeamMember, current_step: int) -> None:
    """
    Mark that this member's ticket changed status at current_step.

    Should be called whenever actual_completion crosses a meaningful threshold
    (e.g., task moves to done) or when a task is reassigned.
    """
    member.ticket_last_changed_step = current_step
