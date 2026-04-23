"""
state.py — Core dataclass definitions for CrisisOps v2.

All state objects use Python dataclasses. The candor float lives ONLY on
TeamMember and is never copied into any observation dict. Everything the
agent sees is derived from reported/signal fields.

Build order: this file is #1; every other module imports from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Candor level range constants — spec section "CANDOR SYSTEM"
# ---------------------------------------------------------------------------
HONEST_CANDOR_RANGE = (0.85, 1.0)
OPTIMISM_BIAS_RANGE = (0.50, 0.70)
SELF_PRESERVATION_RANGE = (0.10, 0.40)

# Candor level labels used across the codebase
CANDOR_LEVEL_HONEST = "honest"
CANDOR_LEVEL_OPTIMISM_BIAS = "optimism_bias"
CANDOR_LEVEL_SELF_PRESERVATION = "self_preservation"

# ---------------------------------------------------------------------------
# Budget constant — spec section "ACTION SYSTEM"
# ---------------------------------------------------------------------------
INITIAL_BUDGET = 20

# ---------------------------------------------------------------------------
# Stakeholder starting values — spec section "STAKEHOLDER STATE MACHINES"
# ---------------------------------------------------------------------------
INITIAL_CLIENT_SATISFACTION = 7.0
INITIAL_EXEC_SUPPORT = 8.0

# Client satisfaction threshold that triggers exec escalation
CLIENT_ESCALATION_THRESHOLD = 4.0

# Exec support threshold below which budget requests silently fail
EXEC_SUPPORT_BUDGET_THRESHOLD = 5.0

# Steps without communication before client satisfaction penalty applies
CLIENT_COMMUNICATION_WINDOW = 5

# Satisfaction/support change magnitudes (spec "STAKEHOLDER STATE MACHINES")
CLIENT_DECAY_NO_COMM = 0.5       # per step, no communication in window
CLIENT_DECAY_BAD_NEWS = 1.5      # bad news delivered without solution
CLIENT_GAIN_PROACTIVE = 1.0      # proactive escalation with plan
EXEC_DECAY_ESCALATION = 1.0      # per exec_escalation event
EXEC_DECAY_BUDGET_NO_TIMELINE = 0.5  # budget request without updated timeline
EXEC_GAIN_RISK_COMM = 0.5        # proactive risk communication

# Drift acknowledgement window (spec "SCHEMA DRIFT")
DRIFT_ACK_WINDOW = 3             # steps after drift fires to acknowledge

# Schema drift firing window (spec "SCHEMA DRIFT", Level 2+)
DRIFT_STEP_MIN = 6
DRIFT_STEP_MAX = 12

# ---------------------------------------------------------------------------
# Team member observable signal names — kept as constants to avoid typos
# ---------------------------------------------------------------------------
SIGNAL_TICKET_AGE_DAYS = "ticket_age_days"
SIGNAL_COMMITS_LAST_72H = "commits_last_72h"
SIGNAL_PEER_MENTIONS = "peer_mentions"


# ---------------------------------------------------------------------------
# TeamMember
# ---------------------------------------------------------------------------

@dataclass
class TeamMember:
    """
    Represents one team member with a hidden candor score.

    The ``candor`` field is the core deception mechanism. It is computed once
    per episode and is NEVER included in any observation returned to the agent.
    The agent must infer reliability by comparing reported_completion with
    observable signals (ticket_age_days, commits_last_72h, peer_mentions).
    """

    member_id: str
    name: str
    role: str

    # --- Hidden true state (never in agent obs) ---
    candor: float                     # 0.0–1.0; sampled per episode
    candor_level: str                 # CANDOR_LEVEL_* constant
    actual_completion: float          # true task completion fraction 0.0–1.0
    actual_availability: float        # true availability fraction 0.0–1.0
    actual_velocity: float            # true progress rate (tasks/step, proxy for commits)
    inflation_bias: float             # per-episode sample; positive = over-reports

    # --- Reported state (derived from candor; agent may read these) ---
    reported_completion: float        # actual + (1 - candor) * inflation_bias, clamped [0,1]
    reported_availability: float      # mirrors actual with candor-level noise

    # --- Task assignments ---
    assigned_task_ids: List[str] = field(default_factory=list)

    # --- For observable signal computation ---
    ticket_last_changed_step: int = 0    # step at which this member's ticket last changed status
    # peer_mentions is computed dynamically from task dependency chains; stored here as cache
    peer_mention_count: int = 0

    # --- Morale contribution (spec uses team_morale_avg) ---
    morale: float = 7.0               # individual morale 0.0–10.0


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """
    A single work item within the project.

    Tasks have both an actual_progress (ground truth) and may be on the
    critical path. The agent can query task state via query_ticket().
    """

    task_id: str
    title: str
    crisis_id: Optional[str]           # which crisis spawned this task, if any
    assigned_member_id: Optional[str]  # who owns it right now

    # Status: "backlog" | "in_progress" | "blocked" | "done"
    status: str = "backlog"

    is_critical_path: bool = False
    estimated_days: float = 3.0
    actual_progress: float = 0.0       # 0.0–1.0 ground truth

    # After a client_scope_change drift, a task may be deprioritized
    is_deprioritized: bool = False

    # After a regulatory_change drift, a task may be compliance-blocked
    is_compliance_blocked: bool = False

    # After a team_policy_change, review overhead added to estimates
    review_overhead_days: float = 0.0


# ---------------------------------------------------------------------------
# Crisis
# ---------------------------------------------------------------------------

@dataclass
class Crisis:
    """
    An active problem that the agent PM must resolve.

    Severity drives the Greedy PM's prioritisation (it always picks highest
    reported severity). The agent should use observable signals plus any
    consult_expert() output to make smarter choices.
    """

    crisis_id: str
    crisis_type: str          # e.g. "technical_debt", "scope_creep", "integration_failure"
    severity: float           # 0.0–10.0
    description: str
    affected_task_ids: List[str] = field(default_factory=list)
    is_resolved: bool = False
    step_detected: int = 0    # episode step when this crisis was first visible

    # Tags used by CrisisGenerator for weakness tracking
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DriftEvent
# ---------------------------------------------------------------------------

@dataclass
class DriftEvent:
    """
    A mid-episode requirement change (Level 2+).

    After firing, the next step() call includes this event in the observation.
    If the agent does not acknowledge (via update_timeline or communicate)
    within DRIFT_ACK_WINDOW steps, a stakeholder satisfaction penalty applies.
    """

    # "regulatory_change" | "client_scope_change" | "team_policy_change"
    event_type: str
    step_fired: int
    description: str
    affected_task_ids: List[str] = field(default_factory=list)
    acknowledged: bool = False
    acknowledgement_deadline: int = 0  # step_fired + DRIFT_ACK_WINDOW


# ---------------------------------------------------------------------------
# StakeholderState
# ---------------------------------------------------------------------------

@dataclass
class StakeholderState:
    """
    Tracks client and executive reactive state machines.

    The agent sees satisfaction/support levels but NOT the exact thresholds
    or internal state-machine logic (spec: "partially observable").
    """

    # Client state
    client_satisfaction: float = INITIAL_CLIENT_SATISFACTION
    client_last_communicated_step: int = -CLIENT_COMMUNICATION_WINDOW  # so first steps don't auto-penalise

    # Executive state
    exec_support: float = INITIAL_EXEC_SUPPORT
    exec_escalation_count: int = 0

    # Tracks whether agent updated timeline after a drift (used for exec penalty)
    timeline_updated_after_drift: bool = False

    # Tracks whether last budget request was accompanied by updated timeline
    last_budget_request_had_timeline: bool = False


# ---------------------------------------------------------------------------
# ProjectState
# ---------------------------------------------------------------------------

@dataclass
class ProjectState:
    """
    Complete mutable state of one CrisisOps episode.

    This is the single source of truth.  The environment's step() function
    mutates this object; reset() produces a fresh instance from a scenario
    template.  The candor float on each TeamMember is the only information
    that must NEVER flow into any observation dict.
    """

    # Simulation clock
    current_step: int = 0

    # Action budget (spec: starts at 20)
    budget_remaining: int = INITIAL_BUDGET

    # Core entities
    team_members: List[TeamMember] = field(default_factory=list)
    crises: List[Crisis] = field(default_factory=list)
    tasks: List[Task] = field(default_factory=list)

    # Stakeholder reactive agents
    stakeholder: StakeholderState = field(default_factory=StakeholderState)

    # Schema drift
    active_drift_events: List[DriftEvent] = field(default_factory=list)
    # Set by scenario/generator; None means no drift in this episode
    drift_fire_step: Optional[int] = None
    # Drift event type that fired in this episode (for fair greedy replay)
    fired_drift_type: Optional[str] = None
    # Drift fired but not yet delivered to agent in observation
    pending_drift_event: Optional[DriftEvent] = None

    # Tracking for cross_verification_rate metric
    cross_verify_calls: int = 0          # query_observable_signals calls
    total_member_query_calls: int = 0    # query_member_report + query_observable_signals

    # History
    actions_used: List[str] = field(default_factory=list)

    # Episode termination
    done: bool = False
    terminated_by_budget: bool = False  # True if budget hit 0 before submit

    # Reproducibility
    seed: Optional[int] = None

    # Curriculum level (1–4)
    curriculum_level: int = 1

    def recovery_pct(self) -> float:
        """
        Fraction of crises resolved, used by project_score().

        Returns 1.0 if there are no crises (degenerate case treated as full
        recovery so the score formula doesn't collapse).
        """
        if not self.crises:
            return 1.0
        resolved = sum(1 for c in self.crises if c.is_resolved)
        return resolved / len(self.crises)

    def team_morale_avg(self) -> float:
        """
        Average individual morale across all team members.

        Morale is stored on each TeamMember as a float 0.0–10.0.
        The project_score formula normalises this to 0.0–1.0 internally.
        """
        if not self.team_members:
            return 0.0
        return sum(m.morale for m in self.team_members) / len(self.team_members)

    def client_satisfaction_normalized(self) -> float:
        """
        Normalise client_satisfaction (0–10 scale) to 0.0–1.0 for
        the project_score formula.
        """
        return max(0.0, min(1.0, self.stakeholder.client_satisfaction / 10.0))

    def get_member(self, member_id: str) -> Optional[TeamMember]:
        """Return the TeamMember with the given id, or None."""
        for m in self.team_members:
            if m.member_id == member_id:
                return m
        return None

    def get_task(self, task_id: str) -> Optional[Task]:
        """Return the Task with the given id, or None."""
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    def get_crisis(self, crisis_id: str) -> Optional[Crisis]:
        """Return the Crisis with the given id, or None."""
        for c in self.crises:
            if c.crisis_id == crisis_id:
                return c
        return None

    def active_crises(self) -> List[Crisis]:
        """Return list of unresolved crises."""
        return [c for c in self.crises if not c.is_resolved]
