"""
scenarios/level1.py — Three Level 1 crisis templates.

Spec: "3 Level 1 crisis templates (single crisis, one deceptive member)"

Level 1 characteristics:
    - Single active crisis
    - One deceptive team member (candor will be assigned by candor.py reset)
    - No schema drift
    - 3–4 team members total

All scenario templates are parametric — the CrisisGenerator can vary
crisis severity, member availability, and descriptions without modifying
the template structure.
"""

from __future__ import annotations

import random
from typing import Callable

from env.state import (
    ProjectState,
    TeamMember,
    Task,
    Crisis,
    StakeholderState,
    CANDOR_LEVEL_HONEST,
    CANDOR_LEVEL_SELF_PRESERVATION,
    INITIAL_CLIENT_SATISFACTION,
    INITIAL_EXEC_SUPPORT,
)

# ---------------------------------------------------------------------------
# Parametric defaults for Level 1 — varied by CrisisGenerator
# ---------------------------------------------------------------------------

# FIX: 2 Slightly raise Level 1 severity to pull greedy/oracle calibration down.
# Severity range for single crisis (Level 1 is moderate difficulty)
L1_SEVERITY_MIN = 5.1
L1_SEVERITY_MAX = 8.6

# FIX: 2 Start deceptive members near-done to increase misleading self-report risk.
DECEPTIVE_ACTUAL_COMPLETION_MIN = 0.60
DECEPTIVE_ACTUAL_COMPLETION_MAX = 0.75

# Team member velocity ranges
VELOCITY_HIGH = (0.6, 0.9)
VELOCITY_LOW  = (0.05, 0.25)  # the deceptive member moves slowly

# Availability ranges
AVAIL_HIGH = (0.7, 1.0)
AVAIL_LOW  = (0.3, 0.6)


def _make_member(
    member_id: str,
    name: str,
    role: str,
    velocity_range: tuple,
    avail_range: tuple,
    task_ids: list,
    rng: random.Random,
) -> TeamMember:
    """
    Build a TeamMember with actual values sampled from given ranges.

    Candor fields are placeholders — they will be overwritten by
    initialise_member_candor() during env.reset().
    """
    velocity = rng.uniform(*velocity_range)
    avail = rng.uniform(*avail_range)
    return TeamMember(
        member_id=member_id,
        name=name,
        role=role,
        # Candor fields — overwritten by candor.py at episode start
        candor=0.9,
        candor_level=CANDOR_LEVEL_HONEST,
        inflation_bias=0.02,
        actual_completion=rng.uniform(0.1, 0.4),
        actual_availability=avail,
        actual_velocity=velocity,
        reported_completion=0.0,    # set by candor.py
        reported_availability=0.0,  # set by candor.py
        assigned_task_ids=task_ids,
        morale=rng.uniform(6.0, 8.0),
    )


# ---------------------------------------------------------------------------
# Template 1: Integration Test Failure
# ---------------------------------------------------------------------------

def scenario_integration_failure(rng: random.Random) -> ProjectState:
    """
    A critical integration test failure is blocking the release.

    One engineer (Bob) is deceptive — self-preservation candor level.
    Bob's ticket is stalled but he reports near-completion.
    Alice and Carol are honest engineers; Dave is a QA lead.
    """
    severity = rng.uniform(L1_SEVERITY_MIN, L1_SEVERITY_MAX)

    tasks = [
        Task(
            task_id="task_int_1",
            title="Fix auth service integration",
            crisis_id="crisis_int",
            assigned_member_id="dev_bob",
            status="in_progress",
            is_critical_path=True,
            estimated_days=rng.uniform(2.0, 5.0),
            actual_progress=rng.uniform(0.05, 0.25),
        ),
        Task(
            task_id="task_int_2",
            title="Update integration test suite",
            crisis_id="crisis_int",
            assigned_member_id="dev_alice",
            status="in_progress",
            is_critical_path=True,
            estimated_days=rng.uniform(1.0, 3.0),
            actual_progress=rng.uniform(0.3, 0.6),
        ),
        Task(
            task_id="task_int_3",
            title="QA regression sweep",
            crisis_id="crisis_int",
            assigned_member_id="dev_carol",
            status="backlog",
            is_critical_path=False,
            estimated_days=rng.uniform(1.0, 2.0),
            actual_progress=0.0,
        ),
    ]

    members = [
        _make_member("dev_alice", "Alice", "engineer",    VELOCITY_HIGH, AVAIL_HIGH, ["task_int_2"], rng),
        _make_member("dev_bob",   "Bob",   "engineer",    VELOCITY_LOW,  AVAIL_LOW,  ["task_int_1"], rng),
        _make_member("dev_carol", "Carol", "qa_engineer", VELOCITY_HIGH, AVAIL_HIGH, ["task_int_3"], rng),
    ]
    # FIX: 2 Bob is the deceptive member and should begin near done but stalled.
    members[1].actual_completion = rng.uniform(
        DECEPTIVE_ACTUAL_COMPLETION_MIN,
        DECEPTIVE_ACTUAL_COMPLETION_MAX,
    )

    crisis = Crisis(
        crisis_id="crisis_int",
        crisis_type="integration_failure",
        severity=severity,
        description=(
            "Critical failures in the auth service integration tests are blocking "
            "the release pipeline. Build is red."
        ),
        affected_task_ids=["task_int_1", "task_int_2", "task_int_3"],
        tags=["integration", "release_blocker"],
    )

    return ProjectState(
        team_members=members,
        tasks=tasks,
        crises=[crisis],
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(6.0, 8.0),
            exec_support=rng.uniform(7.0, 9.0),
        ),
        curriculum_level=1,
    )


# ---------------------------------------------------------------------------
# Template 2: Performance Regression
# ---------------------------------------------------------------------------

def scenario_performance_regression(rng: random.Random) -> ProjectState:
    """
    A severe performance regression was introduced in a recent deployment.

    Dave is the deceptive member — claims the profiling is almost done but
    actual velocity is near zero.
    """
    severity = rng.uniform(L1_SEVERITY_MIN, L1_SEVERITY_MAX)

    tasks = [
        Task(
            task_id="task_perf_1",
            title="Profile and identify regression source",
            crisis_id="crisis_perf",
            assigned_member_id="dev_dave",
            status="in_progress",
            is_critical_path=True,
            estimated_days=rng.uniform(2.0, 4.0),
            actual_progress=rng.uniform(0.05, 0.20),
        ),
        Task(
            task_id="task_perf_2",
            title="Implement query optimisation",
            crisis_id="crisis_perf",
            assigned_member_id="dev_eve",
            status="backlog",
            is_critical_path=True,
            estimated_days=rng.uniform(3.0, 6.0),
            actual_progress=0.0,
        ),
        Task(
            task_id="task_perf_3",
            title="Load-test validation",
            crisis_id="crisis_perf",
            assigned_member_id="dev_frank",
            status="backlog",
            is_critical_path=False,
            estimated_days=rng.uniform(1.0, 2.0),
            actual_progress=0.0,
        ),
    ]

    members = [
        _make_member("dev_dave",  "Dave",  "backend_engineer", VELOCITY_LOW,  AVAIL_LOW,  ["task_perf_1"], rng),
        _make_member("dev_eve",   "Eve",   "backend_engineer", VELOCITY_HIGH, AVAIL_HIGH, ["task_perf_2"], rng),
        _make_member("dev_frank", "Frank", "qa_lead",          VELOCITY_HIGH, AVAIL_HIGH, ["task_perf_3"], rng),
    ]
    # FIX: 2 Dave is the deceptive member and should begin near done but stalled.
    members[0].actual_completion = rng.uniform(
        DECEPTIVE_ACTUAL_COMPLETION_MIN,
        DECEPTIVE_ACTUAL_COMPLETION_MAX,
    )

    crisis = Crisis(
        crisis_id="crisis_perf",
        crisis_type="performance_regression",
        severity=severity,
        description=(
            "p95 API latency jumped from 120ms to 2.4s after last deployment. "
            "Client SLA is at risk."
        ),
        affected_task_ids=["task_perf_1", "task_perf_2", "task_perf_3"],
        tags=["performance", "sla_risk"],
    )

    return ProjectState(
        team_members=members,
        tasks=tasks,
        crises=[crisis],
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(5.5, 7.5),
            exec_support=rng.uniform(7.0, 9.0),
        ),
        curriculum_level=1,
    )


# ---------------------------------------------------------------------------
# Template 3: Data Pipeline Failure
# ---------------------------------------------------------------------------

def scenario_data_pipeline_failure(rng: random.Random) -> ProjectState:
    """
    The nightly ETL pipeline has been silently failing for 3 days.

    Grace is the deceptive member — reports the fix is nearly complete
    but actual velocity shows no progress.
    """
    severity = rng.uniform(L1_SEVERITY_MIN, L1_SEVERITY_MAX)

    tasks = [
        Task(
            task_id="task_pipe_1",
            title="Diagnose ETL failure root cause",
            crisis_id="crisis_pipe",
            assigned_member_id="dev_grace",
            status="in_progress",
            is_critical_path=True,
            estimated_days=rng.uniform(1.0, 3.0),
            actual_progress=rng.uniform(0.05, 0.20),
        ),
        Task(
            task_id="task_pipe_2",
            title="Restore missing data from backups",
            crisis_id="crisis_pipe",
            assigned_member_id="dev_henry",
            status="blocked",
            is_critical_path=True,
            estimated_days=rng.uniform(2.0, 4.0),
            actual_progress=0.0,
        ),
        Task(
            task_id="task_pipe_3",
            title="Add pipeline monitoring alerts",
            crisis_id="crisis_pipe",
            assigned_member_id="dev_irene",
            status="backlog",
            is_critical_path=False,
            estimated_days=rng.uniform(1.0, 2.0),
            actual_progress=0.0,
        ),
    ]

    members = [
        _make_member("dev_grace", "Grace",  "data_engineer",   VELOCITY_LOW,  AVAIL_LOW,  ["task_pipe_1"], rng),
        _make_member("dev_henry", "Henry",  "backend_engineer", VELOCITY_HIGH, AVAIL_HIGH, ["task_pipe_2"], rng),
        _make_member("dev_irene", "Irene",  "devops_engineer",  VELOCITY_HIGH, AVAIL_HIGH, ["task_pipe_3"], rng),
    ]
    # FIX: 2 Grace is the deceptive member and should begin near done but stalled.
    members[0].actual_completion = rng.uniform(
        DECEPTIVE_ACTUAL_COMPLETION_MIN,
        DECEPTIVE_ACTUAL_COMPLETION_MAX,
    )

    crisis = Crisis(
        crisis_id="crisis_pipe",
        crisis_type="data_pipeline_failure",
        severity=severity,
        description=(
            "Nightly ETL pipeline has silently failed for 72 hours. "
            "Reporting dashboards are 3 days stale. Client noticed."
        ),
        affected_task_ids=["task_pipe_1", "task_pipe_2", "task_pipe_3"],
        tags=["data", "etl", "client_facing"],
    )

    return ProjectState(
        team_members=members,
        tasks=tasks,
        crises=[crisis],
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(4.5, 6.5),
            exec_support=rng.uniform(6.0, 8.0),
        ),
        curriculum_level=1,
    )


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

LEVEL1_SCENARIOS = [
    scenario_integration_failure,
    scenario_performance_regression,
    scenario_data_pipeline_failure,
]


def get_random_level1_scenario() -> Callable[[random.Random], ProjectState]:
    """
    Return a randomly chosen Level 1 scenario factory.

    The CrisisGenerator calls this to sample episodes.  The factory itself
    is parametric — each call with a fresh rng produces varied parameters.
    """
    import random as _random
    idx = _random.randint(0, len(LEVEL1_SCENARIOS) - 1)
    return LEVEL1_SCENARIOS[idx]
