"""
scenarios/level3.py — Level 3 crisis templates.

Spec: "Level 3 templates (cascading, adversarial majority)"

Level 3 characteristics:
    - Cascading crises (one crisis triggers another)
    - Adversarial majority: most team members are deceptive
    - 5 team members, 3+ crises
    - Schema drift active
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
)
from scenarios.level1 import _make_member, VELOCITY_HIGH, VELOCITY_LOW, AVAIL_HIGH, AVAIL_LOW

L3_SEVERITY_MIN = 7.0
L3_SEVERITY_MAX = 9.5
VELOCITY_MID = (0.25, 0.50)


def scenario_cascading_infra(rng: random.Random) -> ProjectState:
    """
    Infrastructure outage cascades into data loss and client SLA breach.

    Adversarial majority: 3 of 5 members are deceptive.
    Crisis 3 (SLA breach) unlocks only after crisis 1 is detected.
    """
    s1 = rng.uniform(L3_SEVERITY_MIN, L3_SEVERITY_MAX)
    s2 = rng.uniform(L3_SEVERITY_MIN - 1, L3_SEVERITY_MAX)
    s3 = rng.uniform(6.5, 8.5)

    tasks = [
        Task("t3a_1", "Restore primary storage cluster",      "c3a_infra",  "dev_v1",
             "in_progress", True,  rng.uniform(3, 6), rng.uniform(0.05, 0.15)),
        Task("t3a_2", "Emergency backup restore procedure",   "c3a_infra",  "dev_v2",
             "blocked",     True,  rng.uniform(4, 7), 0.0),
        Task("t3a_3", "Recover corrupted data records",       "c3a_data",   "dev_v3",
             "in_progress", True,  rng.uniform(3, 5), rng.uniform(0.05, 0.20)),
        Task("t3a_4", "Run integrity verification suite",     "c3a_data",   "dev_v4",
             "backlog",     False, rng.uniform(2, 4), 0.0),
        Task("t3a_5", "Prepare SLA breach report for client", "c3a_sla",    "dev_v5",
             "in_progress", True,  rng.uniform(1, 3), rng.uniform(0.10, 0.30)),
        Task("t3a_6", "Negotiate penalty waiver",             "c3a_sla",    "dev_v1",
             "backlog",     False, rng.uniform(1, 2), 0.0),
    ]

    # 3 of 5 deceptive (v1, v3, v5 have low velocity)
    members = [
        _make_member("dev_v1", "Vera",    "devops_engineer",    VELOCITY_LOW,  AVAIL_LOW,  ["t3a_1", "t3a_6"], rng),
        _make_member("dev_v2", "Will",    "backend_engineer",   VELOCITY_HIGH, AVAIL_HIGH, ["t3a_2"],          rng),
        _make_member("dev_v3", "Xena",    "data_engineer",      VELOCITY_LOW,  AVAIL_LOW,  ["t3a_3"],          rng),
        _make_member("dev_v4", "Yusuf",   "qa_engineer",        VELOCITY_HIGH, AVAIL_HIGH, ["t3a_4"],          rng),
        _make_member("dev_v5", "Zoe",     "fullstack_engineer", VELOCITY_LOW,  AVAIL_LOW,  ["t3a_5"],          rng),
    ]

    crises = [
        Crisis("c3a_infra", "infrastructure_outage",  s1,
               "Primary storage cluster down; 60% write failure rate.",
               ["t3a_1", "t3a_2"], tags=["infra", "cascading"]),
        Crisis("c3a_data",  "data_corruption",        s2,
               "Storage failure caused partial data corruption across 3 shards.",
               ["t3a_3", "t3a_4"], tags=["data", "cascading"]),
        Crisis("c3a_sla",   "sla_breach",             s3,
               "Client SLA violated; 4-hour response window expires in 2h.",
               ["t3a_5", "t3a_6"], tags=["client_facing", "cascading"]),
    ]

    return ProjectState(
        team_members=members, tasks=tasks, crises=crises,
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(3.5, 5.5),
            exec_support=rng.uniform(5.5, 7.5),
        ),
        curriculum_level=3,
    )


def scenario_adversarial_majority(rng: random.Random) -> ProjectState:
    """
    Security incident + technical debt explosion with adversarial majority.

    4 of 5 team members are deceptive.  The agent must identify the one
    honest member to rely on for critical tasks.
    """
    s1 = rng.uniform(L3_SEVERITY_MIN, L3_SEVERITY_MAX)
    s2 = rng.uniform(L3_SEVERITY_MIN, L3_SEVERITY_MAX)
    s3 = rng.uniform(6.0, 8.5)

    tasks = [
        Task("t3b_1",  "Patch SQL injection vulnerability",  "c3b_sec",  "dev_am1",
             "in_progress", True,  rng.uniform(2, 4), rng.uniform(0.05, 0.15)),
        Task("t3b_2",  "Audit all user-facing queries",      "c3b_sec",  "dev_am2",
             "blocked",     True,  rng.uniform(3, 6), 0.0),
        Task("t3b_3",  "Refactor legacy auth module",        "c3b_tech", "dev_am3",
             "in_progress", True,  rng.uniform(5, 9), rng.uniform(0.05, 0.20)),
        Task("t3b_4",  "Remove deprecated API endpoints",    "c3b_tech", "dev_am4",
             "in_progress", False, rng.uniform(3, 5), rng.uniform(0.05, 0.20)),
        Task("t3b_5",  "Update dependency versions",         "c3b_dep",  "dev_am5",
             "backlog",     False, rng.uniform(1, 3), 0.0),
        Task("t3b_6",  "Run security penetration test",      "c3b_sec",  "dev_am1",
             "backlog",     True,  rng.uniform(2, 4), 0.0),
    ]

    # 4 of 5 deceptive
    members = [
        _make_member("dev_am1", "Aaron",  "security_engineer",  VELOCITY_LOW,  AVAIL_LOW,  ["t3b_1", "t3b_6"], rng),
        _make_member("dev_am2", "Bella",  "backend_engineer",   VELOCITY_LOW,  AVAIL_LOW,  ["t3b_2"],          rng),
        _make_member("dev_am3", "Carlos", "fullstack_engineer", VELOCITY_LOW,  AVAIL_LOW,  ["t3b_3"],          rng),
        _make_member("dev_am4", "Diana",  "backend_engineer",   VELOCITY_HIGH, AVAIL_HIGH, ["t3b_4"],          rng),  # honest
        _make_member("dev_am5", "Ethan",  "devops_engineer",    VELOCITY_LOW,  AVAIL_LOW,  ["t3b_5"],          rng),
    ]

    crises = [
        Crisis("c3b_sec",  "security_vulnerability", s1,
               "SQL injection found in user search endpoint; CVE filed.",
               ["t3b_1", "t3b_2", "t3b_6"], tags=["security", "cve"]),
        Crisis("c3b_tech", "technical_debt",         s2,
               "Legacy auth module causing cascading failures in 3 services.",
               ["t3b_3", "t3b_4"], tags=["technical_debt", "cascading"]),
        Crisis("c3b_dep",  "dependency_conflict",    s3,
               "12 critical dependencies out of date; 3 have known CVEs.",
               ["t3b_5"], tags=["security", "dependencies"]),
    ]

    return ProjectState(
        team_members=members, tasks=tasks, crises=crises,
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(3.0, 5.0),
            exec_support=rng.uniform(5.0, 7.0),
        ),
        curriculum_level=3,
    )


def scenario_cascading_release_failure(rng: random.Random) -> ProjectState:
    """
    Failed release triggers regression, which triggers client escalation.

    Adversarial majority: 3 of 5 deceptive.
    """
    s1 = rng.uniform(L3_SEVERITY_MIN, L3_SEVERITY_MAX)
    s2 = rng.uniform(L3_SEVERITY_MIN, L3_SEVERITY_MAX)
    s3 = rng.uniform(7.0, 9.5)

    tasks = [
        Task("t3c_1",  "Roll back failed deployment",        "c3c_rel", "dev_f1",
             "in_progress", True,  rng.uniform(1, 3), rng.uniform(0.05, 0.25)),
        Task("t3c_2",  "Fix root cause in release pipeline", "c3c_rel", "dev_f2",
             "in_progress", True,  rng.uniform(3, 6), rng.uniform(0.05, 0.15)),
        Task("t3c_3",  "Identify regression source",        "c3c_reg", "dev_f3",
             "in_progress", True,  rng.uniform(2, 4), rng.uniform(0.05, 0.20)),
        Task("t3c_4",  "Patch regression bug",               "c3c_reg", "dev_f4",
             "blocked",     True,  rng.uniform(2, 5), 0.0),
        Task("t3c_5",  "Draft client incident report",       "c3c_esc", "dev_f5",
             "in_progress", True,  rng.uniform(1, 2), rng.uniform(0.10, 0.30)),
        Task("t3c_6",  "Schedule exec review call",          "c3c_esc", "dev_f1",
             "backlog",     False, rng.uniform(0.5, 1), 0.0),
    ]

    members = [
        _make_member("dev_f1", "Fiona",  "devops_engineer",   VELOCITY_LOW,  AVAIL_LOW,  ["t3c_1", "t3c_6"], rng),
        _make_member("dev_f2", "George", "backend_engineer",  VELOCITY_HIGH, AVAIL_HIGH, ["t3c_2"],           rng),
        _make_member("dev_f3", "Hannah", "backend_engineer",  VELOCITY_LOW,  AVAIL_LOW,  ["t3c_3"],           rng),
        _make_member("dev_f4", "Ivan",   "qa_engineer",       VELOCITY_HIGH, AVAIL_HIGH, ["t3c_4"],           rng),
        _make_member("dev_f5", "Julia",  "pm_engineer",       VELOCITY_LOW,  AVAIL_LOW,  ["t3c_5"],           rng),
    ]

    crises = [
        Crisis("c3c_rel", "release_failure",       s1,
               "Production deployment failed; service down for 45 minutes.",
               ["t3c_1", "t3c_2"], tags=["release", "cascading"]),
        Crisis("c3c_reg", "test_regression",       s2,
               "Post-rollback regression in payment module identified.",
               ["t3c_3", "t3c_4"], tags=["regression", "payments"]),
        Crisis("c3c_esc", "client_escalation",     s3,
               "Client CEO escalated directly to our CTO; demanding post-mortem.",
               ["t3c_5", "t3c_6"], tags=["client_facing", "executive"]),
    ]

    return ProjectState(
        team_members=members, tasks=tasks, crises=crises,
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(2.5, 4.5),
            exec_support=rng.uniform(4.5, 6.5),
        ),
        curriculum_level=3,
    )


LEVEL3_SCENARIOS = [
    scenario_cascading_infra,
    scenario_adversarial_majority,
    scenario_cascading_release_failure,
]


def get_random_level3_scenario() -> Callable[[random.Random], ProjectState]:
    """Return a randomly chosen Level 3 scenario factory."""
    import random as _random
    idx = _random.randint(0, len(LEVEL3_SCENARIOS) - 1)
    return LEVEL3_SCENARIOS[idx]
