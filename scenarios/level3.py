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
    CANDOR_LEVEL_HONEST,
    CANDOR_LEVEL_SELF_PRESERVATION,
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


def scenario_morale_collapse(rng: random.Random) -> ProjectState:
    """
    Low team morale compresses honest productivity signals toward deceptive range.

    The agent must combine multiple cues instead of relying on commit count alone.
    """
    severity = rng.uniform(L3_SEVERITY_MIN, L3_SEVERITY_MAX)

    tasks = [
        Task("t3m_1", "Stabilise delayed integration bridge", "c3m_rel", "dev_mc_1",
             "in_progress", True, rng.uniform(2, 5), rng.uniform(0.05, 0.15)),
        Task("t3m_2", "Unblock partner schema handoff", "c3m_rel", "dev_mc_2",
             "in_progress", True, rng.uniform(2, 5), rng.uniform(0.20, 0.40)),
        Task("t3m_3", "Run constrained regression sweep", "c3m_rel", "dev_mc_3",
             "in_progress", False, rng.uniform(1, 3), rng.uniform(0.20, 0.40)),
    ]

    members = [
        _make_member("dev_mc_1", "Marta", "backend_engineer", VELOCITY_LOW, AVAIL_LOW, ["t3m_1"], rng),
        _make_member("dev_mc_2", "Nikhil", "data_engineer", VELOCITY_MID, AVAIL_HIGH, ["t3m_2", "mc_dep_1"], rng),
        _make_member("dev_mc_3", "Olivia", "qa_engineer", VELOCITY_MID, AVAIL_HIGH, ["t3m_3", "mc_dep_1"], rng),
    ]
    # Tests detection robustness when honest signals are degraded
    members[0].candor_level = CANDOR_LEVEL_SELF_PRESERVATION
    members[1].candor_level = CANDOR_LEVEL_HONEST
    members[2].candor_level = CANDOR_LEVEL_HONEST
    members[1].actual_velocity = rng.uniform(0.34, 0.49)  # honest commits compress to 2-3
    members[2].actual_velocity = rng.uniform(0.34, 0.49)
    for member in members:
        member.morale = 2.5

    crises = [
        Crisis("c3m_rel", "release_stability", severity,
               "Low morale has slowed execution and increased ambiguity in progress signals.",
               ["t3m_1", "t3m_2", "t3m_3"], tags=["morale", "signal_overlap"]),
    ]

    return ProjectState(
        team_members=members,
        tasks=tasks,
        crises=crises,
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(3.5, 5.5),
            exec_support=rng.uniform(5.0, 7.0),
        ),
        curriculum_level=3,
    )


def scenario_cascading_crises(rng: random.Random) -> ProjectState:
    """
    Two simultaneous crises share one honest member across both workstreams.

    This creates resource contention with less budget slack.
    """
    s1 = rng.uniform(L3_SEVERITY_MIN, L3_SEVERITY_MAX)
    s2 = rng.uniform(L3_SEVERITY_MIN - 1, L3_SEVERITY_MAX)

    tasks = [
        Task("t3cc_1", "Restore failed integration handshake", "c3cc_int", "dev_cc_1",
             "in_progress", True, rng.uniform(2, 5), rng.uniform(0.05, 0.15)),
        Task("t3cc_2", "Patch integration rollback guardrails", "c3cc_int", "dev_cc_2",
             "in_progress", True, rng.uniform(2, 4), rng.uniform(0.20, 0.40)),
        Task("t3cc_3", "Repair pipeline checkpoint drift", "c3cc_data", "dev_cc_2",
             "in_progress", True, rng.uniform(2, 5), rng.uniform(0.20, 0.40)),
        Task("t3cc_4", "Backfill delayed warehouse partitions", "c3cc_data", "dev_cc_3",
             "blocked", True, rng.uniform(2, 5), 0.0),
    ]

    members = [
        _make_member("dev_cc_1", "Pavel", "integration_engineer", VELOCITY_LOW, AVAIL_LOW, ["t3cc_1"], rng),
        _make_member("dev_cc_2", "Quincy", "backend_engineer", VELOCITY_HIGH, AVAIL_HIGH, ["t3cc_2", "t3cc_3"], rng),
        _make_member("dev_cc_3", "Rhea", "data_engineer", VELOCITY_HIGH, AVAIL_HIGH, ["t3cc_4"], rng),
    ]
    members[0].candor_level = CANDOR_LEVEL_SELF_PRESERVATION
    members[1].candor_level = CANDOR_LEVEL_HONEST
    members[2].candor_level = CANDOR_LEVEL_HONEST

    crises = [
        Crisis("c3cc_int", "integration_failure", s1,
               "Core integration handshake is failing and rollback safety is incomplete.",
               ["t3cc_1", "t3cc_2"], tags=["integration", "resource_contention"]),
        Crisis("c3cc_data", "data_pipeline_failure", s2,
               "Pipeline checkpoint drift caused partition lag and downstream stale reporting.",
               ["t3cc_3", "t3cc_4"], tags=["data", "resource_contention"]),
    ]

    # Tests prioritization under resource contention
    return ProjectState(
        budget_remaining=15,
        team_members=members,
        tasks=tasks,
        crises=crises,
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(3.5, 5.5),
            exec_support=rng.uniform(5.0, 7.0),
        ),
        curriculum_level=3,
    )


def scenario_trust_reversal(rng: random.Random) -> ProjectState:
    """
    One initially reliable member has high completed load but low remaining
    capacity, while the rest present deceptive profiles.
    """
    s1 = rng.uniform(L3_SEVERITY_MIN, L3_SEVERITY_MAX)
    s2 = rng.uniform(L3_SEVERITY_MIN - 1, L3_SEVERITY_MAX)

    tasks = [
        Task("t3tr_1", "Finalize release hotfix chain", "c3tr_rel", "dev_tr_1",
             "in_progress", True, rng.uniform(2, 4), 0.65),
        Task("t3tr_2", "Repair auth fallback path", "c3tr_rel", "dev_tr_2",
             "in_progress", True, rng.uniform(2, 5), rng.uniform(0.05, 0.20)),
        Task("t3tr_3", "Stabilize job queue retries", "c3tr_ops", "dev_tr_3",
             "in_progress", True, rng.uniform(2, 5), rng.uniform(0.05, 0.20)),
        Task("t3tr_4", "Validate data consistency patch", "c3tr_ops", "dev_tr_4",
             "blocked", True, rng.uniform(2, 5), 0.0),
        Task("t3tr_5", "Draft stakeholder recovery timeline", "c3tr_ops", "dev_tr_1",
             "backlog", False, rng.uniform(1, 2), 0.0),
    ]

    members = [
        _make_member("dev_tr_1", "Sana", "backend_engineer", VELOCITY_HIGH, AVAIL_LOW, ["t3tr_1", "t3tr_5"], rng),
        _make_member("dev_tr_2", "Tariq", "backend_engineer", VELOCITY_LOW, AVAIL_LOW, ["t3tr_2"], rng),
        _make_member("dev_tr_3", "Umair", "devops_engineer", VELOCITY_LOW, AVAIL_LOW, ["t3tr_3"], rng),
        _make_member("dev_tr_4", "Violet", "qa_engineer", VELOCITY_LOW, AVAIL_LOW, ["t3tr_4"], rng),
    ]
    # Approximates trust reversal: high-velocity but low remaining
    # capacity. True mid-episode state change requires env/state.py support —
    # see pending_state_changes TODO if that is added later.
    members[0].actual_completion = 0.65
    members[0].actual_availability = 0.30
    members[0].reported_availability = 0.30
    members[0].candor_level = CANDOR_LEVEL_HONEST
    members[1].candor_level = CANDOR_LEVEL_SELF_PRESERVATION
    members[2].candor_level = CANDOR_LEVEL_SELF_PRESERVATION
    members[3].candor_level = CANDOR_LEVEL_SELF_PRESERVATION

    crises = [
        Crisis("c3tr_rel", "release_failure", s1,
               "Release recovery depends on a previously reliable owner with limited remaining capacity.",
               ["t3tr_1", "t3tr_2"], tags=["trust", "load_balance"]),
        Crisis("c3tr_ops", "operations_instability", s2,
               "Ops instability persists while three low-reliability owners report optimistic progress.",
               ["t3tr_3", "t3tr_4", "t3tr_5"], tags=["trust", "consult_expert"]),
    ]

    return ProjectState(
        team_members=members,
        tasks=tasks,
        crises=crises,
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(3.0, 5.0),
            exec_support=rng.uniform(5.0, 7.0),
        ),
        curriculum_level=3,
    )


LEVEL3_SCENARIOS = [
    scenario_cascading_infra,
    scenario_adversarial_majority,
    scenario_cascading_release_failure,
    scenario_morale_collapse,
    scenario_cascading_crises,
    scenario_trust_reversal,
]


def get_random_level3_scenario() -> Callable[[random.Random], ProjectState]:
    """Return a randomly chosen Level 3 scenario factory."""
    import random as _random
    idx = _random.randint(0, len(LEVEL3_SCENARIOS) - 1)
    return LEVEL3_SCENARIOS[idx]
