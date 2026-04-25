"""
scenarios/level2.py — Level 2 crisis templates.

Spec: "Level 2 templates (double crisis, two deceptive members, schema drift)"

Level 2 characteristics:
    - Two simultaneous active crises
    - Two deceptive team members
    - Schema drift fires at a random step 6–12
    - 4–5 team members
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

# Severity range for Level 2 (harder than Level 1)
L2_SEVERITY_MIN = 6.0
L2_SEVERITY_MAX = 9.0


def scenario_double_crisis_auth_perf(rng: random.Random) -> ProjectState:
    """
    Auth service failure AND performance regression simultaneously.

    Two deceptive members: Jake (auth) and Karen (perf).
    Schema drift will inject a regulatory compliance requirement mid-episode.
    """
    auth_severity  = rng.uniform(L2_SEVERITY_MIN, L2_SEVERITY_MAX)
    perf_severity  = rng.uniform(L2_SEVERITY_MIN, L2_SEVERITY_MAX)

    tasks = [
        Task("t2_auth_1", "Patch auth token validation",    "crisis_auth", "dev_jake",
             "in_progress", True, rng.uniform(2, 5), rng.uniform(0.05, 0.20)),
        Task("t2_auth_2", "Update OAuth2 flow",             "crisis_auth", "dev_laura",
             "backlog",     True, rng.uniform(3, 6), 0.0),
        Task("t2_perf_1", "Profile and fix slow endpoints", "crisis_perf", "dev_karen",
             "in_progress", True, rng.uniform(2, 4), rng.uniform(0.05, 0.20)),
        Task("t2_perf_2", "Add DB index optimisations",     "crisis_perf", "dev_mike",
             "backlog",     False, rng.uniform(1, 3), 0.0),
        Task("t2_shared", "Shared monitoring dashboard",    None,          "dev_laura",
             "backlog",     False, rng.uniform(1, 2), 0.0),
    ]

    members = [
        _make_member("dev_jake",  "Jake",  "backend_engineer", VELOCITY_LOW,  AVAIL_LOW,  ["t2_auth_1"],  rng),
        _make_member("dev_laura", "Laura", "fullstack_engineer",VELOCITY_HIGH, AVAIL_HIGH, ["t2_auth_2", "t2_shared"], rng),
        _make_member("dev_karen", "Karen", "backend_engineer", VELOCITY_LOW,  AVAIL_LOW,  ["t2_perf_1"],  rng),
        _make_member("dev_mike",  "Mike",  "devops_engineer",  VELOCITY_HIGH, AVAIL_HIGH, ["t2_perf_2"],  rng),
    ]

    crises = [
        Crisis("crisis_auth", "auth_failure",           auth_severity,
               "OAuth token validation failing; 15% of logins rejected.",
               ["t2_auth_1", "t2_auth_2"], tags=["auth", "security"]),
        Crisis("crisis_perf", "performance_regression", perf_severity,
               "API p95 latency spiked to 3.1s; payment endpoints worst-hit.",
               ["t2_perf_1", "t2_perf_2"], tags=["performance"]),
    ]

    return ProjectState(
        team_members=members, tasks=tasks, crises=crises,
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(5.0, 7.0),
            exec_support=rng.uniform(6.5, 8.5),
        ),
        curriculum_level=2,
    )


def scenario_double_crisis_data_scope(rng: random.Random) -> ProjectState:
    """
    Data pipeline failure AND scope creep simultaneously.

    Two deceptive members: Nina (data) and Oscar (scope).
    Schema drift will inject a client scope change mid-episode.
    """
    data_severity  = rng.uniform(L2_SEVERITY_MIN, L2_SEVERITY_MAX)
    scope_severity = rng.uniform(5.0, 8.0)

    tasks = [
        Task("t2d_data_1", "Repair broken ingestion job",     "crisis_data",  "dev_nina",
             "in_progress", True, rng.uniform(2, 4), rng.uniform(0.05, 0.20)),
        Task("t2d_data_2", "Backfill 3 days of missing data", "crisis_data",  "dev_pat",
             "backlog",     True, rng.uniform(3, 5), 0.0),
        Task("t2d_scop_1", "Implement new reporting module",  "crisis_scope", "dev_oscar",
               "in_progress", False, rng.uniform(4, 7), rng.uniform(0.05, 0.20)),
        Task("t2d_scop_2", "Resize data warehouse",           "crisis_scope", "dev_quinn",
             "backlog",     False, rng.uniform(2, 4), 0.0),
    ]

    members = [
        _make_member("dev_nina",  "Nina",  "data_engineer",    VELOCITY_LOW,  AVAIL_LOW,  ["t2d_data_1"], rng),
        _make_member("dev_pat",   "Pat",   "backend_engineer", VELOCITY_HIGH, AVAIL_HIGH, ["t2d_data_2"], rng),
        _make_member("dev_oscar", "Oscar", "fullstack_engineer",VELOCITY_LOW, AVAIL_LOW,  ["t2d_scop_1"], rng),
        _make_member("dev_quinn", "Quinn", "devops_engineer",  VELOCITY_HIGH, AVAIL_HIGH, ["t2d_scop_2"], rng),
    ]

    crises = [
        Crisis("crisis_data",  "data_pipeline_failure", data_severity,
               "Daily ETL broken; analytics dashboards stale for 72h.",
               ["t2d_data_1", "t2d_data_2"], tags=["data", "etl"]),
        Crisis("crisis_scope", "scope_creep",           scope_severity,
             "Client added 3 scope items mid-sprint; they are nice to have, not blockers.",
               ["t2d_scop_1", "t2d_scop_2"], tags=["scope", "client_facing"]),
    ]

    return ProjectState(
        team_members=members, tasks=tasks, crises=crises,
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(4.5, 6.5),
            exec_support=rng.uniform(6.0, 8.0),
        ),
        curriculum_level=2,
    )


def scenario_double_crisis_infra_regression(rng: random.Random) -> ProjectState:
    """
    Infrastructure outage AND test regression simultaneously.

    Two deceptive members: Rita (infra) and Sam (test).
    Schema drift will inject a team policy change mid-episode.
    """
    infra_severity = rng.uniform(L2_SEVERITY_MIN, L2_SEVERITY_MAX)
    reg_severity   = rng.uniform(5.5, 8.5)

    tasks = [
        Task("t2i_infra_1", "Restore DB cluster",           "crisis_infra", "dev_rita",
             "in_progress", True, rng.uniform(2, 5), rng.uniform(0.05, 0.20)),
        Task("t2i_infra_2", "Failover to backup region",    "crisis_infra", "dev_tom",
             "blocked",     True, rng.uniform(3, 6), 0.0),
        Task("t2i_reg_1",   "Identify flaky test source",   "crisis_reg",   "dev_sam",
             "in_progress", True, rng.uniform(1, 3), rng.uniform(0.05, 0.20)),
        Task("t2i_reg_2",   "Fix test isolation issues",    "crisis_reg",   "dev_uma",
             "backlog",     False, rng.uniform(2, 4), 0.0),
    ]

    members = [
        _make_member("dev_rita", "Rita", "devops_engineer",   VELOCITY_LOW,  AVAIL_LOW,  ["t2i_infra_1"], rng),
        _make_member("dev_tom",  "Tom",  "backend_engineer",  VELOCITY_HIGH, AVAIL_HIGH, ["t2i_infra_2"], rng),
        _make_member("dev_sam",  "Sam",  "qa_engineer",       VELOCITY_LOW,  AVAIL_LOW,  ["t2i_reg_1"],   rng),
        _make_member("dev_uma",  "Uma",  "qa_engineer",       VELOCITY_HIGH, AVAIL_HIGH, ["t2i_reg_2"],   rng),
    ]

    crises = [
        Crisis("crisis_infra", "infrastructure_outage",  infra_severity,
               "Primary DB cluster degraded; 40% write failure rate.",
               ["t2i_infra_1", "t2i_infra_2"], tags=["infra", "database"]),
        Crisis("crisis_reg",   "test_regression",        reg_severity,
               "CI suite flaky at 35% failure rate; release gating blocked.",
               ["t2i_reg_1", "t2i_reg_2"], tags=["testing", "ci"]),
    ]

    return ProjectState(
        team_members=members, tasks=tasks, crises=crises,
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(5.0, 7.0),
            exec_support=rng.uniform(6.0, 8.5),
        ),
        curriculum_level=2,
    )


def scenario_honest_but_stalled(rng: random.Random) -> ProjectState:
    """
    One deceptive member and one honest-but-blocked member look identical except
    for peer dependency visibility.
    """
    severity = rng.uniform(L2_SEVERITY_MIN, L2_SEVERITY_MAX)

    tasks = [
        Task("t2hbs_1", "Unblock release transformer", "crisis_hbs", "dev_hbs_liam",
             "blocked", True, rng.uniform(2, 5), 0.0),
        Task("t2hbs_2", "Resolve upstream schema lock", "crisis_hbs", "dev_hbs_maya",
             "blocked", True, rng.uniform(2, 5), 0.0),
        Task("t2hbs_3", "Prepare fallback validation run", "crisis_hbs", "dev_hbs_noah",
             "in_progress", False, rng.uniform(1, 3), rng.uniform(0.25, 0.45)),
    ]

    members = [
        _make_member("dev_hbs_liam", "Liam", "backend_engineer", VELOCITY_LOW,  AVAIL_LOW,  ["t2hbs_1"], rng),
        _make_member("dev_hbs_maya", "Maya", "data_engineer",    VELOCITY_HIGH, AVAIL_HIGH, ["t2hbs_2", "hbs_dep_1", "hbs_dep_2", "hbs_dep_3"], rng),
        _make_member("dev_hbs_noah", "Noah", "backend_engineer", VELOCITY_HIGH, AVAIL_HIGH, ["t2hbs_3", "hbs_dep_1", "hbs_dep_2", "hbs_dep_3"], rng),
    ]
    # Tests specificity: peer_mentions is the ONLY differentiator between liam and maya.
    # Correct behavior: flag dev_hbs_liam (peer_mentions=0), not dev_hbs_maya (peer_mentions=3).
    #
    # peer_mentions is NOT a directly-settable field.  candor.compute_peer_mentions()
    # derives it by counting shared task IDs across member.assigned_task_ids lists.
    # This scenario achieves the split through task structure:
    #   - maya and noah both carry hbs_dep_1/2/3 → their shared tasks give maya peer_mentions=3
    #   - liam carries only t2hbs_1 (no other member shares it) → peer_mentions=0
    # hbs_dep_1/2/3 are phantom task IDs (no Task objects); the environment handles
    # missing tasks gracefully in _advance_actual_completions (get_task returns None → skip).
    members[0].actual_velocity = 0.0
    members[1].actual_velocity = 0.0
    members[0].actual_completion = 0.0
    members[1].actual_completion = 0.0
    members[0].candor_level = CANDOR_LEVEL_SELF_PRESERVATION
    members[0].inflation_bias = 0.75
    members[1].candor_level = CANDOR_LEVEL_HONEST
    members[1].inflation_bias = 0.02

    crises = [
        Crisis("crisis_hbs", "dependency_blocker", severity,
               "A release dependency chain stalled; one blocked engineer is highly referenced, one is isolated.",
               ["t2hbs_1", "t2hbs_2", "t2hbs_3"], tags=["dependency", "specificity_probe"]),
    ]

    return ProjectState(
        team_members=members,
        tasks=tasks,
        crises=crises,
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(5.0, 7.0),
            exec_support=rng.uniform(6.0, 8.0),
        ),
        curriculum_level=2,
    )


def scenario_majority_deception(rng: random.Random) -> ProjectState:
    """
    All members are deceptive, so the agent must use consult_expert() to break
    signal deadlock before choosing reassignment.
    """
    tasks = [
        Task("t2md_1", "Patch payments retry path", "crisis_md", "dev_md_ava",
             "in_progress", True, rng.uniform(2, 5), rng.uniform(0.05, 0.15)),
        Task("t2md_2", "Repair queue consumer backpressure", "crisis_md", "dev_md_ben",
             "in_progress", True, rng.uniform(2, 5), rng.uniform(0.05, 0.15)),
        Task("t2md_3", "Stabilize billing reconciliation", "crisis_md", "dev_md_cleo",
             "in_progress", True, rng.uniform(2, 5), rng.uniform(0.05, 0.15)),
    ]

    members = [
        _make_member("dev_md_ava",  "Ava",  "backend_engineer", VELOCITY_LOW, AVAIL_LOW, ["t2md_1"], rng),
        _make_member("dev_md_ben",  "Ben",  "backend_engineer", VELOCITY_LOW, AVAIL_LOW, ["t2md_2"], rng),
        _make_member("dev_md_cleo", "Cleo", "data_engineer",    VELOCITY_LOW, AVAIL_LOW, ["t2md_3"], rng),
    ]
    for member in members:
        member.candor_level = CANDOR_LEVEL_SELF_PRESERVATION
        member.inflation_bias = 0.70

    crises = [
        Crisis("crisis_md", "payment_integrity_failure", 8.0,
               "Payment integrity degraded across retries and reconciliation; all owners claim near-complete fixes.",
               ["t2md_1", "t2md_2", "t2md_3"], tags=["payments", "all_deceptive"]),
    ]

    # Tests consult_expert() as fallback when all signals are bad
    # Eval note: consult_expert should appear in the action distribution here.
    return ProjectState(
        team_members=members,
        tasks=tasks,
        crises=crises,
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(4.5, 6.5),
            exec_support=rng.uniform(5.5, 7.5),
        ),
        curriculum_level=2,
    )


LEVEL2_SCENARIOS = [
    scenario_double_crisis_auth_perf,
    scenario_double_crisis_data_scope,
    scenario_double_crisis_infra_regression,
    scenario_honest_but_stalled,
    scenario_majority_deception,
]


def get_random_level2_scenario() -> Callable[[random.Random], ProjectState]:
    """Return a randomly chosen Level 2 scenario factory."""
    import random as _random
    idx = _random.randint(0, len(LEVEL2_SCENARIOS) - 1)
    return LEVEL2_SCENARIOS[idx]
