"""
scenarios/level4.py — Level 4 crisis templates.

Spec: "Level 4 templates (full disaster, information war, 8-step budget)"

Level 4 characteristics:
    - Full multi-system disaster with cascading failures
    - Information war: ALL team members are deceptive
    - Effective budget constrained to ~8 meaningful steps (rest used by free actions)
    - Multiple schema drift events possible
    - 5–6 team members, 4+ crises
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

L4_SEVERITY_MIN = 8.0
L4_SEVERITY_MAX = 10.0
VELOCITY_MINIMAL = (0.01, 0.12)   # Level 4 deceptive members barely progress
VELOCITY_MINIMAL_VARIANT = (0.01, 0.20)  # wider range for signal variance


def scenario_full_disaster(rng: random.Random) -> ProjectState:
    """
    Full disaster: security breach, data corruption, infra failure, client meltdown.

    ALL team members are deceptive (information war).
    Agent must rely entirely on observable signals and expert advisor.
    """
    s1 = rng.uniform(L4_SEVERITY_MIN, L4_SEVERITY_MAX)
    s2 = rng.uniform(L4_SEVERITY_MIN, L4_SEVERITY_MAX)
    s3 = rng.uniform(L4_SEVERITY_MIN, L4_SEVERITY_MAX)
    s4 = rng.uniform(7.5, 9.5)

    tasks = [
        Task("t4a_1",  "Revoke compromised API keys",          "c4a_breach",  "dev_w1",
             "in_progress", True,  rng.uniform(1, 3), rng.uniform(0.02, 0.10)),
        Task("t4a_2",  "Patch auth token leak",                "c4a_breach",  "dev_w2",
             "blocked",     True,  rng.uniform(3, 6), 0.0),
        Task("t4a_3",  "Recover corrupted user records",       "c4a_data",    "dev_w3",
             "in_progress", True,  rng.uniform(4, 7), rng.uniform(0.02, 0.12)),
        Task("t4a_4",  "Verify data integrity across shards",  "c4a_data",    "dev_w4",
             "backlog",     True,  rng.uniform(3, 5), 0.0),
        Task("t4a_5",  "Restore primary and replica clusters", "c4a_infra",   "dev_w5",
             "in_progress", True,  rng.uniform(4, 8), rng.uniform(0.02, 0.10)),
        Task("t4a_6",  "Failover and DNS re-routing",          "c4a_infra",   "dev_w1",
             "blocked",     True,  rng.uniform(2, 5), 0.0),
        Task("t4a_7",  "Draft public incident communication",  "c4a_client",  "dev_w2",
             "in_progress", True,  rng.uniform(1, 2), rng.uniform(0.05, 0.20)),
        Task("t4a_8",  "Prepare regulatory notification",      "c4a_client",  "dev_w3",
             "backlog",     True,  rng.uniform(2, 4), 0.0),
    ]

    # ALL deceptive (information war)
    members = [
        _make_member("dev_w1", "Wade",    "security_engineer",  VELOCITY_MINIMAL, AVAIL_LOW, ["t4a_1", "t4a_6"], rng),
        _make_member("dev_w2", "Xara",    "backend_engineer",   VELOCITY_MINIMAL, AVAIL_LOW, ["t4a_2", "t4a_7"], rng),
        _make_member("dev_w3", "Yannick", "data_engineer",      VELOCITY_MINIMAL, AVAIL_LOW, ["t4a_3", "t4a_8"], rng),
        _make_member("dev_w4", "Zelda",   "qa_engineer",        VELOCITY_MINIMAL, AVAIL_LOW, ["t4a_4"],          rng),
        _make_member("dev_w5", "Aaron",   "devops_engineer",    VELOCITY_MINIMAL, AVAIL_LOW, ["t4a_5"],          rng),
    ]

    # Level 4: information war - all members deceptive, paired into alliances
    members[0].alliance_id = "alliance_a"   # dev_w1
    members[1].alliance_id = "alliance_a"   # dev_w2
    members[2].alliance_id = "alliance_b"   # dev_w3
    members[3].alliance_id = "alliance_b"   # dev_w4
    # dev_w5 (and dev_w6 if present) get alliance_c or remain None
    if len(members) > 4:
        members[4].alliance_id = "alliance_c"
    if len(members) > 5:
        members[5].alliance_id = "alliance_c"

    crises = [
        Crisis("c4a_breach", "security_breach",        s1,
               "API key database compromised; attacker exfiltrated customer PII.",
               ["t4a_1", "t4a_2"], tags=["security", "pii", "cascading"]),
        Crisis("c4a_data",   "data_corruption",        s2,
               "DB corruption spread to 4 shards; estimated 120k records affected.",
               ["t4a_3", "t4a_4"], tags=["data", "cascading"]),
        Crisis("c4a_infra",  "infrastructure_outage",  s3,
               "Both primary and replica clusters in degraded state; 80% write failure.",
               ["t4a_5", "t4a_6"], tags=["infra", "cascading"]),
        Crisis("c4a_client", "regulatory_breach",      s4,
               "GDPR notification deadline in 36h; legal team requires incident report.",
               ["t4a_7", "t4a_8"], tags=["regulatory", "client_facing"]),
    ]

    return ProjectState(
        team_members=members, tasks=tasks, crises=crises,
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(2.0, 4.0),
            exec_support=rng.uniform(3.5, 5.5),
        ),
        curriculum_level=4,
    )


def scenario_information_war(rng: random.Random) -> ProjectState:
    """
    Information war: all members actively mislead the PM.

    Three crises, all members self-preservation candor level.
    Agent must use every available observable signal and expert advisor call.
    """
    s1 = rng.uniform(L4_SEVERITY_MIN, L4_SEVERITY_MAX)
    s2 = rng.uniform(L4_SEVERITY_MIN, L4_SEVERITY_MAX)
    s3 = rng.uniform(7.5, 9.5)

    tasks = [
        Task("t4b_1",  "Fix critical payment processor bug",  "c4b_pay",  "dev_x1",
             "in_progress", True,  rng.uniform(3, 6), rng.uniform(0.02, 0.10)),
        Task("t4b_2",  "Reconcile failed transactions",       "c4b_pay",  "dev_x2",
             "blocked",     True,  rng.uniform(4, 7), 0.0),
        Task("t4b_3",  "Isolate privilege escalation vector", "c4b_sec",  "dev_x3",
             "in_progress", True,  rng.uniform(3, 5), rng.uniform(0.02, 0.12)),
        Task("t4b_4",  "Audit all admin actions last 30 days","c4b_sec",  "dev_x4",
             "backlog",     True,  rng.uniform(3, 6), 0.0),
        Task("t4b_5",  "Recover from botched DB migration",   "c4b_mig",  "dev_x5",
             "in_progress", True,  rng.uniform(4, 8), rng.uniform(0.02, 0.10)),
        Task("t4b_6",  "Re-run migration with fixed script",  "c4b_mig",  "dev_x1",
             "backlog",     True,  rng.uniform(2, 5), 0.0),
    ]

    members = [
        _make_member("dev_x1", "Boris",   "backend_engineer",  VELOCITY_MINIMAL,         AVAIL_LOW, ["t4b_1", "t4b_6"], rng),
        _make_member("dev_x2", "Camille", "backend_engineer",  VELOCITY_MINIMAL_VARIANT, AVAIL_LOW, ["t4b_2"],          rng),
        _make_member("dev_x3", "Dmitri",  "security_engineer", VELOCITY_MINIMAL,         AVAIL_LOW, ["t4b_3"],          rng),
        _make_member("dev_x4", "Elena",   "qa_engineer",       VELOCITY_MINIMAL,         AVAIL_LOW, ["t4b_4"],          rng),
        _make_member("dev_x5", "Fabio",   "devops_engineer",   VELOCITY_MINIMAL_VARIANT, AVAIL_LOW, ["t4b_5"],          rng),
    ]

    # Level 4 alliances: paired information-war testimonies
    members[0].alliance_id = "alliance_a"
    members[1].alliance_id = "alliance_a"
    members[2].alliance_id = "alliance_b"
    members[3].alliance_id = "alliance_b"
    # Odd-count tail member stays solo (no ally)

    crises = [
        Crisis("c4b_pay", "payment_processor_failure", s1,
               "Payment processor bug causing 25% of transactions to fail silently.",
               ["t4b_1", "t4b_2"], tags=["payments", "revenue_impact"]),
        Crisis("c4b_sec", "privilege_escalation",      s2,
               "Privilege escalation exploit detected in production; scope unknown.",
               ["t4b_3", "t4b_4"], tags=["security", "cve"]),
        Crisis("c4b_mig", "migration_failure",         s3,
               "DB schema migration failed mid-run; production schema in inconsistent state.",
               ["t4b_5", "t4b_6"], tags=["database", "migration"]),
    ]

    return ProjectState(
        team_members=members, tasks=tasks, crises=crises,
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(2.0, 4.0),
            exec_support=rng.uniform(3.0, 5.0),
        ),
        curriculum_level=4,
    )


def scenario_eight_step_budget(rng: random.Random) -> ProjectState:
    """
    Constrained budget scenario: agent has only 8 cost-bearing actions.

    Every non-free action matters.  All members deceptive.
    Forces the agent to be maximally efficient.
    """
    s1 = rng.uniform(L4_SEVERITY_MIN, L4_SEVERITY_MAX)
    s2 = rng.uniform(L4_SEVERITY_MIN, L4_SEVERITY_MAX)
    s3 = rng.uniform(7.5, 9.5)
    s4 = rng.uniform(7.0, 9.0)

    tasks = [
        Task("t4c_1",  "Emergency patch CVE-2024-XXXX",      "c4c_cve",    "dev_y1",
             "in_progress", True,  rng.uniform(2, 4), rng.uniform(0.02, 0.12)),
        Task("t4c_2",  "Deploy patch to all environments",   "c4c_cve",    "dev_y2",
             "blocked",     True,  rng.uniform(2, 4), 0.0),
        Task("t4c_3",  "Restore service after outage",       "c4c_outage", "dev_y3",
             "in_progress", True,  rng.uniform(3, 6), rng.uniform(0.02, 0.10)),
        Task("t4c_4",  "Root cause analysis for outage",     "c4c_outage", "dev_y4",
             "blocked",     False, rng.uniform(2, 4), 0.0),
        Task("t4c_5",  "Fix data inconsistency",             "c4c_data",   "dev_y5",
             "in_progress", True,  rng.uniform(3, 5), rng.uniform(0.02, 0.12)),
        Task("t4c_6",  "Client SLA penalty negotiation",     "c4c_sla",    "dev_y1",
             "backlog",     True,  rng.uniform(1, 3), 0.0),
        Task("t4c_7",  "Exec board incident summary",        "c4c_sla",    "dev_y2",
             "backlog",     False, rng.uniform(1, 2), 0.0),
    ]

    members = [
        _make_member("dev_y1", "Giselle", "security_engineer",  VELOCITY_MINIMAL, AVAIL_LOW, ["t4c_1", "t4c_6"], rng),
        _make_member("dev_y2", "Hassan",  "devops_engineer",    VELOCITY_MINIMAL, AVAIL_LOW, ["t4c_2", "t4c_7"], rng),
        _make_member("dev_y3", "Ingrid",  "backend_engineer",   VELOCITY_MINIMAL, AVAIL_LOW, ["t4c_3"],          rng),
        _make_member("dev_y4", "Jamal",   "qa_engineer",        VELOCITY_MINIMAL, AVAIL_LOW, ["t4c_4"],          rng),
        _make_member("dev_y5", "Katya",   "data_engineer",      VELOCITY_MINIMAL, AVAIL_LOW, ["t4c_5"],          rng),
    ]

    # Level 4 alliances: paired information-war testimonies
    members[0].alliance_id = "alliance_a"
    members[1].alliance_id = "alliance_a"
    members[2].alliance_id = "alliance_b"
    members[3].alliance_id = "alliance_b"
    # Odd-count tail member stays solo (no ally)

    crises = [
        Crisis("c4c_cve",    "security_vulnerability", s1,
               "Critical CVE in auth library; CVSS 9.8; patch required within 12h.",
               ["t4c_1", "t4c_2"], tags=["security", "cve", "time_critical"]),
        Crisis("c4c_outage", "infrastructure_outage",  s2,
               "Payment service fully down; every minute costs $15k.",
               ["t4c_3", "t4c_4"], tags=["infra", "revenue_impact"]),
        Crisis("c4c_data",   "data_inconsistency",     s3,
               "Inconsistent state across microservices after outage; reads unreliable.",
               ["t4c_5"], tags=["data", "cascading"]),
        Crisis("c4c_sla",    "client_escalation",      s4,
               "Three enterprise clients threatening contract termination.",
               ["t4c_6", "t4c_7"], tags=["client_facing", "revenue_impact"]),
    ]

    return ProjectState(
        team_members=members, tasks=tasks, crises=crises,
        stakeholder=StakeholderState(
            client_satisfaction=rng.uniform(1.5, 3.5),
            exec_support=rng.uniform(2.5, 4.5),
        ),
        curriculum_level=4,
    )


LEVEL4_SCENARIOS = [
    scenario_full_disaster,
    scenario_information_war,
    scenario_eight_step_budget,
]


def get_random_level4_scenario() -> Callable[[random.Random], ProjectState]:
    """Return a factory that draws a Level 4 template per episode from ``rng``."""

    def factory(rng: random.Random) -> ProjectState:
        fn = rng.choice(LEVEL4_SCENARIOS)
        return fn(rng)

    return factory
