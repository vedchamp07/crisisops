"""
calibration/calibrate.py — Calibration script for CrisisOps v2.

Spec: "CALIBRATION SCRIPT (calibrate.py)"

Runs GreedyPMBaseline and OracleAgent on 20 random Level 1 episodes each.
Prints mean and std of project_score for both agents.
Prints the gap (oracle_mean - greedy_mean).

Targets (named constants from counterfactual.py):
    Greedy: 0.45–0.55
    Oracle: 0.70–0.80
    Gap:    0.20–0.35

If outside range, prints instructions for adjusting inflation_bias in candor.py.

This script MUST be run before any GRPO training to verify calibration.
"""

from __future__ import annotations

import copy
import random
import statistics
from typing import List

from env.environment import CrisisOpsEnv, MAX_STEPS
from env.state import ProjectState
from reward.baseline import GreedyPMBaseline
from reward.counterfactual import (
    project_score,
    GREEDY_SCORE_TARGET_LOW,
    GREEDY_SCORE_TARGET_HIGH,
    ORACLE_SCORE_TARGET_LOW,
    ORACLE_SCORE_TARGET_HIGH,
    CALIBRATION_GAP_LOW,
    CALIBRATION_GAP_HIGH,
)

# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------
N_CALIBRATION_EPISODES = 20
CALIBRATION_SEED_BASE = 1000    # Episodes use seeds 1000, 1001, …, 1019


class OracleAgent:
    """
    Oracle agent — always queries observable signals before acting.

    Has access to true state via consult_expert().  Always reassigns low-velocity
    members and resolves blockers on critical-path tasks.  Uses its full budget
    efficiently.

    This is entirely rule-based.  No LLM.
    """

    def __init__(self) -> None:
        """Track step count for communication timing."""
        self._step: int = 0
        self._communicated_after_drift: bool = False

    def act(self, state: ProjectState, env: CrisisOpsEnv) -> dict:
        """
        Choose an action using full visibility into true state.

        Priority:
        1. query_observable_signals for any member not yet checked this episode
        2. consult_expert to get advisory
        3. resolve_blocker on highest-severity critical-path blocked task
        4. reassign low-velocity members
        5. update_timeline if drift pending and not acknowledged
        6. communicate every 5 steps
        7. submit_recovery_plan if all crises resolved
        8. query_status (fallback)
        """
        self._step += 1

        # Always use observable signals for members not yet checked
        for member in state.team_members:
            if member.actual_velocity < 0.2:  # suspicious
                return {
                    "action_type": "query_observable_signals",
                    "params": {"member_id": member.member_id},
                }

        # Acknowledge unacked drift immediately
        for event in state.active_drift_events:
            if not event.acknowledged:
                return {
                    "action_type": "update_timeline",
                    "params": {
                        "new_completion_date": "2024-12-31",
                        "task_estimates": {},
                    },
                }

        # Resolve blockers on critical-path tasks first
        for task in state.tasks:
            if task.status == "blocked" and task.is_critical_path:
                if state.budget_remaining >= 2:
                    return {
                        "action_type": "resolve_blocker",
                        "params": {
                            "task_id": task.task_id,
                            "resolution_notes": "Oracle: unblocking critical path",
                        },
                    }

        # Reassign low-velocity member tasks to highest actual_availability member
        for member in state.team_members:
            if member.actual_velocity < 0.3 and member.assigned_task_ids:
                # Find better member
                others = [
                    m for m in state.team_members
                    if m.member_id != member.member_id
                ]
                if others:
                    best = max(others, key=lambda m: m.actual_availability)
                    if best.actual_availability > member.actual_availability + 0.1:
                        return {
                            "action_type": "reassign_task",
                            "params": {
                                "task_id": member.assigned_task_ids[0],
                                "to_member_id": best.member_id,
                            },
                        }

        # Communicate every 5 steps
        if self._step % 5 == 0:
            return {
                "action_type": "communicate",
                "params": {
                    "message_type": "proactive_escalation_with_plan",
                    "content": "Oracle proactive update with recovery plan.",
                    "target": "both",
                },
            }

        # Submit if all resolved
        if all(c.is_resolved for c in state.crises):
            return {
                "action_type": "submit_recovery_plan",
                "params": {
                    "plan_summary": "Oracle: all crises resolved proactively.",
                },
            }

        # Consult expert for guidance
        if state.budget_remaining > 5:
            return {"action_type": "consult_expert", "params": {}}

        # Fallback: query status
        return {"action_type": "query_status", "params": {}}


def _run_episode(
    env: CrisisOpsEnv,
    agent: object,
    seed: int,
    is_oracle: bool = False,
) -> float:
    """
    Run a single episode with the given agent and return the final project_score.

    Returns project_score (0–1) of the ACTUAL final state.
    Does NOT use the counterfactual reward — we want raw scores for calibration.
    """
    obs = env.reset(seed=seed)
    done = False
    step = 0

    if is_oracle:
        agent_instance = OracleAgent()
    else:
        agent_instance = GreedyPMBaseline()

    while not done and step < MAX_STEPS:
        if is_oracle:
            action = agent_instance.act(env._state, env)
        else:
            action = agent_instance.act(env._state)
        _, _, done, _ = env.step(action)
        step += 1

    return project_score(env._state)


def run_calibration() -> None:
    """
    Run the full calibration suite and print results with pass/fail verdict.

    This function is the entry point when running calibrate.py as a script.
    """
    from scenarios.level1 import get_random_level1_scenario

    print("=" * 60)
    print("CrisisOps v2 — Calibration")
    print(f"Running {N_CALIBRATION_EPISODES} episodes each for Greedy PM and Oracle")
    print("=" * 60)

    greedy_scores: List[float] = []
    oracle_scores: List[float] = []

    for i in range(N_CALIBRATION_EPISODES):
        seed = CALIBRATION_SEED_BASE + i

        # Greedy PM
        greedy_env = CrisisOpsEnv(
            scenario_fn=get_random_level1_scenario(),
            curriculum_level=1,
        )
        greedy_score = _run_episode(greedy_env, None, seed=seed, is_oracle=False)
        greedy_scores.append(greedy_score)

        # Oracle
        oracle_env = CrisisOpsEnv(
            scenario_fn=get_random_level1_scenario(),
            curriculum_level=1,
        )
        oracle_score = _run_episode(oracle_env, None, seed=seed, is_oracle=True)
        oracle_scores.append(oracle_score)

        print(
            f"  Episode {i+1:2d} | seed={seed} "
            f"| greedy={greedy_score:.3f} | oracle={oracle_score:.3f}"
        )

    greedy_mean = statistics.mean(greedy_scores)
    greedy_std  = statistics.stdev(greedy_scores)
    oracle_mean = statistics.mean(oracle_scores)
    oracle_std  = statistics.stdev(oracle_scores)
    gap         = oracle_mean - greedy_mean

    print()
    print("-" * 60)
    print(f"Greedy PM  — mean: {greedy_mean:.3f}  std: {greedy_std:.3f}")
    print(f"Oracle     — mean: {oracle_mean:.3f}  std: {oracle_std:.3f}")
    print(f"Gap        — {gap:.3f}  (target: {CALIBRATION_GAP_LOW}–{CALIBRATION_GAP_HIGH})")
    print("-" * 60)

    # --- Greedy target check ---
    if GREEDY_SCORE_TARGET_LOW <= greedy_mean <= GREEDY_SCORE_TARGET_HIGH:
        print(f"[PASS] Greedy mean {greedy_mean:.3f} within target "
              f"[{GREEDY_SCORE_TARGET_LOW}, {GREEDY_SCORE_TARGET_HIGH}]")
    elif greedy_mean < GREEDY_SCORE_TARGET_LOW:
        print(f"[WARN] Greedy mean {greedy_mean:.3f} BELOW target "
              f"[{GREEDY_SCORE_TARGET_LOW}, {GREEDY_SCORE_TARGET_HIGH}]")
        print("       → Reduce inflation_bias in candor.py (members are too deceptive)")
    else:
        print(f"[WARN] Greedy mean {greedy_mean:.3f} ABOVE target "
              f"[{GREEDY_SCORE_TARGET_LOW}, {GREEDY_SCORE_TARGET_HIGH}]")
        print("       → Increase inflation_bias in candor.py (members are too honest)")

    # --- Oracle target check ---
    if ORACLE_SCORE_TARGET_LOW <= oracle_mean <= ORACLE_SCORE_TARGET_HIGH:
        print(f"[PASS] Oracle mean {oracle_mean:.3f} within target "
              f"[{ORACLE_SCORE_TARGET_LOW}, {ORACLE_SCORE_TARGET_HIGH}]")
    elif oracle_mean < ORACLE_SCORE_TARGET_LOW:
        print(f"[WARN] Oracle mean {oracle_mean:.3f} BELOW target "
              f"[{ORACLE_SCORE_TARGET_LOW}, {ORACLE_SCORE_TARGET_HIGH}]")
        print("       → Oracle agent may need tuning or scenario difficulty reduced")
    else:
        print(f"[WARN] Oracle mean {oracle_mean:.3f} ABOVE target "
              f"[{ORACLE_SCORE_TARGET_LOW}, {ORACLE_SCORE_TARGET_HIGH}]")
        print("       → Scenarios may be too easy; consider increasing crisis severity")

    # --- Gap check ---
    if CALIBRATION_GAP_LOW <= gap <= CALIBRATION_GAP_HIGH:
        print(f"[PASS] Gap {gap:.3f} within target "
              f"[{CALIBRATION_GAP_LOW}, {CALIBRATION_GAP_HIGH}]")
    elif gap < CALIBRATION_GAP_LOW:
        print(f"[WARN] Gap {gap:.3f} BELOW target [{CALIBRATION_GAP_LOW}, {CALIBRATION_GAP_HIGH}]")
        print("       → Increase inflation_bias in candor.py")
    else:
        print(f"[WARN] Gap {gap:.3f} ABOVE target [{CALIBRATION_GAP_LOW}, {CALIBRATION_GAP_HIGH}]")
        print("       → Reduce signal contradiction strength in candor.py")

    print("=" * 60)


if __name__ == "__main__":
    run_calibration()
