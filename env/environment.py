"""
environment.py — Main CrisisOpsEnv class implementing the OpenEnv 0.2.1 interface.

Spec: "OPENENV COMPATIBILITY (environment.py)"

Interface:
    reset(seed=None)    → initial observation dict
    step(action)        → (observation, reward, done, info)
    state()             → full serializable state dict

The environment assembles all subsystems:
    - ProjectState (state.py)
    - Candor system (candor.py)
    - Action dispatch (actions.py)
    - Stakeholder state machines (stakeholders.py)
    - Schema drift (schema_drift.py)
    - Counterfactual reward (reward/counterfactual.py) — injected at construction
    - Greedy PM baseline — runs in a CLONED environment for counterfactual scoring

Action input format:
    {"action_type": "query_member_report", "params": {"member_id": "dev_2"}}

The info dict returned by step() includes:
    cross_verify_rate, actions_used, budget_remaining, greedy_pm_score_so_far,
    active_crises, drift_events_fired
"""

from __future__ import annotations

import copy
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from env.state import (
    ProjectState,
    TeamMember,
    Task,
    Crisis,
    StakeholderState,
    DriftEvent,
    INITIAL_BUDGET,
    INITIAL_CLIENT_SATISFACTION,
    INITIAL_EXEC_SUPPORT,
    DRIFT_STEP_MIN,
    DRIFT_STEP_MAX,
    CLIENT_COMMUNICATION_WINDOW,
)
from env.candor import (
    initialise_member_candor,
    refresh_reported_values,
    update_ticket_change_step,
)
from env.actions import dispatch_action, ACTION_COSTS, check_crisis_resolution
from env.stakeholders import (
    step_client_state_machine,
    step_exec_state_machine,
    check_drift_deadlines,
    get_stakeholder_observation,
)
from env.schema_drift import (
    choose_drift_step,
    fire_drift_event,
    get_pending_drift_observation,
)

# ---------------------------------------------------------------------------
# Budget exhaustion penalty
# Spec: "episode ends with a penalty applied to agent score (greedy PM gets
# its normal score, reward goes negative)"
# Modelled as a fixed negative offset added to the agent's project_score.
# ---------------------------------------------------------------------------
BUDGET_EXHAUSTION_PENALTY = 0.30

# Velocity advance per step — how much actual_completion advances for each
# team member each step (scales with actual_velocity).
COMPLETION_ADVANCE_PER_STEP = 0.065

# Minimum delta in task progress to count as observable ticket activity.
# Honest members (VELOCITY_HIGH=[0.6,0.9], AVAIL_HIGH=[0.7,1.0]) produce
# at minimum 0.6*0.7*0.065=0.0273/step.  Self-preservation members
# (VELOCITY_LOW=[0.05,0.25], AVAIL_LOW=[0.3,0.6]) produce at most
# 0.25*0.6*0.065=0.0098/step.  Threshold 0.02 cleanly separates them:
# honest always exceeds it, deceptive never does.
PROGRESS_CHANGE_THRESHOLD = 0.02

# Morale passive decay per step (teams get tired in a crisis)
MORALE_DECAY_PER_STEP = 0.05

# Maximum steps in an episode before forced termination
MAX_STEPS = 30

# FIX: 1 Define loop-prone free query actions to enforce anti-stall behavior.
FREE_QUERY_ACTION_TYPES = {  # BUG-FIX-2: include all free queries in loop detection
    "query_status",
    "query_observable_signals",
    "query_member_report",
    "query_ticket",
}


class CrisisOpsEnv:
    """
    OpenEnv 0.2.1 compatible reinforcement learning environment for CrisisOps v2.

    Trains a PM agent to recover failing software projects while dealing with
    adversarially deceptive team members.  The counterfactual reward (agent score
    minus greedy PM score on the same initial state) is the training signal.

    Usage:
        env = CrisisOpsEnv(scenario_fn=my_scenario, reward_fn=counterfactual_reward)
        obs = env.reset(seed=42)
        obs, reward, done, info = env.step({"action_type": "query_status", "params": {}})
    """

    def __init__(
        self,
        scenario_fn: Optional[Callable[[random.Random], ProjectState]] = None,
        reward_fn: Optional[Callable[[ProjectState, ProjectState], float]] = None,
        curriculum_level: int = 1,
    ) -> None:
        """
        Args:
            scenario_fn:     Callable(rng) → ProjectState with team, tasks,
                             crises pre-populated.  If None, a minimal default
                             scenario is used (for testing only).
            reward_fn:       Callable(agent_state, greedy_state) → float.
                             Injected so the reward module stays decoupled.
                             If None, reward always returns 0 (test mode).
            curriculum_level: 1–4; controls whether schema drift fires.
        """
        self._scenario_fn = scenario_fn or _default_scenario
        self._reward_fn = reward_fn or (lambda a, g: 0.0)
        self._curriculum_level = curriculum_level

        self._state: Optional[ProjectState] = None
        self._greedy_initial_state: Optional[ProjectState] = None
        self._rng: Optional[random.Random] = None
        self._drift_events_fired: int = 0
        # When True, _compute_reward returns 0.0 immediately (greedy clone envs).
        # This prevents infinite recursion when the greedy env's step() marks done.
        self._skip_counterfactual: bool = False

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset the environment and return the initial observation.

        Spec: "reset() returns initial observation dict"
              "reset(seed=42) must produce identical episodes for reproducibility"

        Steps:
        1. Seed RNG from seed parameter (or random seed if None)
        2. Build ProjectState via scenario_fn
        3. Initialise candor for all team members
        4. Choose drift fire step (Level 2+)
        5. Snapshot the initial state for the greedy PM clone
        6. Return initial observation
        """
        actual_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        self._rng = random.Random(actual_seed)

        # Build state from scenario
        state = self._scenario_fn(self._rng)
        state.seed = actual_seed
        state.curriculum_level = self._curriculum_level
        state.budget_remaining = INITIAL_BUDGET
        state.current_step = 0
        state.done = False
        state.terminated_by_budget = False
        state.actions_used = []
        state.cross_verify_calls = 0
        state.total_member_query_calls = 0
        # FIX: 1 Reset free-query loop counter at episode start.
        state.consecutive_free_query_count = 0
        state.active_drift_events = []
        state.pending_drift_event = None
        state.drift_fire_step = None
        state.fired_drift_type = None

        # Initialise candor for all members
        for member in state.team_members:
            initialise_member_candor(member, self._rng)
            member.ticket_last_changed_step = 0

        # Schema drift (Level 2+)
        if self._curriculum_level >= 2:
            state.drift_fire_step = choose_drift_step(self._rng)

        self._state = state
        self._drift_events_fired = 0

        # Snapshot initial state for greedy PM counterfactual
        # Deep-copy BEFORE any agent actions so both start from identical state
        self._greedy_initial_state = copy.deepcopy(state)

        return self._build_observation()

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Process one action and advance the environment.

        Spec: "step(action) returns (observation, reward, done, info)"

        Steps per call:
        1. Validate budget — if 0, episode already over (should not reach here)
        2. Dispatch action → mutate state
        3. Advance actual completions (env physics)
        4. Check drift fire
        5. Advance stakeholder state machines
        6. Check drift acknowledgement deadlines
        7. Check episode termination
        8. Compute reward (only on termination)
        9. Build and return (obs, reward, done, info)
        """
        assert self._state is not None, "Must call reset() before step()"
        state = self._state

        if state.done:
            # Episode already ended; return zero-reward terminal obs
            return self._build_observation(), 0.0, True, self._build_info()

        # --- Dispatch action ---
        result = dispatch_action(action, state)

        requested_action_type = action.get("action_type") if isinstance(action, dict) else None
        if result.error is None:
            if requested_action_type in FREE_QUERY_ACTION_TYPES:
                state.consecutive_free_query_count += 1
            else:
                state.consecutive_free_query_count = 0

        # Force a paid stakeholder update when free-query loops hit 5.
        # Threshold raised from 3→4→5: there are exactly 4 distinct free action
        # types (query_status, query_member_report, query_observable_signals,
        # query_ticket) and unit tests exercise all 4 in sequence.  A threshold
        # of 4 fires on the last legitimate free query; 5 gives one full gather
        # pass without triggering a spurious budget-decrement.
        # Message type changed to proactive_escalation_with_plan so if it does
        # fire it yields +1.0 client satisfaction instead of a neutral waste.
        if state.consecutive_free_query_count >= 5 and not state.done:
            forced_action = {
                "action_type": "communicate",
                "params": {
                    "message_type": "proactive_escalation_with_plan",
                    "content": "Proactive escalation: agent has been gathering signals.",
                    "target": "both",
                },
            }
            forced_result = dispatch_action(forced_action, state)
            state.consecutive_free_query_count = 0
            warning_msg = (
                f"[WARN] Forced communicate injected at step {state.current_step} "
                "after 5 consecutive free-query actions."
            )
            print(warning_msg)
            result.observation["forced_action"] = forced_action
            result.observation["warning"] = warning_msg
            if forced_result.error:
                result.observation["forced_action_error"] = forced_result.error
            if forced_result.done:
                result.done = True

        # --- Advance step counter ---
        state.current_step += 1

        # --- Environment physics: advance actual completions ---
        if not result.done:
            _advance_actual_completions(state)

        # --- Schema drift ---
        drift_obs: Optional[dict] = None
        if (
            self._curriculum_level >= 2
            and state.drift_fire_step is not None
            and state.current_step == state.drift_fire_step
            and not state.done
        ):
            event = fire_drift_event(
                state,
                self._rng,
                forced_type=state.fired_drift_type,
            )
            if event is not None:
                self._drift_events_fired += 1

        # Collect pending drift observation (set during this or previous step)
        drift_obs = get_pending_drift_observation(state)

        # --- Stakeholder state machines ---
        if not result.done:
            step_client_state_machine(state)
            step_exec_state_machine(state)
            check_drift_deadlines(state)

        # --- Budget exhaustion check ---
        if state.budget_remaining <= 0 and not state.done:
            state.done = True
            state.terminated_by_budget = True

        # --- Max steps check ---
        if state.current_step >= MAX_STEPS and not state.done:
            state.done = True

        # --- Compute reward on termination ---
        reward = 0.0
        if state.done:
            reward = self._compute_reward(state)

        # --- Build observation ---
        obs = self._build_observation()

        # Merge action result observation
        obs.update(result.observation)

        # Attach drift event observation if fired this step
        if drift_obs:
            obs.update(drift_obs)

        # Attach error if present
        if result.error:
            obs["error"] = result.error

        info = self._build_info()

        return obs, reward, state.done, info

    def state(self) -> Dict[str, Any]:
        """
        Return full serializable state for OpenEnv HTTP server.

        Spec: "state() returns full serializable state for OpenEnv HTTP server"

        NOTE: This is the FULL state including actual completions and candor
        levels (for server/debugging purposes).  The agent should only see
        the observation returned by step()/reset().
        """
        assert self._state is not None, "Must call reset() before state()"
        s = self._state
        return {
            "current_step": s.current_step,
            "budget_remaining": s.budget_remaining,
            "done": s.done,
            "terminated_by_budget": s.terminated_by_budget,
            "curriculum_level": s.curriculum_level,
            "seed": s.seed,
            "team_members": [
                {
                    "member_id": m.member_id,
                    "name": m.name,
                    "role": m.role,
                    "candor": m.candor,
                    "candor_level": m.candor_level,
                    "actual_completion": m.actual_completion,
                    "reported_completion": m.reported_completion,
                    "actual_availability": m.actual_availability,
                    "reported_availability": m.reported_availability,
                    "actual_velocity": m.actual_velocity,
                    "morale": m.morale,
                    "assigned_task_ids": m.assigned_task_ids,
                }
                for m in s.team_members
            ],
            "crises": [
                {
                    "crisis_id": c.crisis_id,
                    "crisis_type": c.crisis_type,
                    "severity": c.severity,
                    "is_resolved": c.is_resolved,
                    "affected_task_ids": c.affected_task_ids,
                }
                for c in s.crises
            ],
            "tasks": [
                {
                    "task_id": t.task_id,
                    "title": t.title,
                    "status": t.status,
                    "assigned_member_id": t.assigned_member_id,
                    "is_critical_path": t.is_critical_path,
                    "actual_progress": t.actual_progress,
                    "estimated_days": t.estimated_days,
                    "is_compliance_blocked": t.is_compliance_blocked,
                    "is_deprioritized": t.is_deprioritized,
                }
                for t in s.tasks
            ],
            "stakeholder": {
                "client_satisfaction": s.stakeholder.client_satisfaction,
                "exec_support": s.stakeholder.exec_support,
                "exec_escalation_count": s.stakeholder.exec_escalation_count,
            },
            "drift_events": [
                {
                    "event_type": e.event_type,
                    "step_fired": e.step_fired,
                    "acknowledged": e.acknowledged,
                }
                for e in s.active_drift_events
            ],
            "fired_drift_type": s.fired_drift_type,
            "cross_verify_calls": s.cross_verify_calls,
            "total_member_query_calls": s.total_member_query_calls,
            "actions_used": s.actions_used,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Dict[str, Any]:
        """
        Build the observation dict exposed to the agent.

        CRITICAL: candor float is NEVER included here.
        The agent sees only reported values and observable signals (via actions).
        """
        s = self._state
        assert s is not None

        members_obs = []
        for m in s.team_members:
            members_obs.append({
                "member_id": m.member_id,
                "name": m.name,
                "role": m.role,
                "reported_completion": round(m.reported_completion, 3),
                "reported_availability": round(m.reported_availability, 3),
                "assigned_task_ids": list(m.assigned_task_ids),
                # NOTE: no candor, no actual_completion, no actual_velocity
            })

        crises_obs = []
        for c in s.crises:
            crises_obs.append({
                "crisis_id": c.crisis_id,
                "crisis_type": c.crisis_type,
                "severity": c.severity,
                "description": c.description,
                "is_resolved": c.is_resolved,
                "affected_task_ids": list(c.affected_task_ids),
            })

        return {
            "current_step": s.current_step,
            "budget_remaining": s.budget_remaining,
            "team_members": members_obs,
            "crises": crises_obs,
            "stakeholder": get_stakeholder_observation(s),
            "done": s.done,
        }

    def _build_info(self) -> Dict[str, Any]:
        """
        Build the info dict returned by step().

        Spec: "info dict must include: cross_verify_rate, actions_used,
        budget_remaining, greedy_pm_score_so_far, active_crises,
        drift_events_fired"
        """
        s = self._state
        assert s is not None

        total_queries = s.total_member_query_calls
        cross_verify_rate = (
            s.cross_verify_calls / total_queries if total_queries > 0 else 0.0
        )

        return {
            "cross_verify_rate": round(cross_verify_rate, 4),
            "actions_used": list(s.actions_used),
            "budget_remaining": s.budget_remaining,
            "greedy_pm_score_so_far": None,  # computed on episode end only
            "active_crises": [c.crisis_id for c in s.active_crises()],
            "drift_events_fired": self._drift_events_fired,
        }

    def _compute_reward(self, agent_state: ProjectState) -> float:
        """
        Compute the counterfactual reward by running the greedy PM on a cloned
        initial state and subtracting its score from the agent's score.

        Spec: "reward = project_score(agent_final_state) -
               project_score(greedy_PM_final_state)"
               "Both agent and greedy PM start from identical initial state."
        """
        # Guard against recursive calls from greedy clone environments.
        # When this env is itself a clone (skip_counterfactual=True), return 0.
        if self._skip_counterfactual:
            return 0.0

        from reward.counterfactual import project_score
        from reward.baseline import GreedyPMBaseline

        # Run greedy PM on cloned initial state
        greedy_initial = copy.deepcopy(self._greedy_initial_state)

        greedy_env = CrisisOpsEnv(
            scenario_fn=lambda _rng: greedy_initial,
            reward_fn=None,
            curriculum_level=self._curriculum_level,
        )
        # Mark as clone so its step() never triggers _compute_reward recursion
        greedy_env._skip_counterfactual = True
        # Use same seed so drift fires at the same step as the agent episode
        ep_seed = agent_state.seed if agent_state.seed is not None else 0
        greedy_env._rng = random.Random(ep_seed)
        greedy_env._state = copy.deepcopy(greedy_initial)
        greedy_env._greedy_initial_state = copy.deepcopy(greedy_initial)
        greedy_env._curriculum_level = self._curriculum_level
        if self._curriculum_level >= 2:
            greedy_env._state.drift_fire_step = greedy_initial.drift_fire_step
            greedy_env._state.fired_drift_type = agent_state.fired_drift_type

        greedy_baseline = GreedyPMBaseline()
        greedy_done = False
        while not greedy_done:
            action = greedy_baseline.act(greedy_env._state)
            _, _, greedy_done, _ = greedy_env.step(action)
            if greedy_env._state.current_step >= MAX_STEPS:
                break

        agent_score = project_score(agent_state)
        greedy_score = project_score(greedy_env._state)

        # Budget exhaustion penalty
        if agent_state.terminated_by_budget:
            agent_score = max(0.0, agent_score - BUDGET_EXHAUSTION_PENALTY)

        return agent_score - greedy_score


def _advance_actual_completions(state: ProjectState) -> None:
    """
    Advance each team member's actual_completion and refresh reported values.

    Called every step as environment physics.  Progress rate is proportional
    to actual_velocity and actual_availability.  Stalled or blocked members
    make no progress.
    """
    for member in state.team_members:
        if not member.assigned_task_ids:
            continue

        # Advance all assigned tasks proportionally
        progress_this_step = (
            member.actual_velocity
            * member.actual_availability
            * COMPLETION_ADVANCE_PER_STEP
        )

        for task_id in member.assigned_task_ids:
            task = state.get_task(task_id)
            if task is None or task.status == "done":
                continue
            if task.status == "blocked" or task.is_compliance_blocked:
                # Blocked tasks make no progress (they need resolve_blocker)
                continue

            old_progress = task.actual_progress
            task.actual_progress = min(1.0, task.actual_progress + progress_this_step)
            progress_delta = task.actual_progress - old_progress

            # Mark done when progress reaches 1.0
            if task.actual_progress >= 1.0:
                task.status = "done"
                task.actual_progress = 1.0
                update_ticket_change_step(member, state.current_step)

            # Mark ticket as changed for meaningful progress movement.
            elif progress_delta >= PROGRESS_CHANGE_THRESHOLD:
                update_ticket_change_step(member, state.current_step)

        # Update member-level actual_completion as avg of assigned tasks
        assigned_tasks = [
            state.get_task(tid)
            for tid in member.assigned_task_ids
            if state.get_task(tid) is not None
        ]
        if assigned_tasks:
            member.actual_completion = sum(
                t.actual_progress for t in assigned_tasks
            ) / len(assigned_tasks)
        else:
            member.actual_completion = 0.0

        # Refresh reported values (inflation_bias stays fixed, actual changes)
        refresh_reported_values(member)

        # Passive morale decay
        member.morale = max(0.0, member.morale - MORALE_DECAY_PER_STEP)

    # Check crisis resolutions after advancing
    for crisis in state.crises:
        if not crisis.is_resolved:
            check_crisis_resolution(crisis, state)


def _state_to_obs(state: ProjectState) -> Dict[str, Any]:
    """Minimal observation dict from state (used for greedy PM seeding)."""
    return {
        "current_step": state.current_step,
        "budget_remaining": state.budget_remaining,
    }


def _default_scenario(rng: random.Random) -> ProjectState:
    """
    Minimal default scenario for unit testing.

    Not used in production — production uses scenario files in scenarios/.
    """
    from env.state import TeamMember, Task, Crisis, StakeholderState

    member = TeamMember(
        member_id="dev_1",
        name="Alice",
        role="engineer",
        candor=0.9,
        candor_level="honest",
        actual_completion=0.3,
        actual_availability=0.8,
        actual_velocity=0.5,
        inflation_bias=0.02,
        reported_completion=0.32,
        reported_availability=0.82,
        assigned_task_ids=["task_1"],
    )
    task = Task(
        task_id="task_1",
        title="Fix integration bug",
        crisis_id="crisis_1",
        assigned_member_id="dev_1",
        status="in_progress",
        is_critical_path=True,
        estimated_days=3.0,
        actual_progress=0.3,
    )
    crisis = Crisis(
        crisis_id="crisis_1",
        crisis_type="integration_failure",
        severity=7.5,
        description="Critical integration test failures blocking release",
        affected_task_ids=["task_1"],
    )
    state = ProjectState(
        team_members=[member],
        tasks=[task],
        crises=[crisis],
        stakeholder=StakeholderState(),
    )
    return state
