"""
env/crisis_generator.py — CrisisGenerator with weakness tracking and curriculum escalation.

Spec: "CRISIS GENERATOR (crisis_generator.py)"

- Maintains a per-crisis-type recovery rate tracker across episodes
- After each episode, updates recovery_rate[crisis_type] with exponential
  moving average (alpha=0.3)
- For next episode, samples crisis types with probability inversely proportional
  to recovery rate — agent's weakest areas appear most often
- Tracks which candor levels the agent handles worst and increases prevalence
- Exposes get_difficulty_report() showing current recovery rates per crisis type
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

from env.state import (
    ProjectState,
    CANDOR_LEVEL_HONEST,
    CANDOR_LEVEL_OPTIMISM_BIAS,
    CANDOR_LEVEL_SELF_PRESERVATION,
)

# ---------------------------------------------------------------------------
# EMA constant — spec: "alpha=0.3"
# ---------------------------------------------------------------------------
EMA_ALPHA = 0.3

# Initial recovery rate for unseen crisis types (optimistic prior — assume
# 50% base rate so agent gets challenged gradually)
INITIAL_RECOVERY_RATE = 0.50

# Initial candor level difficulty (same prior as crisis types)
INITIAL_CANDOR_DIFFICULTY = 0.50

# Minimum sampling weight to ensure all types remain reachable
MIN_SAMPLING_WEIGHT = 0.05


class CrisisGenerator:
    """
    Adaptive curriculum manager that tracks agent weakness across episodes.

    After each episode, the generator updates per-crisis-type recovery rates
    and per-candor-level difficulty estimates using exponential moving averages.
    It then adjusts scenario sampling probabilities so harder areas appear more
    often, accelerating the agent's learning of its weak spots.
    """

    def __init__(self, curriculum_level: int = 1) -> None:
        """
        Args:
            curriculum_level: Starting curriculum level (1–4).
        """
        self.curriculum_level = curriculum_level

        # Per-crisis-type EMA of recovery rate (0.0 = never recovered, 1.0 = always)
        self._recovery_rates: Dict[str, float] = {}

        # Per-candor-level EMA of recovery rate
        self._candor_recovery_rates: Dict[str, float] = {
            CANDOR_LEVEL_HONEST:            INITIAL_CANDOR_DIFFICULTY,
            CANDOR_LEVEL_OPTIMISM_BIAS:     INITIAL_CANDOR_DIFFICULTY,
            CANDOR_LEVEL_SELF_PRESERVATION: INITIAL_CANDOR_DIFFICULTY,
        }

        # Episode history for difficulty report
        self._episode_history: List[Dict] = []

    def get_scenario_fn(
        self, rng: Optional[random.Random] = None
    ) -> Callable[[random.Random], ProjectState]:
        """
        Sample a scenario factory weighted toward the agent's weak crisis types.

        Returns a callable that takes an rng and returns a ProjectState.
        The scenario is chosen from the pool for the current curriculum level,
        with probability inversely proportional to recovery rate.
        """
        if rng is None:
            rng = random.Random()

        scenario_pool = self._get_scenario_pool()
        weights = self._compute_weights(scenario_pool)

        # Weighted random choice
        total = sum(weights)
        r = rng.uniform(0, total)
        cumulative = 0.0
        chosen_fn = scenario_pool[0][1]
        for (crisis_types, fn), w in zip(scenario_pool, weights):
            cumulative += w
            if r <= cumulative:
                chosen_fn = fn
                break

        return chosen_fn

    def update_after_episode(
        self,
        state: ProjectState,
        episode_recovery_pct: float,
    ) -> None:
        """
        Update EMA recovery rates after an episode completes.

        Args:
            state:                  Final ProjectState (for crisis types and candor levels)
            episode_recovery_pct:   Fraction of crises resolved this episode (0.0–1.0)
        """
        # Update per-crisis-type recovery rates
        crisis_types_seen = set(c.crisis_type for c in state.crises)
        for ct in crisis_types_seen:
            old = self._recovery_rates.get(ct, INITIAL_RECOVERY_RATE)
            self._recovery_rates[ct] = (
                EMA_ALPHA * episode_recovery_pct + (1.0 - EMA_ALPHA) * old
            )

        # Update per-candor-level recovery rates
        candor_levels_seen = set(m.candor_level for m in state.team_members)
        for cl in candor_levels_seen:
            old = self._candor_recovery_rates.get(cl, INITIAL_CANDOR_DIFFICULTY)
            self._candor_recovery_rates[cl] = (
                EMA_ALPHA * episode_recovery_pct + (1.0 - EMA_ALPHA) * old
            )

        # Store history for difficulty report
        self._episode_history.append({
            "crisis_types": list(crisis_types_seen),
            "candor_levels": list(candor_levels_seen),
            "recovery_pct": episode_recovery_pct,
            "curriculum_level": self.curriculum_level,
        })

    def get_difficulty_report(self) -> Dict:
        """
        Return current recovery rates per crisis type and candor level.

        Spec: "Exposes a get_difficulty_report() method showing current recovery
        rates per crisis type — useful for demo"

        Lower recovery rate = harder for the agent = will appear more often.
        """
        return {
            "curriculum_level": self.curriculum_level,
            "recovery_rates_by_crisis_type": dict(sorted(
                self._recovery_rates.items(), key=lambda kv: kv[1]
            )),
            "recovery_rates_by_candor_level": dict(sorted(
                self._candor_recovery_rates.items(), key=lambda kv: kv[1]
            )),
            "total_episodes": len(self._episode_history),
            "recent_recovery_avg": (
                sum(e["recovery_pct"] for e in self._episode_history[-10:])
                / max(1, len(self._episode_history[-10:]))
            ),
        }

    def _get_scenario_pool(self) -> List[tuple]:
        """
        Return the pool of (crisis_types_list, scenario_fn) for the current level.

        Each entry includes the crisis types that scenario produces, so we can
        weight it by the worst recovery rate among those types.
        """
        from scenarios.level1 import (
            scenario_integration_failure,
            scenario_performance_regression,
            scenario_data_pipeline_failure,
        )
        from scenarios.level2 import (
            scenario_double_crisis_auth_perf,
            scenario_double_crisis_data_scope,
            scenario_double_crisis_infra_regression,
        )
        from scenarios.level3 import (
            scenario_cascading_infra,
            scenario_adversarial_majority,
            scenario_cascading_release_failure,
        )
        from scenarios.level4 import (
            scenario_full_disaster,
            scenario_information_war,
            scenario_eight_step_budget,
        )

        level_pools = {
            1: [
                (["integration_failure"],        scenario_integration_failure),
                (["performance_regression"],     scenario_performance_regression),
                (["data_pipeline_failure"],      scenario_data_pipeline_failure),
            ],
            2: [
                (["auth_failure", "performance_regression"],        scenario_double_crisis_auth_perf),
                (["data_pipeline_failure", "scope_creep"],          scenario_double_crisis_data_scope),
                (["infrastructure_outage", "test_regression"],      scenario_double_crisis_infra_regression),
            ],
            3: [
                (["infrastructure_outage", "data_corruption", "sla_breach"],             scenario_cascading_infra),
                (["security_vulnerability", "technical_debt", "dependency_conflict"],    scenario_adversarial_majority),
                (["release_failure", "test_regression", "client_escalation"],            scenario_cascading_release_failure),
            ],
            4: [
                (["security_breach", "data_corruption", "infrastructure_outage", "regulatory_breach"], scenario_full_disaster),
                (["payment_processor_failure", "privilege_escalation", "migration_failure"],            scenario_information_war),
                (["security_vulnerability", "infrastructure_outage", "data_inconsistency", "client_escalation"], scenario_eight_step_budget),
            ],
        }

        return level_pools.get(self.curriculum_level, level_pools[1])

    def _compute_weights(self, pool: List[tuple]) -> List[float]:
        """
        Compute sampling weights inversely proportional to recovery rate.

        Scenarios covering crisis types the agent handles poorly get higher weight.
        All weights clamped above MIN_SAMPLING_WEIGHT to ensure diversity.
        """
        weights = []
        for crisis_types, _ in pool:
            # Use the WORST (lowest) recovery rate among this scenario's crisis types
            rates = [
                self._recovery_rates.get(ct, INITIAL_RECOVERY_RATE)
                for ct in crisis_types
            ]
            worst_rate = min(rates)
            # Invert: lower recovery → higher weight
            weight = max(MIN_SAMPLING_WEIGHT, 1.0 - worst_rate)
            weights.append(weight)
        return weights
