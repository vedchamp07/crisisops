"""
tests/test_curriculum.py — Curriculum manager tests.

Spec: "verify level unlock triggers on correct mastery threshold"

Thresholds:
    Level 2 unlocks at mean reward > 0.15 over last 10 episodes
    Level 3 unlocks at mean reward > 0.25 over last 10 episodes
    Level 4 unlocks at mean reward > 0.35 over last 10 episodes
"""

from __future__ import annotations

import pytest

from training.curriculum import CurriculumManager, MAX_CURRICULUM_LEVEL
from training.grpo_trainer import (
    LEVEL2_UNLOCK_THRESHOLD,
    LEVEL3_UNLOCK_THRESHOLD,
    LEVEL4_UNLOCK_THRESHOLD,
    CURRICULUM_WINDOW,
)


def feed_rewards(cm: CurriculumManager, reward: float, n: int = CURRICULUM_WINDOW) -> None:
    """Feed n identical rewards to the curriculum manager."""
    for _ in range(n):
        cm.record_reward(reward)


class TestLevelUnlocks:
    """Level unlocks at correct thresholds."""

    def test_no_unlock_below_level2_threshold(self):
        cm = CurriculumManager(starting_level=1)
        feed_rewards(cm, LEVEL2_UNLOCK_THRESHOLD - 0.01)
        assert cm.current_level == 1, "Should not unlock Level 2 below threshold"

    def test_unlock_level2_at_threshold(self):
        cm = CurriculumManager(starting_level=1)
        feed_rewards(cm, LEVEL2_UNLOCK_THRESHOLD + 0.01)
        assert cm.current_level == 2, (
            f"Expected Level 2 unlock, got level {cm.current_level}"
        )

    def test_no_unlock_below_level3_threshold(self):
        cm = CurriculumManager(starting_level=2)
        feed_rewards(cm, LEVEL3_UNLOCK_THRESHOLD - 0.01)
        assert cm.current_level == 2, "Should not unlock Level 3 below threshold"

    def test_unlock_level3_at_threshold(self):
        cm = CurriculumManager(starting_level=2)
        feed_rewards(cm, LEVEL3_UNLOCK_THRESHOLD + 0.01)
        assert cm.current_level == 3

    def test_no_unlock_below_level4_threshold(self):
        cm = CurriculumManager(starting_level=3)
        feed_rewards(cm, LEVEL4_UNLOCK_THRESHOLD - 0.01)
        assert cm.current_level == 3

    def test_unlock_level4_at_threshold(self):
        cm = CurriculumManager(starting_level=3)
        feed_rewards(cm, LEVEL4_UNLOCK_THRESHOLD + 0.01)
        assert cm.current_level == 4

    def test_does_not_unlock_with_fewer_than_window_episodes(self):
        """Must have at least CURRICULUM_WINDOW episodes before unlocking."""
        cm = CurriculumManager(starting_level=1)
        for _ in range(CURRICULUM_WINDOW - 1):
            cm.record_reward(1.0)  # perfect reward, but not enough episodes
        assert cm.current_level == 1

    def test_level_is_monotonically_increasing(self):
        """Level should never decrease even if performance drops."""
        cm = CurriculumManager(starting_level=1)
        feed_rewards(cm, LEVEL2_UNLOCK_THRESHOLD + 0.05)
        assert cm.current_level == 2
        # Feed poor rewards
        feed_rewards(cm, -1.0, n=20)
        assert cm.current_level == 2, "Level should not regress"

    def test_max_level_not_exceeded(self):
        """Cannot unlock beyond Level 4."""
        cm = CurriculumManager(starting_level=4)
        feed_rewards(cm, 1.0, n=50)
        assert cm.current_level == MAX_CURRICULUM_LEVEL


class TestUnlockLog:
    """Unlock log records correct information."""

    def test_unlock_log_records_event(self):
        cm = CurriculumManager(starting_level=1)
        feed_rewards(cm, 0.20)  # above Level 2 threshold of 0.15
        log = cm.unlock_log()
        assert len(log) == 1
        assert log[0]["from_level"] == 1
        assert log[0]["to_level"] == 2

    def test_unlock_log_records_window_mean(self):
        cm = CurriculumManager(starting_level=1)
        feed_rewards(cm, 0.20)
        log = cm.unlock_log()
        assert abs(log[0]["window_mean"] - 0.20) < 0.001

    def test_check_unlock_returns_new_level(self):
        cm = CurriculumManager(starting_level=1)
        level = cm.check_unlock(LEVEL2_UNLOCK_THRESHOLD + 0.05)
        assert level == 2


class TestWindowMean:
    """Window mean computed correctly."""

    def test_window_mean_uses_last_n_episodes(self):
        cm = CurriculumManager(starting_level=1)
        # First 5 episodes: reward 0.0
        for _ in range(5):
            cm.record_reward(0.0)
        # Last CURRICULUM_WINDOW episodes: reward 1.0
        for _ in range(CURRICULUM_WINDOW):
            cm.record_reward(1.0)
        assert cm.window_mean() == 1.0

    def test_window_mean_zero_with_no_history(self):
        cm = CurriculumManager(starting_level=1)
        assert cm.window_mean() == 0.0
