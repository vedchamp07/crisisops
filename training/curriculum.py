"""
training/curriculum.py — Level 1→4 curriculum manager.

Spec: "Curriculum: unlock Level 2 when mean reward over last 10 episodes > 0.15.
       Unlock Level 3 when > 0.25. Unlock Level 4 when > 0.35."

The CurriculumManager tracks recent rewards and fires level unlocks at the
correct thresholds.  It is separate from CrisisGenerator so the two can be
tested independently.
"""

from __future__ import annotations

from typing import List, Optional

from training.grpo_trainer import (
    LEVEL2_UNLOCK_THRESHOLD,
    LEVEL3_UNLOCK_THRESHOLD,
    LEVEL4_UNLOCK_THRESHOLD,
    CURRICULUM_WINDOW,
)

# Maximum curriculum level (spec defines 4 levels)
MAX_CURRICULUM_LEVEL = 4


class CurriculumManager:
    """
    Tracks episode rewards and fires curriculum level unlocks.

    Level unlocks are monotonic — once a level is unlocked it stays unlocked.
    The manager does NOT revert to a lower level even if performance degrades.

    Usage:
        cm = CurriculumManager(starting_level=1)
        for ep in episodes:
            reward = run_episode(...)
            cm.record_reward(reward)
            current_level = cm.current_level
    """

    def __init__(self, starting_level: int = 1) -> None:
        """
        Args:
            starting_level: Initial curriculum level (1–4).
        """
        if starting_level < 1 or starting_level > MAX_CURRICULUM_LEVEL:
            raise ValueError(
                f"starting_level must be 1–{MAX_CURRICULUM_LEVEL}, "
                f"got {starting_level}"
            )
        self.current_level: int = starting_level
        self._reward_history: List[float] = []
        self._unlock_log: List[dict] = []

    def record_reward(self, reward: float) -> None:
        """
        Record the reward for a completed episode.

        Automatically checks unlock conditions after each recording.
        """
        self._reward_history.append(reward)
        self._check_unlock()

    def check_unlock(self, window_mean: float) -> int:
        """
        Check if the current window mean warrants a level unlock.

        Returns the (possibly updated) current level.
        Called by grpo_trainer after computing the rolling window mean.
        """
        self._maybe_unlock(window_mean)
        return self.current_level

    def window_mean(self) -> float:
        """Compute the mean reward over the last CURRICULUM_WINDOW episodes."""
        if not self._reward_history:
            return 0.0
        window = self._reward_history[-CURRICULUM_WINDOW:]
        return sum(window) / len(window)

    def unlock_log(self) -> List[dict]:
        """Return a log of all level unlock events with episode index and reward."""
        return list(self._unlock_log)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_unlock(self) -> None:
        """Check unlock condition after recording a new reward."""
        if len(self._reward_history) < CURRICULUM_WINDOW:
            return
        mean = self.window_mean()
        self._maybe_unlock(mean)

    def _maybe_unlock(self, mean: float) -> None:
        """Fire level unlock if threshold met and level not already at target."""
        if self.current_level >= MAX_CURRICULUM_LEVEL:
            return

        thresholds = {
            1: (2, LEVEL2_UNLOCK_THRESHOLD),
            2: (3, LEVEL3_UNLOCK_THRESHOLD),
            3: (4, LEVEL4_UNLOCK_THRESHOLD),
        }

        target_level, threshold = thresholds.get(self.current_level, (None, None))
        if target_level is None:
            return

        if mean > threshold:
            self._unlock_log.append({
                "from_level": self.current_level,
                "to_level": target_level,
                "window_mean": mean,
                "threshold": threshold,
                "episode": len(self._reward_history),
            })
            self.current_level = target_level
