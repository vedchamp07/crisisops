"""
Loud logging for GRPO / Trainer runs (e.g. Colab) so each update is visible.

Usage (after building trainer, before train):

    from training.grpo_verbose_callback import GRPOVerboseStepsCallback
    trainer.add_callback(GRPOVerboseStepsCallback())
    trainer.train()

Print every `print_every_step` optimizer updates; HF still emits richer `on_log` lines
every `logging_steps` (e.g. loss / rewards when the trainer logs them).
"""

from __future__ import annotations

from typing import Any, Optional

from transformers import TrainerCallback


def _fmt_log_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


class GRPOVerboseStepsCallback(TrainerCallback):
    def __init__(self, print_every_step: int = 1) -> None:
        self.print_every_step = max(1, int(print_every_step))

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.global_step % self.print_every_step != 0:
            return
        ms = getattr(state, "max_steps", None)
        if ms is None:
            ms = getattr(args, "max_steps", "?")
        print(
            f"[GRPO] update {state.global_step}/{ms} | epoch={float(state.epoch):.4f}",
            flush=True,
        )

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        parts = [
            f"{k}={_fmt_log_value(v)}"
            for k, v in sorted(logs.items())
            if not str(k).startswith("_")
        ]
        print(f"[GRPO] metrics @ step {state.global_step}: " + " | ".join(parts), flush=True)
