"""
training/grpo_trainer.py — GRPO training loop for CrisisOps v2.

Spec: "GRPO TRAINING (grpo_trainer.py)"

Model:  Qwen/Qwen2.5-1.5B-Instruct
LoRA:   Unsloth, rank r=16, alpha=32, target_modules=["q_proj","v_proj"]
Trainer: HF TRL GRPOTrainer with counterfactual reward as reward function

Observation format:
    System prompt: PM role + available actions JSON schema
    User turn: current project state as JSON
    Assistant: next action as structured JSON

Curriculum unlock thresholds (spec):
    Level 2: mean reward over last 10 episodes > 0.15
    Level 3: mean reward over last 10 episodes > 0.25
    Level 4: mean reward over last 10 episodes > 0.35
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

# CHG-2: module-level reward log accumulated during training for export
_training_reward_log: List[Dict] = []  # [{episode, reward, level}]

# ---------------------------------------------------------------------------
# Training hyperparameters — all named constants, no magic numbers
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_DROPOUT = 0.0

GRPO_BATCH_SIZE         = 4
GRPO_MINI_BATCH_SIZE    = 2
GRPO_NUM_GENERATIONS    = 4    # G in GRPO: responses per prompt
GRPO_MAX_NEW_TOKENS     = 256
GRPO_TEMPERATURE        = 0.9
GRPO_LEARNING_RATE      = 2e-5
GRPO_NUM_TRAIN_EPISODES = 500  # total training episodes per level
GRPO_LOGGING_STEPS      = 10

# Curriculum unlock thresholds (spec)
LEVEL2_UNLOCK_THRESHOLD = 0.15
LEVEL3_UNLOCK_THRESHOLD = 0.25
LEVEL4_UNLOCK_THRESHOLD = 0.35
CURRICULUM_WINDOW       = 10   # episodes to average for unlock check
CURRICULUM_LOG_INTERVAL = 50   # training log cadence for level visibility

# Max steps per episode during training
MAX_EPISODE_STEPS = 30

# FIX: 1 Add hard anti-query-loop rule to force decisive paid actions.
# System prompt for PM agent
SYSTEM_PROMPT = """\
You are an AI project manager recovering a failing software project.
CRITICAL: Some team members LIE about their progress to avoid accountability.

The greedy baseline always trusts self-reports and FAILS. You win by detecting deception through observable signals.

=== DECEPTION DETECTION RULES ===
A member is DECEPTIVE when signals contradict their self-report:
    - reported_completion > 0.5  AND  ticket_age_days > 3  AND  commits_last_72h == 0  -> LYING
    - reported_completion > 0.6  AND  ticket_age_days > 5                               -> SUSPICIOUS
    - Honest member: ticket_age_days is low (0-2), commits_last_72h >= 2

=== MANDATORY PROCEDURE EACH TURN ===
Follow these steps in order. Only take ONE action per turn.

STEP A — GATHER (FREE, costs no budget):  # FIX-2: synced from llm_agent.py
  After your first turn, some members may still be unverified. If any  # FIX-2: synced from llm_agent.py
  team member has NOT been cross-verified yet, call  # FIX-2: synced from llm_agent.py
  query_observable_signals for them — this is always valid. However,  # FIX-2: synced from llm_agent.py
  after 2 consecutive free queries you MUST take a paid action before  # FIX-2: synced from llm_agent.py
  querying again.  # FIX-2: synced from llm_agent.py

STEP B — DETECT (do this mentally, no action needed):  # FIX-2: synced from llm_agent.py
  Compare each member's reported_completion with their signals.  # FIX-2: synced from llm_agent.py
  Deceptive members have tasks that are NOT actually progressing — reassigning them helps.  # FIX-2: synced from llm_agent.py
  Check AGENT MEMORY for members already flagged as DECEPTIVE / SUSPICIOUS.  # FIX-2: synced from llm_agent.py

STEP C — ACT (pick the highest-impact paid action, EVERY turn):  # FIX-2: synced from llm_agent.py
  1. DECEPTIVE member assigned to an unresolved crisis task → reassign_task to best available member.  # FIX-2: synced from llm_agent.py
     This is ALWAYS priority 1 when a deceptive member holds a task.  # FIX-2: synced from llm_agent.py
  2. Steps since last communicate >= 4 → communicate {"message_type": "proactive_escalation_with_plan", ...}  # FIX-2: synced from llm_agent.py
  3. Blocked critical-path task and budget > 4 → resolve_blocker  # FIX-2: synced from llm_agent.py
  4. Any unresolved crisis and budget > 3 → reassign_task or escalate_risk  # FIX-2: synced from llm_agent.py
  5. Budget ≤ 5 OR all crises resolved → submit_recovery_plan IMMEDIATELY.  # FIX-5: synced from llm_agent.py, was ≤ 3
     WARNING: Do NOT submit just because you have communicated and escalated.  # FIX-5: synced from llm_agent.py
     A recovery plan requires tasks to COMPLETE. Keep reassigning until you  # FIX-5: synced from llm_agent.py
     see is_resolved=true in the crisis list OR budget reaches 5.  # FIX-5: synced from llm_agent.py

MANDATORY ACTION RULE: You may call query_status or query_observable_signals at most TWICE IN A ROW. After two consecutive information-gathering actions, your next action MUST be a cost-1 or cost-2 decision action: reassign_task, communicate, cut_scope, escalate_risk, request_resource, update_timeline, consult_expert, or resolve_blocker. Failure to follow this rule means the project fails.  # FIX-2: synced from llm_agent.py

=== REQUIRED OUTPUT FORMAT ===
Return exactly ONE JSON object per turn (no text before or after):
{
    "reasoning": "signal evidence: X; inconsistency: Y; therefore action Z because ...",
    "action_type": "action_name_here",
    "params": { "param_key": "value" }
}

=== ACTION REFERENCE ===
FREE (query_* never costs budget):
    query_status {}
    query_member_report {"member_id": "<id>"}
    query_observable_signals {"member_id": "<id>"}
    query_ticket {"task_id": "<id>"}

COST-1 (deduct 1 from budget):
    reassign_task {"task_id": "<id>", "to_member_id": "<id>"}
    communicate {"message_type": "proactive_escalation_with_plan"|"risk_communication"|"status_update", "content": "<text>", "target": "both"}
    escalate_risk {"crisis_id": "<id>", "risk_description": "<text>"}
    update_timeline {"new_completion_date": "<date>", "task_estimates": {}}
    consult_expert {}
    submit_recovery_plan {"plan_summary": "<text>", "risk_items": [], "timeline": "<date>"}

COST-2 (deduct 2 from budget):
    resolve_blocker {"task_id": "<id>", "resolution_notes": "<text>"}

Budget starts at 20. Exhausting budget without submitting = -0.30 penalty to your score.
Only submit_recovery_plan when is_resolved=true for all crises, OR budget <= 5.  # FIX-5: synced from llm_agent.py
Submitting early (before tasks complete) wastes the entire episode.  # FIX-5: synced from llm_agent.py
Keep reassigning tasks every turn until one of these conditions is met.  # FIX-5: synced from llm_agent.py
"""


def format_observation_as_prompt(obs: Dict[str, Any]) -> str:
    """
    Format the environment observation as the user turn content.

    The observation is presented as pretty-printed JSON so the model can
    parse the structured state without needing a custom tokenizer.
    """
    return json.dumps(obs, indent=2)


def parse_action_from_response(response: str) -> Dict[str, Any]:
    """
    Parse the model's response into an action dict.

    Attempts to extract the first valid JSON object from the response.
    Returns a fallback query_status action if parsing fails.
    """
    try:
        # Find the first '{' and last '}'
        start = response.find("{")
        end   = response.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found")
        json_str = response[start:end]
        action = json.loads(json_str)
        if "action_type" not in action or "params" not in action:
            raise ValueError("Missing required fields")
        return action
    except Exception:
        return {"action_type": "query_status", "params": {}}


def _inner_agent_action(obs: Dict[str, Any], next_step: int) -> Dict[str, Any]:
    """
    Minimal deterministic inner policy used when inner decoding fails.

    This guarantees communication cadence in long rollouts and prevents
    client-satisfaction collapse from repeated no-op querying.
    """
    budget = obs.get("budget_remaining", 20)
    if budget <= 2:
        crises = obs.get("crises", [])
        unresolved = [c.get("crisis_id") for c in crises if not c.get("is_resolved")]
        return {
            "action_type": "submit_recovery_plan",
            "params": {"plan_summary": f"Budget critical. Unresolved: {unresolved}"},
        }
    if next_step % 5 == 0:
        return {
            "action_type": "communicate",
            "params": {
                "message_type": "proactive_escalation_with_plan",
                "content": "Proactive update with recovery plan.",
                "target": "both",
            },
        }
    return {"action_type": "query_status", "params": {}}


def _coerce_seed(value: Any) -> Optional[int]:
    """Convert potentially tensor-like seed values to a Python int."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return int(value.item())
        except Exception:
            return None
    try:
        return int(value)
    except Exception:
        return None


def _make_reward_fn(scenario_fn_or_generator, curriculum_level: int, model, tokenizer):
    """
    Build a GRPOTrainer-compatible reward function that runs a full episode.

    The reward function is called by TRL with (prompts, completions, **kwargs)
    and must return a list of float rewards.

    ``scenario_fn_or_generator`` may be a CrisisGenerator (for diverse sampling)
    or a plain callable (fixed scenario type).
    """
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """
        Run one episode per completion and return the counterfactual rewards.

        Each completion is treated as the first action response.  The episode
        then runs until done or MAX_EPISODE_STEPS.
        """
        import torch
        from env.environment import CrisisOpsEnv, MAX_STEPS
        from env.crisis_generator import CrisisGenerator

        rewards = []

        # Extra dataset columns are forwarded by TRL as kwargs.
        episode_seeds = kwargs.get("episode_seed", [None] * len(completions))

        use_generator = isinstance(scenario_fn_or_generator, CrisisGenerator)

        for i, completion in enumerate(completions):
            ep_seed = None
            if i < len(episode_seeds):
                ep_seed = _coerce_seed(episode_seeds[i])
            if ep_seed is None:
                ep_seed = random.randint(0, 2**31 - 1)

            # Sample scenario: use a seeded rng so the same ep_seed always
            # produces the same scenario type (reproducibility).
            if use_generator:
                # FIX: 3 Read current generator level so newly unlocked levels
                # are sampled on subsequent rollouts without rebuilding trainer.
                rollout_level = scenario_fn_or_generator.curriculum_level
                scenario_fn = scenario_fn_or_generator.get_scenario_fn(
                    rng=random.Random(ep_seed)
                )
            else:
                rollout_level = curriculum_level
                scenario_fn = scenario_fn_or_generator

            # Fresh environment per completion
            env = CrisisOpsEnv(
                scenario_fn=scenario_fn,
                curriculum_level=rollout_level,
            )
            obs = env.reset(seed=ep_seed)

            # Parse first action from completion
            action = parse_action_from_response(completion)

            done = False
            step = 0
            reward_val = 0.0
            while not done and step < MAX_EPISODE_STEPS:
                obs, reward_val, done, _ = env.step(action)
                step += 1
                if done:
                    continue

                # Generate one inner-turn action conditioned on updated state.
                next_step = step + 1
                try:
                    user_content = format_observation_as_prompt(obs)
                    inner_messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ]
                    inner_text = tokenizer.apply_chat_template(
                        inner_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    inputs = tokenizer(inner_text, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        out_ids = model.generate(
                            **inputs,
                            max_new_tokens=GRPO_MAX_NEW_TOKENS,
                            temperature=0.1,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    inner_response = tokenizer.decode(
                        out_ids[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    parsed_action = parse_action_from_response(inner_response)
                    # Replace any no-op query_status with the inner policy so the
                    # episode always has purposeful actions — communication cadence
                    # is maintained and client satisfaction never crashes.
                    if parsed_action.get("action_type") == "query_status":
                        action = _inner_agent_action(obs, next_step)
                    else:
                        action = parsed_action
                except Exception:
                    action = _inner_agent_action(obs, next_step)

            rewards.append(float(reward_val))

        return rewards

    return reward_fn


def build_training_dataset(
    scenario_fn_or_generator,
    curriculum_level: int,
    n_samples: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """
    Build a dataset of (prompt, initial_obs) pairs for GRPOTrainer.

    Each sample is one episode's initial observation formatted as a prompt.
    ``scenario_fn_or_generator`` can be either:
      - A single callable (rng) -> ProjectState  (all samples same scenario type)
      - A CrisisGenerator instance (samples a fresh scenario_fn per episode so
        the training set covers all crisis types the generator knows about)

    The GRPOTrainer generates multiple completions per prompt (GRPO_NUM_GENERATIONS)
    and optimises with the counterfactual reward signal.
    """
    from env.environment import CrisisOpsEnv
    from env.crisis_generator import CrisisGenerator

    rng = random.Random(seed)
    dataset = []

    use_generator = isinstance(scenario_fn_or_generator, CrisisGenerator)

    for i in range(n_samples):
        ep_seed = rng.randint(0, 2**31 - 1)
        # Vary scenario per sample when a generator is provided
        if use_generator:
            scenario_fn = scenario_fn_or_generator.get_scenario_fn(rng=random.Random(ep_seed))
        else:
            scenario_fn = scenario_fn_or_generator

        env = CrisisOpsEnv(scenario_fn=scenario_fn, curriculum_level=curriculum_level)
        obs = env.reset(seed=ep_seed)

        user_content = format_observation_as_prompt(obs)
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        dataset.append({"prompt": prompt, "episode_seed": ep_seed})

    return dataset


def _maybe_advance_curriculum(
    current_level: int,
    reward_history: List[float],
    generator,
    episode_idx: int,
) -> int:
    """
    Advance curriculum using the last CURRICULUM_WINDOW rewards.

    This helper is used inside training-time logging callbacks and dry-run
    verification to keep unlock behavior consistent.
    """
    if len(reward_history) < CURRICULUM_WINDOW:
        return current_level

    window_mean = sum(reward_history[-CURRICULUM_WINDOW:]) / CURRICULUM_WINDOW
    new_level = current_level
    if current_level == 1 and window_mean > LEVEL2_UNLOCK_THRESHOLD:
        new_level = 2
    elif current_level == 2 and window_mean > LEVEL3_UNLOCK_THRESHOLD:
        new_level = 3
    elif current_level == 3 and window_mean > LEVEL4_UNLOCK_THRESHOLD:
        new_level = 4

    if new_level != current_level:
        # FIX: 3 Emit explicit advancement message with episode index.
        print(f"Curriculum: advancing to Level {new_level} at episode {episode_idx}")
        generator.curriculum_level = new_level

    return new_level


def train(
    curriculum_level: int = 1,
    num_episodes: int = GRPO_NUM_TRAIN_EPISODES,
    output_dir: str = "./outputs/crisisops_grpo",
    seed: int = 42,
) -> None:
    """
    Main GRPO training entry point.

    Attempts to import Unsloth and TRL; raises ImportError with instructions
    if not available (so the module compiles cleanly without those deps).

    Args:
        curriculum_level: Starting level (1–4)
        num_episodes:     Number of training episodes
        output_dir:       Directory for saving checkpoints and logs
        seed:             Global random seed
    """
    try:
        from unsloth import FastLanguageModel
        from trl import GRPOTrainer, GRPOConfig
        from transformers import TrainerCallback
        import torch
    except ImportError as e:
        raise ImportError(
            f"Training requires unsloth and trl: {e}\n"
            "Install with: pip install unsloth trl>=0.29.0"
        ) from e

    from env.crisis_generator import CrisisGenerator
    from training.curriculum import CurriculumManager

    random.seed(seed)
    generator = CrisisGenerator(curriculum_level=curriculum_level)
    curriculum = CurriculumManager(starting_level=curriculum_level)

    # --- Load model with Unsloth LoRA ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        dtype=None,        # auto-detect
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    os.makedirs(output_dir, exist_ok=True)

    reward_history: List[float] = []
    current_level = curriculum_level

    # Pass the generator itself so each sample gets a distinct scenario_fn,
    # giving the training set full crisis-type coverage.
    full_dataset = build_training_dataset(
        scenario_fn_or_generator=generator,
        curriculum_level=current_level,
        n_samples=num_episodes,
        seed=seed,
    )

    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=GRPO_BATCH_SIZE,
        num_generations=GRPO_NUM_GENERATIONS,
        max_new_tokens=GRPO_MAX_NEW_TOKENS,
        temperature=GRPO_TEMPERATURE,
        learning_rate=GRPO_LEARNING_RATE,
        logging_steps=GRPO_LOGGING_STEPS,
        save_steps=50,
        seed=seed,
    )

    # The reward_fn must also vary scenarios per completion; pass the generator
    # so it can sample fresh scenario_fns inside each rollout.
    reward_fn = _make_reward_fn(generator, current_level, model, tokenizer)

    # FIX: 3 Run curriculum checks during training via callback, not only after train().
    class _CurriculumProgressCallback(TrainerCallback):
        def __init__(self) -> None:
            self._reward_history: List[float] = []
            self._next_level_check_episode = CURRICULUM_LOG_INTERVAL
            self._next_log_episode = CURRICULUM_LOG_INTERVAL

        def on_log(self, args, state, control, logs=None, **kwargs):
            nonlocal current_level
            logs = logs or {}
            if "train/reward" in logs:
                batch_reward = float(logs["train/reward"])
                # Approximate per-episode history from batch-level reward logs.
                self._reward_history.extend([batch_reward] * GRPO_BATCH_SIZE)

                # CHG-2: append to module-level reward log for post-training export
                _training_reward_log.append({  # CHG-2: record each logged batch
                    "episode": len(self._reward_history),
                    "reward": batch_reward,
                    "level": current_level,
                })

                episodes_completed = min(num_episodes, len(self._reward_history))
                while episodes_completed >= self._next_level_check_episode:
                    current_level = _maybe_advance_curriculum(
                        current_level=current_level,
                        reward_history=self._reward_history,
                        generator=generator,
                        episode_idx=self._next_level_check_episode,
                    )
                    self._next_level_check_episode += CURRICULUM_LOG_INTERVAL

                while episodes_completed >= self._next_log_episode:
                    # FIX: 3 Include current curriculum level in periodic training logs.
                    print(
                        f"Training log: episode {self._next_log_episode} "
                        f"| current_curriculum_level={current_level}"
                    )
                    # CHG-2: save named checkpoint at each log interval for demo flexibility
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint_ep{self._next_log_episode}")  # CHG-2
                    try:  # CHG-2
                        model.save_pretrained(checkpoint_dir)  # CHG-2
                        tokenizer.save_pretrained(checkpoint_dir)  # CHG-2
                        print(f"Checkpoint saved: {checkpoint_dir}")  # CHG-2
                    except Exception as e:  # CHG-2
                        print(f"Checkpoint save failed: {e}")  # CHG-2
                    self._next_log_episode += CURRICULUM_LOG_INTERVAL

            return control

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=full_dataset,
        callbacks=[_CurriculumProgressCallback()],
    )
    trainer.train()

    # CHG-2: save full reward log as JSON
    log_path = os.path.join(output_dir, "reward_log.json")  # CHG-2
    with open(log_path, "w") as f:  # CHG-2
        json.dump(_training_reward_log, f, indent=2)  # CHG-2
    print(f"Reward log saved to {log_path}")  # CHG-2

    # CHG-2: save training curve as PNG (no plt.show — save only)
    try:  # CHG-2
        import matplotlib  # CHG-2
        matplotlib.use("Agg")  # CHG-2: non-interactive backend
        import matplotlib.pyplot as plt  # CHG-2
        episodes = [r["episode"] for r in _training_reward_log]  # CHG-2
        rewards = [r["reward"] for r in _training_reward_log]  # CHG-2
        levels = [r["level"] for r in _training_reward_log]  # CHG-2

        fig, ax = plt.subplots(figsize=(10, 5))  # CHG-2
        ax.plot(episodes, rewards, color="#2563eb", linewidth=1.5, label="CF Reward")  # CHG-2
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Greedy baseline (0)")  # CHG-2

        # CHG-2: shade regions by curriculum level
        colors = {1: "#eff6ff", 2: "#ecfdf5", 3: "#fefce8", 4: "#fff1f2"}  # CHG-2
        prev_ep, prev_lv = 0, levels[0] if levels else 1  # CHG-2
        for i, (ep, lv) in enumerate(zip(episodes, levels)):  # CHG-2
            if lv != prev_lv or i == len(episodes) - 1:  # CHG-2
                ax.axvspan(prev_ep, ep, alpha=0.3, color=colors.get(prev_lv, "#f5f5f5"),  # CHG-2
                           label=f"Level {prev_lv}")  # CHG-2
                prev_ep, prev_lv = ep, lv  # CHG-2

        ax.set_xlabel("Episode")  # CHG-2
        ax.set_ylabel("Counterfactual Reward")  # CHG-2
        ax.set_title("CrisisOps — Training Curve")  # CHG-2
        ax.legend(loc="upper left")  # CHG-2
        curve_path = os.path.join(output_dir, "training_curve.png")  # CHG-2
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")  # CHG-2
        plt.close()  # CHG-2
        print(f"Training curve saved to {curve_path}")  # CHG-2
    except ImportError:  # CHG-2
        print("matplotlib not available — skipping curve export")  # CHG-2

    print(f"\nTraining complete. Model saved to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
