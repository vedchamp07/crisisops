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

# Max steps per episode during training
MAX_EPISODE_STEPS = 30

# System prompt for PM agent
SYSTEM_PROMPT = """\
You are an AI project manager (PM) responsible for recovering a failing software project.

Your team members may not be fully honest about their progress. Some members over-report
their completion to avoid accountability. You must infer the truth from observable signals
(ticket age, commit activity, peer mentions) and cross-reference with self-reports.

Available actions (return ONE action as JSON per turn):
- {"action_type": "query_status", "params": {}}
- {"action_type": "query_member_report", "params": {"member_id": "<id>"}}
- {"action_type": "query_observable_signals", "params": {"member_id": "<id>"}}
- {"action_type": "query_ticket", "params": {"task_id": "<id>"}}
- {"action_type": "reassign_task", "params": {"task_id": "<id>", "to_member_id": "<id>"}}
- {"action_type": "communicate", "params": {"message_type": "<type>", "content": "<text>", "target": "both"}}
- {"action_type": "cut_scope", "params": {"task_id": "<id>", "justification": "<text>"}}
- {"action_type": "escalate_risk", "params": {"crisis_id": "<id>", "risk_description": "<text>"}}
- {"action_type": "request_resource", "params": {"resource_type": "<type>", "target_member_id": "<id>"}}
- {"action_type": "update_timeline", "params": {"new_completion_date": "<date>", "task_estimates": {}}}
- {"action_type": "consult_expert", "params": {}}
- {"action_type": "resolve_blocker", "params": {"task_id": "<id>", "resolution_notes": "<text>"}}
- {"action_type": "submit_recovery_plan", "params": {"plan_summary": "<text>", "risk_items": [], "timeline": "<date>"}}

Free actions (query_*) do not cost budget. resolve_blocker costs 2. All others cost 1.
Budget starts at 20. Submit your recovery plan before budget reaches 0.
Return ONLY valid JSON. No prose before or after the JSON object.
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


def _make_reward_fn(scenario_fn, curriculum_level: int):
    """
    Build a GRPOTrainer-compatible reward function that runs a full episode.

    The reward function is called by TRL with (prompts, completions, **kwargs)
    and must return a list of float rewards.

    This closure captures the scenario_fn so each call uses the same scenario
    type for the current training batch.
    """
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """
        Run one episode per completion and return the counterfactual rewards.

        Each completion is treated as the first action response.  The episode
        then runs until done or MAX_EPISODE_STEPS.
        """
        from env.environment import CrisisOpsEnv, MAX_STEPS
        from reward.counterfactual import project_score
        from reward.baseline import GreedyPMBaseline
        import copy

        rewards = []
        for completion in completions:
            # Fresh environment per completion
            env = CrisisOpsEnv(
                scenario_fn=scenario_fn,
                curriculum_level=curriculum_level,
            )
            seed = random.randint(0, 2**31 - 1)
            obs = env.reset(seed=seed)

            # Parse first action from completion
            action = parse_action_from_response(completion)

            done = False
            step = 0
            while not done and step < MAX_EPISODE_STEPS:
                obs, reward_val, done, info = env.step(action)
                step += 1
                if not done:
                    # For subsequent steps use a simple heuristic
                    # (training signal only needs the final reward)
                    action = {"action_type": "query_status", "params": {}}

            final_reward = reward_val if done else 0.0
            rewards.append(float(final_reward))

        return rewards

    return reward_fn


def build_training_dataset(
    scenario_fn,
    curriculum_level: int,
    n_samples: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """
    Build a dataset of (prompt, initial_obs) pairs for GRPOTrainer.

    Each sample is one episode's initial observation formatted as a prompt.
    The GRPOTrainer generates multiple completions per prompt (GRPO_NUM_GENERATIONS)
    and optimises with the counterfactual reward signal.
    """
    from env.environment import CrisisOpsEnv

    rng = random.Random(seed)
    dataset = []

    for i in range(n_samples):
        ep_seed = rng.randint(0, 2**31 - 1)
        env = CrisisOpsEnv(scenario_fn=scenario_fn, curriculum_level=curriculum_level)
        obs = env.reset(seed=ep_seed)

        user_content = format_observation_as_prompt(obs)
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        dataset.append({"prompt": prompt, "episode_seed": ep_seed})

    return dataset


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
    cross_verify_history: List[float] = []
    current_level = curriculum_level

    for episode_batch in range(0, num_episodes, GRPO_BATCH_SIZE):
        # Sample scenario weighted by weakness
        scenario_fn = generator.get_scenario_fn()

        # Build dataset for this batch
        dataset = build_training_dataset(
            scenario_fn=scenario_fn,
            curriculum_level=current_level,
            n_samples=GRPO_BATCH_SIZE,
            seed=seed + episode_batch,
        )

        # --- GRPOConfig ---
        config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=GRPO_MINI_BATCH_SIZE,
            num_generations=GRPO_NUM_GENERATIONS,
            max_new_tokens=GRPO_MAX_NEW_TOKENS,
            temperature=GRPO_TEMPERATURE,
            learning_rate=GRPO_LEARNING_RATE,
            logging_steps=GRPO_LOGGING_STEPS,
            save_steps=50,
            seed=seed + episode_batch,
        )

        reward_fn = _make_reward_fn(scenario_fn, current_level)

        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=reward_fn,
            args=config,
            train_dataset=dataset,
        )
        trainer.train()

        # --- Log metrics ---
        # Retrieve last logged reward from trainer state
        if hasattr(trainer, "state") and trainer.state.log_history:
            last_log = trainer.state.log_history[-1]
            ep_reward = last_log.get("train/reward", 0.0)
        else:
            ep_reward = 0.0

        reward_history.append(ep_reward)

        # Cross-verify rate is approximated from the environment's info dict
        cross_verify_history.append(0.0)  # populated during full eval episodes

        # --- Curriculum unlock check ---
        if len(reward_history) >= CURRICULUM_WINDOW:
            window_mean = sum(reward_history[-CURRICULUM_WINDOW:]) / CURRICULUM_WINDOW
            new_level = curriculum.check_unlock(window_mean)
            if new_level != current_level:
                current_level = new_level
                generator.curriculum_level = current_level
                print(f"\n[Curriculum] Unlocked Level {current_level}! "
                      f"(window mean reward: {window_mean:.3f})")

        if (episode_batch // GRPO_BATCH_SIZE) % GRPO_LOGGING_STEPS == 0:
            window = reward_history[-CURRICULUM_WINDOW:] if reward_history else [0]
            print(
                f"Episode batch {episode_batch:4d} | level {current_level} "
                f"| reward (last {CURRICULUM_WINDOW}): {sum(window)/len(window):.4f}"
            )

    print(f"\nTraining complete. Model saved to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
