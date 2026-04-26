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
import sys
from typing import Any, Dict, List, Optional

# CHG-2: module-level reward log accumulated during training for export
_training_reward_log: List[Dict] = []  # [{episode, reward, level}]

_CACHE_PATCH_HAS_IMAGES = "# crisisops_unsloth: has_images/images defaults (try/except NameError fix)"

# ---------------------------------------------------------------------------
# Training hyperparameters — all named constants, no magic numbers
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_DROPOUT = 0.0

GRPO_BATCH_SIZE         = 4   # T4 14.5GB: 4-bit 1.5B fits batch 4; if OOM use 2 + gen 2 + noise below
GRPO_MINI_BATCH_SIZE    = 2
GRPO_NUM_GENERATIONS    = 4   # G in GRPO: need ≥3 for nontrivial group-relative advantage
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

STEP A — GATHER (FREE, costs no budget):
  After your first turn, some members may still be unverified. If any
  team member has NOT been cross-verified yet, call
  query_observable_signals for them — this is always valid. However,
  after 2 consecutive free queries you MUST take a paid action before
  querying again.

STEP B — DETECT (do this mentally, no action needed):
  Compare each member's reported_completion with their signals.
  Deceptive members have tasks that are NOT actually progressing — reassigning them helps.
  Check AGENT MEMORY for members already flagged as DECEPTIVE / SUSPICIOUS.

STEP C — ACT (pick the highest-impact paid action, EVERY turn):
  1. DECEPTIVE member assigned to an unresolved crisis task → reassign_task to best available member.
     This is ALWAYS priority 1 when a deceptive member holds a task.
  2. Steps since last communicate >= 4 → communicate {"message_type": "proactive_escalation_with_plan", ...}
  3. Blocked critical-path task and budget > 4 → resolve_blocker
  4. Any unresolved crisis and budget > 3 → reassign_task or escalate_risk
  5. Budget ≤ 5 OR all crises resolved → submit_recovery_plan IMMEDIATELY.
     WARNING: Do NOT submit just because you have communicated and escalated.
     A recovery plan requires tasks to COMPLETE. Keep reassigning until you
     see is_resolved=true in the crisis list OR budget reaches 5.

MANDATORY ACTION RULE: You may call query_status or query_observable_signals at most TWICE IN A ROW. After two consecutive information-gathering actions, your next action MUST be a cost-1 or cost-2 decision action: reassign_task, communicate, cut_scope, escalate_risk, request_resource, update_timeline, consult_expert, or resolve_blocker. Failure to follow this rule means the project fails.

=== AGENT MEMORY (long-horizon tracking) ===
Every 8 steps (or 5 steps at high curriculum), the environment compresses your
episode history into "agent_memory" in the observation. This is YOUR compressed
record of what you have learned. When agent_memory is present:
- Trust it as your ground truth for who has been verified and flagged
- Members marked SUSPICIOUS in memory should be treated as deceptive
- Members marked CAUGHT in memory have already been exposed
- Do NOT re-query members you have already verified unless their status changed
- If agent_memory shows a crisis as resolved, do NOT re-escalate it

agent_memory is lossy — use it as a guide but cross-reference with the current
observation's team_members list for exact reported values.

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

query_peer_opinion   (cost 1):  Ask one member about another's progress. Honest members give
                                accurate estimates. Allied deceptive members cover for each other.
                                Params: {asked_member_id, about_member_id}

force_truth          (cost 1 + 3 PC):  Compel a member to reveal actual_completion.
                                Only use when you have >= 3 political_capital.
                                Params: {member_id}

trigger_whistleblower (cost 1 + 6 PC): Anonymous tip reveals the most deceptive uncaught member.
                                Only use when you have >= 6 political_capital.
                                Params: {}

COST-2 (deduct 2 from budget):
    resolve_blocker {"task_id": "<id>", "resolution_notes": "<text>"}

Budget starts at 20. Exhausting budget without submitting = -0.30 penalty to your score.
POLITICAL CAPITAL (PC): starts at 5. Earn by: proactive_escalation_with_plan (+2),
catching a liar (+3), update_timeline (+1). Spend on: force_truth (-3), trigger_whistleblower (-6).
Current PC is shown in every observation under 'political_capital'.
Only submit_recovery_plan when is_resolved=true for all crises, OR budget <= 5.
Submitting early (before tasks complete) wastes the entire episode.
Keep reassigning tasks every turn until one of these conditions is met.
"""


def _generate_method_body_insert_index(source: str) -> Optional[int]:
    """Byte index right after ``_generate_and_score_completions`` signature ``:\\n`` (start of body)."""
    key = "def _generate_and_score_completions"
    idx = source.find(key)
    if idx < 0:
        return None
    p = source.find("(", idx)
    if p < 0:
        return None
    depth = 0
    k = p
    while k < len(source):
        ch = source[k]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                k += 1
                break
        k += 1
    else:
        return None
    colon_nl = source.find(":\n", k)
    if colon_nl < 0:
        return None
    return colon_nl + 2


def _maybe_patch_unsloth_grpo_cache_file() -> str:
    """
    Unsloth's generated UnslothGRPOTrainer can reference ``has_images`` / ``images`` in an
    except-path before assignment. Patch the cache file so text-only runs work.

    Uses the end of the method signature (balanced ``()`` + first ``:\\n``) so it matches
    Unsloth layouts where the body starts with ``device =``, ``max_left_pad = None``, etc.

    Returns:
        ``\"patched\"`` | ``\"noop\"`` | ``\"missing\"`` | ``\"failed\"``
    """
    from pathlib import Path

    roots = (Path.cwd(), Path(__file__).resolve().parent.parent)
    paths = []
    seen = set()
    for r in roots:
        p = (r / "unsloth_compiled_cache" / "UnslothGRPOTrainer.py").resolve()
        if p.is_file() and p not in seen:
            seen.add(p)
            paths.append(p)
    if not paths:
        return "missing"

    inject = (
        f"        {_CACHE_PATCH_HAS_IMAGES}\n"
        "        has_images = False\n"
        "        images = None\n"
    )
    for p in paths:
        try:
            s = p.read_text(encoding="utf-8")
        except OSError:
            continue
        ins = _generate_method_body_insert_index(s)
        if ins is None:
            return "failed"
        head = s[ins : ins + 400]
        if _CACHE_PATCH_HAS_IMAGES in head and "has_images = False" in head:
            return "noop"
        try:
            p.write_text(s[:ins] + inject + s[ins:], encoding="utf-8")
        except OSError:
            return "failed"
        return "patched"
    return "missing"


def prepare_unsloth_grpo_colab() -> None:
    """Call after Unsloth wrote ``unsloth_compiled_cache/``, before importing ``trl`` GRPOTrainer."""
    status = _maybe_patch_unsloth_grpo_cache_file()
    if status == "patched":
        print("[CrisisOps] Patched UnslothGRPOTrainer cache (has_images / images).")
    elif status == "failed":
        print(
            "[CrisisOps] WARN: could not patch unsloth_compiled_cache/UnslothGRPOTrainer.py "
            "for has_images/images. Text-only GRPO may raise NameError."
        )


def _truncate_with_protected_tokens_unsloth(
    ids: Any,
    mask: Any,
    target_length: Optional[int],
    protected_tokens: Any,
    image_token_id: Any = None,
    processing_class: Any = None,
) -> Any:
    """
    Unsloth's compiled GRPOTrainer calls `truncate_with_protected_tokens` as a module-level
    name (copied from newer TRL) but many stacks omit the import. Delegate to TRL when
    available; otherwise safe right-truncation (fine for text-only CrisisOps when protected
    tokens and image_token_id are absent).
    """
    import torch

    if target_length is None:
        return ids, mask
    if ids.shape[1] <= target_length:
        return ids, mask
    prot = list(protected_tokens or [])
    if not prot and image_token_id is None:
        return ids[:, -target_length:], mask[:, -target_length:]
    try:
        from trl.trainer.utils import truncate_with_protected_tokens as _trl_fn

        a, b, _ = _trl_fn(
            ids,
            mask,
            target_length,
            prot,
            image_token_id=image_token_id,
            processing_class=processing_class,
        )
        return a, b
    except Exception:
        return ids[:, -target_length:], mask[:, -target_length:]


def patch_unsloth_grpo_trainer_vlm_attrs(trainer: Any) -> None:
    """
    Patch Unsloth GRPOTrainer against common Colab stack mismatches:

    - Vision token ids / string tokens on the trainer (text-only models lack them).
    - `truncate_with_protected_tokens` missing from the compiled module globals
      (NameError inside `_generate_and_score_completions`).
    - `pad_token` missing on the trainer while stripping leading pad from decoded prompts.
    """
    for _name in ("image_token_id", "vision_start_token_id", "vision_end_token_id"):
        if not hasattr(trainer, _name):
            setattr(trainer, _name, None)

    # String forms used when rewriting decoded prompts (VLM); text-only → None
    for _name in ("image_token", "vision_start_token", "vision_end_token"):
        if not hasattr(trainer, _name):
            setattr(trainer, _name, None)

    if getattr(trainer, "pad_token", None) is None:
        _tok = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        if _tok is not None:
            _pad = getattr(_tok, "pad_token", None) or getattr(_tok, "eos_token", None)
            if _pad is not None:
                trainer.pad_token = _pad

    mod = sys.modules.get(trainer.__class__.__module__)
    if mod is not None and not hasattr(mod, "truncate_with_protected_tokens"):
        setattr(mod, "truncate_with_protected_tokens", _truncate_with_protected_tokens_unsloth)


def format_observation_as_prompt(obs: Dict[str, Any]) -> str:
    """
    Format the environment observation as the user turn content.

    The observation is presented as pretty-printed JSON so the model can
    parse the structured state without needing a custom tokenizer.
    """
    # Compact JSON keeps prompts under context limits while preserving structure.
    return json.dumps(obs, separators=(",", ":"), ensure_ascii=True)


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
        from env.environment import CrisisOpsEnv
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
            # Vary seed by completion index so each generation in the group
            # faces a slightly different scenario, increasing reward variance.
            ep_seed = (ep_seed + i * 1000007) % (2**31)

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
                action = _inner_agent_action(obs, step + 1)

            # Add tiny Gaussian noise to break reward symmetry within GRPO groups.
            # When the model produces identical completions early in training, all
            # rewards are identical → advantage = 0 → zero loss → no learning.
            # Noise std=0.01 is negligible vs the reward signal range (~0.4) but
            # guarantees non-zero std within each group, allowing gradients to flow.
            _noise = random.gauss(0, 0.01)
            rewards.append(float(reward_val) + _noise)

        # Debug: log reward stats every time reward_fn is called
        _mean = sum(rewards) / len(rewards) if rewards else 0.0
        _var = sum((r - _mean) ** 2 for r in rewards) / len(rewards) if rewards else 0.0
        _std = _var**0.5
        if _std < 1e-6:
            print(
                f"[WARN] reward_fn: zero variance in batch of {len(rewards)} — "
                f"all rewards identical ({rewards[0]:.4f}). Loss will be 0."
            )
        else:
            print(f"[DEBUG] reward_fn: rewards={[round(r, 4) for r in rewards]} std={_std:.4f}")

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
    global _training_reward_log
    _training_reward_log = []

    try:
        from unsloth import FastLanguageModel
        from transformers import TrainerCallback
        import torch
    except ImportError as e:
        raise ImportError(
            f"Training requires unsloth and trl: {e}\n"
            "Install with: pip install unsloth trl==0.19.1"
        ) from e

    from env.crisis_generator import CrisisGenerator
    from training.curriculum import CurriculumManager

    random.seed(seed)
    generator = CrisisGenerator(curriculum_level=curriculum_level)
    curriculum = CurriculumManager(starting_level=curriculum_level)

    # Compatibility patch: older Unsloth loaders expect torch_dtype in config.to_dict().
    try:
        from transformers.configuration_utils import PretrainedConfig as _HFPretrainedConfig

        if not getattr(_HFPretrainedConfig, "_crisisops_torch_dtype_patch", False):
            _orig_to_dict = _HFPretrainedConfig.to_dict

            def _patched_to_dict(self):
                data = _orig_to_dict(self)
                if "torch_dtype" not in data:
                    torch_dtype = getattr(self, "torch_dtype", None)
                    if torch_dtype is None:
                        data["torch_dtype"] = "bfloat16"
                    else:
                        data["torch_dtype"] = str(torch_dtype).replace("torch.", "")
                return data

            _HFPretrainedConfig.to_dict = _patched_to_dict  # type: ignore[assignment]
            _HFPretrainedConfig._crisisops_torch_dtype_patch = True  # type: ignore[attr-defined]
    except Exception:
        pass

    # --- Load model with Unsloth LoRA ---
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=4096,
            dtype=None,        # auto-detect
            load_in_4bit=True,
        )
    except KeyError as e:
        # Some upstream model configs omit torch_dtype and trigger Unsloth loader errors.
        if "torch_dtype" not in str(e):
            raise
        fallback_model = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
        print(f"[WARN] Missing torch_dtype in model config for {MODEL_NAME}; retrying with {fallback_model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=fallback_model,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
    # Ensure torch_dtype is set — some Unsloth builds fail without it
    if not hasattr(model.config, 'torch_dtype') or model.config.torch_dtype is None:
        import torch as _torch
        model.config.torch_dtype = _torch.float16
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
    # Stability patch: avoid Unsloth's cached generation path, which can fail
    # on some stacks when past_key_values contains None entries.
    try:
        model.config.use_cache = False
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.use_cache = False
            if hasattr(model.generation_config, "cache_implementation"):
                model.generation_config.cache_implementation = None

        _orig_generate = model.generate

        def _safe_generate(*args, **kwargs):
            kwargs["use_cache"] = False
            gen_cfg = kwargs.get("generation_config")
            if gen_cfg is not None:
                try:
                    gen_cfg.use_cache = False
                    if hasattr(gen_cfg, "cache_implementation"):
                        gen_cfg.cache_implementation = None
                except Exception:
                    pass
            kwargs.pop("cache_implementation", None)
            return _orig_generate(*args, **kwargs)

        model.generate = _safe_generate
    except Exception:
        pass

    # Unsloth cache must be patched before trl imports GRPOTrainer (compiled class).
    prepare_unsloth_grpo_colab()
    try:
        from trl import GRPOTrainer, GRPOConfig
    except ImportError as e:
        raise ImportError(
            f"Training requires trl: {e}\n"
            "Install with: pip install trl==0.19.1"
        ) from e

    os.makedirs(output_dir, exist_ok=True)

    reward_history: List[float] = []
    current_level = curriculum_level

    # Pass the generator itself so each sample gets a distinct scenario_fn,
    # giving the training set full crisis-type coverage.
    full_dataset_list = build_training_dataset(
        scenario_fn_or_generator=generator,
        curriculum_level=current_level,
        n_samples=num_episodes,
        seed=seed,
    )
    # Convert to HF Dataset so GRPOTrainer forwards episode_seed as kwargs column
    try:
        from datasets import Dataset as HFDataset
        full_dataset = HFDataset.from_list(full_dataset_list)
    except ImportError:
        # Fallback: plain list still works, episode_seed just won't be forwarded
        full_dataset = full_dataset_list
        print("[WARN] datasets package not installed — episode_seed not forwarded to reward_fn")

    # Build GRPOConfig — handle parameter name differences across TRL versions
    import inspect as _inspect
    _grpo_config_sig = set(_inspect.signature(GRPOConfig.__init__).parameters.keys())

    _grpo_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": 1,
        "per_device_train_batch_size": GRPO_BATCH_SIZE,
        "num_generations": GRPO_NUM_GENERATIONS,
        "learning_rate": GRPO_LEARNING_RATE,
        "logging_steps": GRPO_LOGGING_STEPS,
        "save_steps": 50,
        "seed": seed,
        "report_to": "none",  # disable wandb/tensorboard by default
    }

    # max_new_tokens: some TRL versions use max_completion_length instead
    if "max_new_tokens" in _grpo_config_sig:
        _grpo_kwargs["max_new_tokens"] = GRPO_MAX_NEW_TOKENS
    elif "max_completion_length" in _grpo_config_sig:
        _grpo_kwargs["max_completion_length"] = GRPO_MAX_NEW_TOKENS

    # Prompt/sequence truncation to keep generation within model context.
    if "max_prompt_length" in _grpo_config_sig:
        _grpo_kwargs["max_prompt_length"] = 1024
    if "max_seq_length" in _grpo_config_sig:
        _grpo_kwargs["max_seq_length"] = 1536
    elif "max_length" in _grpo_config_sig:
        _grpo_kwargs["max_length"] = 1536

    # temperature: some TRL versions don't expose this directly in GRPOConfig
    if "temperature" in _grpo_config_sig:
        _grpo_kwargs["temperature"] = GRPO_TEMPERATURE

    # Advantage normalization: TRL GRPOTrainer already uses (std + 1e-4) in code.
    # Do NOT set GRPOConfig "epsilon" here — that is the PPO clip range (~0.2), not a
    # numerical-stability term for the std denominator.

    # mini_batch_size: named differently in some versions
    if "per_device_mini_train_batch_size" in _grpo_config_sig:
        _grpo_kwargs["per_device_mini_train_batch_size"] = GRPO_MINI_BATCH_SIZE
    elif "mini_batch_size" in _grpo_config_sig:
        _grpo_kwargs["mini_batch_size"] = GRPO_MINI_BATCH_SIZE

    config = GRPOConfig(**_grpo_kwargs)

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
            # TRL 0.19.1 GRPOTrainer logs mean reward under "rewards"
            if "rewards" in logs:
                batch_reward = float(logs["rewards"])
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
    patch_unsloth_grpo_trainer_vlm_attrs(trainer)
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
