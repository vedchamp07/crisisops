# CrisisOps v2 — Training Debug: Exact Fixes Only

**Copilot already spent many tokens finding the working dependency combination. Do not re-investigate dependencies. The working stack is already installed in `.venv`. Your job is to apply 7 precise code fixes to `training/grpo_trainer.py`, then execute the debug run and report results.**

---

## Context: what Copilot found

Working dependency stack (already installed in `.venv`):
- `trl==0.19.1` (has both `GRPOTrainer` and `ConstantLengthDataset`)
- `unsloth==2024.x` (the version that was pre-installed)
- `transformers==4.56.2`
- GPU: RTX 3050 6GB, CUDA available

Four crashes hit in order, all in `training/grpo_trainer.py`:
1. `KeyError: 'torch_dtype'` on model load
2. `past_key_values is None` in Unsloth's fast generation path
3. Prompt too long: system prompt (~1617 tokens) + observation + 256 generation > 2048 limit
4. Fix A and Fix B from the previous prompt were **never actually applied** to the real file (Copilot's patches went to a temp session)

---

## Verify installed stack first (one command, no iteration)

```bash
cd /path/to/crisisops
.venv/bin/python -c "
import unsloth, trl, transformers, torch
from trl import GRPOTrainer, GRPOConfig
print(f'unsloth: OK | trl: {trl.__version__} | transformers: {transformers.__version__}')
print(f'torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')
"
```

If this passes, go straight to the fixes. If unsloth fails to import, run:
```bash
.venv/bin/pip install "trl==0.19.1" "transformers==4.56.2" --quiet
```
Then re-verify. Do not change unsloth version.

---

## THE 7 FIXES — apply all to `training/grpo_trainer.py` in one editing pass

Read the full file first. Then apply every fix below. Do not apply them one at a time.

### Fix 1 — max_seq_length: 2048 → 4096

Find the `FastLanguageModel.from_pretrained(` call. Change `max_seq_length=2048` to `max_seq_length=4096`.

Qwen2.5-1.5B supports 131k context natively. At 4096, the model fits prompt + observation + 256 generation tokens comfortably. This is the primary cause of the generation hang.

### Fix 2 — torch_dtype KeyError on model load

Directly after `FastLanguageModel.from_pretrained(...)` returns `model, tokenizer`, add:

```python
    # Ensure torch_dtype is set — some Unsloth builds fail without it
    if not hasattr(model.config, 'torch_dtype') or model.config.torch_dtype is None:
        import torch as _torch
        model.config.torch_dtype = _torch.float16
```

### Fix 3 — Disable KV cache to fix past_key_values=None crash

Directly after Fix 2, add:

```python
    # Disable KV cache — required for GRPO training mode.
    # Unsloth's fast_forward_inference path crashes when GRPOTrainer
    # calls generate() without a pre-seeded past_key_values.
    model.config.use_cache = False
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.use_cache = False
```

### Fix 4 — Do NOT call FastLanguageModel.for_inference()

Search for any call to `FastLanguageModel.for_inference(model)` in the `train()` function. If it exists, remove it entirely. GRPOTrainer requires the model in training mode. Calling `for_inference()` before training triggers the Unsloth cache crash.

Note: `for_inference()` is correct for the evaluation cell in the notebook — do not touch the notebook.

### Fix 5 — Compact JSON observations (token savings ~170 tokens per call)

Find the function `format_observation_as_prompt`. It currently does `json.dumps(obs, indent=2)`. Change to:

```python
def format_observation_as_prompt(obs: Dict[str, Any]) -> str:
    return json.dumps(obs, separators=(',', ':'))
```

This saves ~170 tokens per forward pass. Combined with Fix 1 (4096 context), every level including level 3/4 scenarios now fits.

### Fix 6 — Strip wasted tokens from SYSTEM_PROMPT

The `SYSTEM_PROMPT` string contains 24 inline developer comments like `# FIX-2: synced from llm_agent.py` and `# FIX-5: synced from llm_agent.py`. The model processes these as tokens on every forward pass — 332 wasted tokens.

Find the `SYSTEM_PROMPT = """..."""` block. Inside the string, remove every occurrence of text matching the pattern `  # FIX-\d+:.*` (two spaces + `# FIX-N: ...` to end of line). Also remove `  # CHG-\d+:.*` if any appear in the string.

Do this with a regex or by manually deleting all such comment suffixes from the lines inside the triple-quoted string. The surrounding Python code is unaffected.

After removal, the system prompt should be ~1285 tokens instead of ~1617.

### Fix 7A — Convert dataset to HF Dataset

Find the block in `train()` that calls `build_training_dataset(...)` and assigns `full_dataset`. Replace it with:

```python
    full_dataset_list = build_training_dataset(
        scenario_fn_or_generator=generator,
        curriculum_level=current_level,
        n_samples=num_episodes,
        seed=seed,
    )
    try:
        from datasets import Dataset as HFDataset
        full_dataset = HFDataset.from_list(full_dataset_list)
    except ImportError:
        full_dataset = full_dataset_list
        print("[WARN] datasets not installed — episode_seed not forwarded to reward_fn")
```

### Fix 7B — GRPOConfig version-safe construction

Find the `config = GRPOConfig(...)` block in `train()`. Replace the entire block with:

```python
    import inspect as _inspect
    _grpo_sig = set(_inspect.signature(GRPOConfig.__init__).parameters.keys())

    _grpo_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": 1,
        "per_device_train_batch_size": GRPO_BATCH_SIZE,
        "num_generations": GRPO_NUM_GENERATIONS,
        "learning_rate": GRPO_LEARNING_RATE,
        "logging_steps": GRPO_LOGGING_STEPS,
        "save_steps": 50,
        "seed": seed,
        "report_to": "none",
    }
    # max_new_tokens vs max_completion_length (renamed in some TRL versions)
    if "max_new_tokens" in _grpo_sig:
        _grpo_kwargs["max_new_tokens"] = GRPO_MAX_NEW_TOKENS
    elif "max_completion_length" in _grpo_sig:
        _grpo_kwargs["max_completion_length"] = GRPO_MAX_NEW_TOKENS
    # temperature (removed in some versions)
    if "temperature" in _grpo_sig:
        _grpo_kwargs["temperature"] = GRPO_TEMPERATURE
    # mini batch size (renamed across versions)
    if "per_device_mini_train_batch_size" in _grpo_sig:
        _grpo_kwargs["per_device_mini_train_batch_size"] = GRPO_MINI_BATCH_SIZE
    elif "mini_batch_size" in _grpo_sig:
        _grpo_kwargs["mini_batch_size"] = GRPO_MINI_BATCH_SIZE

    config = GRPOConfig(**_grpo_kwargs)
```

---

## Compile check (must pass before running)

```bash
.venv/bin/python -m py_compile training/grpo_trainer.py && echo "COMPILE OK"
```

If it fails, fix the syntax error before proceeding.

---

## Sanity check (no GPU needed, ~10 seconds)

```bash
cd /path/to/crisisops
TOKENIZERS_PARALLELISM=false .venv/bin/python - <<'EOF'
from training.grpo_trainer import build_training_dataset, parse_action_from_response, format_observation_as_prompt
from env.crisis_generator import CrisisGenerator
from datasets import Dataset
import json

g = CrisisGenerator(curriculum_level=1)
ds = Dataset.from_list(build_training_dataset(g, 1, n_samples=3, seed=0))
assert 'episode_seed' in ds.column_names, "episode_seed missing from dataset columns"
assert ds[0]['prompt'][0]['role'] == 'system'

obs = json.loads(ds[0]['prompt'][1]['content'])
# Verify compact JSON (no indent spaces)
compact = format_observation_as_prompt(obs)
assert '\n' not in compact, "format_observation_as_prompt must return compact JSON (no newlines)"

from training.grpo_trainer import SYSTEM_PROMPT
assert '# FIX-' not in SYSTEM_PROMPT, "FIX comments still in SYSTEM_PROMPT"
sp_tokens = len(SYSTEM_PROMPT) // 4
print(f"System prompt: ~{sp_tokens} tokens (must be < 1450)")
assert sp_tokens < 1450, f"System prompt still too long: {sp_tokens} tokens"

action = parse_action_from_response('{"action_type": "query_status", "params": {}}')
assert action['action_type'] == 'query_status'

print("ALL SANITY CHECKS PASSED")
EOF
```

Fix any failure. Do not proceed past a failing sanity check.

---

## Debug run (30 episodes, ~15-20 min on RTX 3050)

```bash
cd /path/to/crisisops
TOKENIZERS_PARALLELISM=false .venv/bin/python -c "
from training.grpo_trainer import train
train(curriculum_level=1, num_episodes=30, output_dir='./outputs/debug_run', seed=42)
print('DEBUG COMPLETE')
" 2>&1 | tee debug_run.log
```

**Watch the first 3 minutes of output for:**
- `Unsloth: ...` loading message — good
- `Training log:` lines — good, training is running
- `KeyError: torch_dtype` — Fix 2 didn't apply
- `past_key_values is None` — Fix 3 didn't apply
- `Input length exceeds` or truncation warnings — Fix 1 didn't apply or Fix 6 didn't trim enough

**If OOM (out of memory):**
Set `GRPO_BATCH_SIZE = 2` and `GRPO_NUM_GENERATIONS = 2` at the top of `grpo_trainer.py` and retry.

---

## After debug run — report these exact outputs

```bash
# Run this after the debug run finishes (or fails):
python - <<'EOF'
import json, os, sys

print("=== DEBUG RUN REPORT ===")

log_path = './outputs/debug_run/reward_log.json'
if os.path.exists(log_path):
    with open(log_path) as f:
        data = json.load(f)
    rewards = [r['reward'] for r in data]
    print(f"Episodes logged: {len(data)}")
    if rewards:
        import statistics
        print(f"Mean reward: {statistics.mean(rewards):.4f}")
        print(f"Min/Max: {min(rewards):.4f} / {max(rewards):.4f}")
        print(f"All same value: {len(set(round(r,3) for r in rewards)) == 1}")
else:
    print("reward_log.json NOT FOUND — training did not complete")

print()
if os.path.exists('debug_run.log'):
    with open('debug_run.log') as f:
        lines = f.readlines()
    print(f"Log lines: {len(lines)}")
    print("Last 25 lines:")
    print("".join(lines[-25:]))
EOF
```

Paste the full output of this report back.

---

## If debug run succeeds, immediately start full run
```bash
tmux new-session -d -s crisisops_train \
  "cd /path/to/crisisops && TOKENIZERS_PARALLELISM=false .venv/bin/python -c \
  'from training.grpo_trainer import train; train(curriculum_level=1, num_episodes=300, output_dir=\"./outputs/full_run\", seed=42)' \
  2>&1 | tee full_run.log"
echo "Training running in tmux session 'crisisops_train'"
echo "Monitor: tmux attach -t crisisops_train"
```

---

## Do not investigate beyond these fixes

The dependency stack is solved. The 7 fixes above cover every confirmed crash. If a new error appears that is not covered above, paste it back — do not attempt further dependency changes.