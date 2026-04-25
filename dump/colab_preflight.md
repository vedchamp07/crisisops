# CrisisOps v2 — Colab Pre-flight Checklist

Based on the actual `colab_notebook.ipynb` content. Every item is a concrete code change, not a generic suggestion.

---

## 1. Runtime Check

- [ ] Runtime → Change runtime type → **GPU (T4)**
- [ ] Confirm with:
  ```python
  import torch
  print(torch.cuda.get_device_name(0))  # should say Tesla T4
  print(torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")  # ~15.8 GB
  ```
- [ ] If no GPU: the notebook already warns you ("No GPU — training will be slow") but training is effectively useless without one

---

## 2. Model — Switch to 3B (Cell 5)

Current:
```python
model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
```

Change to:
```python
model_name: str = "Qwen/Qwen2.5-3B-Instruct"
```

- [ ] Changed in `Config` dataclass in Cell 5
- [ ] T4 has ~15.8GB VRAM. 3B at float16 ≈ 6GB, fits fine. With LoRA activations during training, you'll use ~10–12GB — still okay.
- [ ] If you hit OOM: add `load_in_4bit=True` to `AutoModelForCausalLM.from_pretrained()` in Cell 6 (currently not there, Colab uses float16 directly)

---

## 3. Sequence Length Fix (Cell 6 — `_log_prob_for_action` and `sample_action`)

**This is the bug causing the truncation warning you saw on Kaggle.**

The prompt in `format_obs()` lists all 13 actions + state + hint, pushing you over 2048 tokens on some observations.

Fix in `sample_action()` — add explicit truncation:
```python
# In sample_action(), replace:
inputs = self.tokenizer(prompt, return_tensors="pt").to(model_dev)

# With:
inputs = self.tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=1800,   # leaves 248 tokens for max_new_tokens=10 + safety margin
).to(model_dev)
```

Do the same in `_log_prob_for_action()`:
```python
# Replace:
inputs = self.tokenizer(prompt, return_tensors="pt").to(model_dev)

# With:
inputs = self.tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=1800,
).to(model_dev)
```

- [ ] Both `sample_action` and `_log_prob_for_action` updated with `truncation=True, max_length=1800`

---

## 4. Num Episodes (Cell 5 — Config)

Current:
```python
num_episodes: int = 50  # increase to 200+ for real training
```

The comment already tells you. 50 updates × G=4 = 200 total episodes. That's not enough for a signal.

Change to:
```python
num_episodes: int = 300   # 300 × G=4 = 1200 total episodes, ~45 min on T4
```

- [ ] `num_episodes` set to at least 200, ideally 300
- [ ] Colab sessions time out at ~12h (Pro) or ~1-2h idle (free). Match num_episodes to your session budget.

---

## 5. G — Group Size (Cell 5 — Config)

Current:
```python
G: int = 4
```

On a single T4 with float16 (not 4-bit), G=4 is the max before you risk OOM on 3B. Keep at 4 for Colab.

- [ ] Keep `G = 4` on Colab T4 (unlike Kaggle T4×2 where you can use 8)
- [ ] If you switched to 4-bit loading, you could push to `G = 6` — test with a 5-episode run first

---

## 6. Format Reward Addition (Cell 6 — `rollout()`)

Currently missing from the notebook. The `rollout()` method only adds `step_rewards` for verification actions. Add format reward:

```python
VALID_ACTIONS_SET = set(VALID_ACTIONS)

def format_reward(action_str):
    return 0.1 if action_str.strip() in VALID_ACTIONS_SET else -0.1
```

Then in `rollout()`, find the cf_reward computation:
```python
# Current:
cf_reward = (proj_score - g_score) + step_rewards

# Change to:
episode_format_bonus = sum(format_reward(a) for a in actions) / max(1, len(actions))
cf_reward = (proj_score - g_score) + step_rewards + episode_format_bonus
```

- [ ] `format_reward()` function added above the `Trainer` class
- [ ] `cf_reward` computation in `rollout()` updated to include it

---

## 7. Checkpoint Path (Cell 6 — Config and `train()`)

Current:
```python
output_dir: str = "./checkpoints"
save_every: int = 25
```

On Colab free tier, `/content/` is wiped when the session ends. If you want checkpoints to survive, save to Google Drive.

**Option A — Google Drive (persistent):**
```python
# Add at top of Cell 2, after drive.mount (it's not in the current notebook — add it):
from google.colab import drive
drive.mount('/content/drive')

# In Config:
output_dir: str = "/content/drive/MyDrive/crisisops_checkpoints"
```

**Option B — Keep local but download manually before session ends:**
```python
from google.colab import files
import shutil
shutil.make_archive('/content/checkpoints_backup', 'zip', './checkpoints')
files.download('/content/checkpoints_backup.zip')
```

- [ ] Decided on Drive or manual download strategy
- [ ] `save_every` set to **25 or less** — on a 300-episode run you want at least 10 checkpoints
- [ ] If using Drive: `drive.mount()` cell added before Cell 2's directory setup

---

## 8. `attn_implementation="eager"` — Keep It

Cell 6 already has this:
```python
attn_implementation="eager",  # avoids fragile SDPA kernels on Colab T4
```

- [ ] Do NOT remove this. Colab T4 has SDPA issues that cause silent wrong outputs, not just slowness.

---

## 9. Wandb — Disable or Set Up (Cell 1)

Cell 1 installs `wandb`. If you run without logging in, it'll prompt mid-training and block execution.

Either disable:
```python
import os
os.environ["WANDB_DISABLED"] = "true"
```

Or login before training:
```python
import wandb
wandb.login()  # will prompt for API key
wandb.init(project="crisisops", name="grpo-run-1")
```

- [ ] Either `WANDB_DISABLED=true` set, or `wandb.login()` + `wandb.init()` called before Cell 6

---

## 10. EMA Baseline — Tune `min_group_std` (Cell 5 — Config)

Current:
```python
min_group_std: float = 0.02
```

This is the threshold below which GRPO falls back to EMA-baseline centering. With G=4 on Level 1 (low scenario diversity), you'll hit this fallback constantly. Lower it slightly so GRPO stays in control longer:

```python
min_group_std: float = 0.01
```

- [ ] `min_group_std` lowered to `0.01`
- [ ] Watch `adv_mode` in the training logs — if it says `ema_fallback` more than 30% of updates, lower further to `0.005`

---

## 11. Calibration Assert (Cell 4)

The current calibration cell runs and prints results but does **not** assert. Add this at the end of Cell 4:

```python
gap = oracle_scores.mean() - greedy_scores.mean()
assert 0.20 <= gap <= 0.35, (
    f"Calibration gap {gap:.3f} out of target [0.20, 0.35]. "
    f"Fix env/candor.py before training."
)
print("✅ Calibration gap OK — safe to train")
```

- [ ] Assert added to end of Cell 4
- [ ] Do not skip calibration to save time — a broken reward signal wastes the entire session

---

## 12. Observation Sanity Check (New Cell — between Cell 5 and Cell 6)

Add this as a new cell between Config and Training to catch prompt issues before burning GPU time:

```python
# Sanity check: can the model parse the observation format?
from env.environment import CrisisOpsEnv

_check_env = CrisisOpsEnv()
_obs = _check_env.reset(seed=42)

_check_trainer = Trainer.__new__(Trainer)
_check_trainer.config = config
_check_trainer.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
_check_trainer._fallback_counter = 0

_prompt = _check_trainer.format_obs(_obs)
_tokens = _check_trainer.tokenizer(_prompt, return_tensors="pt")
print(f"Prompt token length: {_tokens['input_ids'].shape[1]}")
print(f"First 400 chars:\n{_prompt[:400]}")
assert _tokens['input_ids'].shape[1] < 1800, "Prompt too long — shorten format_obs()"
print("✅ Observation format OK")
```

- [ ] Sanity check cell added
- [ ] Token length is under 1800 (if over, shorten `format_obs()` — remove the full action list and just say "Reply with one valid action name")

---

## 13. `ipywidgets` Warning Fix (Cell 1)

The install cell installs `ipywidgets` but Colab sometimes requires a kernel restart for it to activate. `tqdm` progress bars may silently fail without it.

After Cell 1 completes:
- [ ] **Restart runtime** (Runtime → Restart runtime), then re-run from Cell 1
- [ ] Do NOT skip the restart — otherwise module imports from the git clone in Cell 2 may use stale cached versions

---

## 14. Final Pre-run Check

- [ ] Runtime is T4 GPU (re-verify after any restart)
- [ ] `num_episodes` ≥ 200
- [ ] Model name is `Qwen2.5-3B-Instruct`
- [ ] Truncation fix applied in both `sample_action` and `_log_prob_for_action`
- [ ] Calibration assert will fire before training
- [ ] Checkpoint output path is Drive or you have a download plan
- [ ] `WANDB_DISABLED=true` or wandb logged in
- [ ] Sanity check cell passes
- [ ] Runtime → **Run all** (not cell by cell — Cell 2 clones the repo and sets `sys.path`, later cells depend on it)

---

## Quick Reference — Config Values for Colab T4

| Parameter | Value | Note |
|---|---|---|
| `model_name` | `Qwen2.5-3B-Instruct` | Up from 1.5B |
| `G` | `4` | Max safe for single T4 float16 |
| `num_episodes` | `300` | ~45 min on T4 |
| `save_every` | `25` | 12 checkpoints over full run |
| `min_group_std` | `0.01` | Down from 0.02 |
| `max_length` (tokenizer) | `1800` | Prevents truncation warning |
| `attn_implementation` | `"eager"` | Keep — T4 SDPA is flaky |
| `lora_r` | `16` | No change needed |
| `learning_rate` | `2e-5` | No change needed |
