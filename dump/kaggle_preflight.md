# CrisisOps — Kaggle Notebook Pre-flight Checklist

Before creating `kaggle_notebook.ipynb` from `training/colab_notebook.ipynb`, verify every item below.
Work top-to-bottom. Do not start the notebook until all boxes are checked.

---

## 1. Environment & Hardware

- [ ] Kaggle runtime set to **GPU T4 x2** (not T4 x1, not CPU)
  - Settings → Accelerator → GPU T4 x2
- [ ] Internet access **enabled** in Kaggle notebook settings (required for HuggingFace model download)
- [ ] Session persistence understood: **12-hour hard limit**, no warning before kill
- [ ] Verified that `/kaggle/output/` exists and is writable:
  ```python
  import os
  os.makedirs("/kaggle/output/checkpoints", exist_ok=True)
  print("output dir OK")
  ```

---

## 2. Install Cell

Colab's install cell will not work on Kaggle as-is. Replace with:

```python
# Kaggle install cell
!pip install -q unsloth trl accelerate peft datasets huggingface_hub
!pip install -q openenv  # or your local path if not on PyPI
```

- [ ] Removed any `!pip install google-colab*` or `from google.colab import drive` lines
- [ ] Removed any `drive.mount('/content/drive')` calls
- [ ] Confirmed `unsloth` installs without CUDA errors on T4 (it should; Kaggle ships CUDA 12.x)
- [ ] Added `import os; os.environ["TOKENIZERS_PARALLELISM"] = "false"` to suppress tokenizer warnings

---

## 3. Model Change

- [ ] Changed model name from `Qwen/Qwen2.5-1.5B-Instruct` to `Qwen/Qwen2.5-3B-Instruct`
  ```python
  MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # updated from 1.5B
  ```
- [ ] Confirmed 4-bit quantization is on (`load_in_4bit=True`) — mandatory for 3B on T4×2
- [ ] Verified Unsloth supports 3B: `FastLanguageModel.from_pretrained(MODEL_NAME, load_in_4bit=True)`

---

## 4. Multi-GPU Setup (critical — Colab notebook won't have this)

Kaggle T4×2 requires `accelerate`. Add this cell **before** the trainer cell:

```python
from accelerate import Accelerator
accelerator = Accelerator()
print(f"Using {accelerator.num_processes} GPU(s)")
# should print: Using 2 GPU(s)
```

- [ ] Launch string updated: if running via CLI it's `accelerate launch --num_processes 2`
  - In a notebook, `accelerate` handles this automatically if initialized correctly
- [ ] Confirmed `device_map="auto"` is set in `from_pretrained` so model shards across both GPUs

---

## 5. GRPO Hyperparameters

Update these in the trainer config — the Colab defaults are wrong for T4×2:

| Parameter | Colab value | Kaggle value | Why |
|---|---|---|---|
| `num_generations` | 4 | **8** | Lower variance gradient estimate |
| `per_device_train_batch_size` | 4 | **4** | Per GPU, effective = 8 |
| `gradient_accumulation_steps` | 1 | **2** | Effective batch = 16 |
| `max_steps` | (whatever) | **1000+** | You have 12h, use it |

- [ ] All four parameters updated in notebook

---

## 6. Checkpoint Saving (non-negotiable)

Kaggle kills sessions at 12h. If checkpoints aren't saved mid-training, you lose everything.

- [ ] Checkpoint save every 50 steps wired to `/kaggle/output/`:
  ```python
  # inside training loop or via TRL callback
  if step % 50 == 0:
      model.save_pretrained(f"/kaggle/output/checkpoints/ckpt_step{step}")
      tokenizer.save_pretrained(f"/kaggle/output/checkpoints/ckpt_step{step}")
      print(f"Saved checkpoint at step {step}")
  ```
- [ ] Final model saved at end:
  ```python
  model.save_pretrained("/kaggle/output/crisisops_final")
  tokenizer.save_pretrained("/kaggle/output/crisisops_final")
  ```
- [ ] Verified nothing is saved to `/kaggle/working/` alone (it does NOT persist after session ends)
- [ ] Optionally push to HuggingFace Hub mid-training as a backup:
  ```python
  from huggingface_hub import HfApi
  # add your HF token as a Kaggle secret (see Section 9)
  ```

---

## 7. Format Reward Addition

The Colab notebook likely has only counterfactual reward. Add the format reward shaping:

```python
VALID_ACTIONS = {
    "query_status", "query_member_report",
    "query_observable_signals", "query_ticket",
    "reassign_task", "communicate", "cut_scope",
    "escalate_risk", "request_resource",
    "update_timeline", "consult_expert",
    "resolve_blocker", "submit_recovery_plan"
}

def compute_reward(action_str, counterfactual_reward):
    format_bonus = 0.1 if action_str.strip() in VALID_ACTIONS else -0.1
    return counterfactual_reward + format_bonus
```

- [ ] Format reward function added
- [ ] Reward function wired into GRPO rollout loop (not just logging)
- [ ] Confirmed reward is applied to *all* G=8 rollouts, not just the first

---

## 8. Calibration Cell

Colab notebook may skip calibration. Kaggle notebook must run it first.

- [ ] Calibration cell added before training:
  ```python
  from calibration.calibrate import run_calibration
  greedy_mean, oracle_mean, gap = run_calibration(n_episodes=20, seed=42)
  print(f"Greedy: {greedy_mean:.3f} | Oracle: {oracle_mean:.3f} | Gap: {gap:.3f}")
  assert 0.20 <= gap <= 0.35, f"Reward gap {gap:.3f} out of target range — fix env before training"
  ```
- [ ] If gap < 0.20: increase `inflation_bias` in `env/candor.py` before running notebook
- [ ] If gap > 0.35: reduce signal contradiction strength in `env/candor.py` before running notebook

---

## 9. Secrets & API Keys

Kaggle uses the **Secrets** panel (Add-ons → Secrets), not environment variables or `.env` files.

- [ ] `HF_TOKEN` added to Kaggle secrets if pushing checkpoints to HuggingFace Hub
- [ ] Secret read in notebook:
  ```python
  from kaggle_secrets import UserSecretsClient
  secrets = UserSecretsClient()
  hf_token = secrets.get_secret("HF_TOKEN")
  ```
- [ ] No hardcoded API keys anywhere in the notebook
- [ ] `.gitignore` confirmed to exclude any local secrets files

---

## 10. Observation Sanity Check Cell

Add this cell after env setup, before training — catches prompt format bugs before wasting GPU time:

```python
from env.environment import CrisisOpsEnv
from transformers import AutoTokenizer

env = CrisisOpsEnv()
obs, _ = env.reset(seed=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
prompt = YOUR_PROMPT_TEMPLATE.format(obs=obs)  # use your actual template
inputs = tokenizer(prompt, return_tensors="pt")

print("Prompt token length:", inputs["input_ids"].shape[1])
print("First 500 chars of prompt:")
print(prompt[:500])
# Manually verify the model can parse this and output a valid action name
```

- [ ] Token length is under 1024 (if over, truncate the observation or shorten prompt)
- [ ] Prompt unambiguously asks for a single action name from the valid set
- [ ] System prompt tells the model to output *only* the action name, nothing else

---

## 11. Plotting & Logging

Colab has nice inline plots. Kaggle does too, but verify:

- [ ] `matplotlib` inline plots work: `%matplotlib inline` at top of notebook
- [ ] Reward curve plotted every 100 steps (not just at end — you want to catch flat curves early)
- [ ] Cross-verification rate (CVR) logged alongside reward — this is your leading indicator
  - If CVR is rising but reward is flat → environment or reward formula issue
  - If both are flat → model not learning at all → check prompt format
- [ ] W&B / TensorBoard optional but recommended:
  ```python
  # lightweight alternative: just print to stdout, Kaggle captures it in logs
  print(f"step={step} reward={reward:.4f} cvr={cvr:.4f}")
  ```

---

## 12. Curriculum Settings

- [ ] Start on **Level 1 only**
- [ ] Unlock threshold kept at 0.20 rolling mean (more conservative than README's 0.15)
- [ ] Rolling window is at least 50 episodes before unlocking (not 10-20)
- [ ] Level printed at each checkpoint so you know where training is when you check back

---

## 13. Path Differences from Colab

All Colab-specific paths must be replaced:

| Colab path | Kaggle equivalent |
|---|---|
| `/content/` | `/kaggle/working/` (temp) or `/kaggle/output/` (persistent) |
| `/content/drive/MyDrive/` | `/kaggle/output/` |
| `from google.colab import drive` | Remove entirely |
| `drive.mount(...)` | Remove entirely |

- [ ] Grep the Colab notebook for `/content` and replace every instance
- [ ] Grep for `google.colab` and remove every import/call

---

## 14. Final Pre-run Checklist

Run through this immediately before hitting "Save and Run All":

- [ ] Runtime is T4 x2 (re-check, Kaggle resets this sometimes)
- [ ] Internet is enabled
- [ ] All secrets are saved in Kaggle Secrets panel
- [ ] Calibration assert will run before training starts
- [ ] Checkpoint save is wired to `/kaggle/output/`
- [ ] "Save and Run All" selected (not interactive run)
- [ ] You have enough Kaggle GPU quota for the session (check at kaggle.com/settings)

---

## Quick Reference: Key Numbers

| Thing | Value |
|---|---|
| Model | Qwen2.5-3B-Instruct |
| Quantization | 4-bit |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| G (generations) | 8 |
| Batch per device | 4 |
| Grad accumulation | 2 |
| Effective batch | 16 |
| Checkpoint every | 50 steps |
| Calibration gap target | 0.20 – 0.35 |
| Level 1 unlock threshold | reward mean > 0.20 (rolling 50 eps) |
