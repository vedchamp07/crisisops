# CrisisOps — Start Full Training Run

## What you already know

- 22/22 implementation checks pass on this codebase
- `unsloth_compiled_cache/UnslothGRPOTrainer.py` exists — Unsloth already compiled its GRPO trainer on this machine in a previous debug run. Model loading works.
- Working dep stack (already installed in `.venv`): `trl==0.19.1`, `transformers==4.56.2`, Unsloth with CUDA
- All trainer fixes are in place: `max_seq_length=4096`, `use_cache=False`, HF Dataset conversion, GRPOConfig compat, compact JSON, FIX comments stripped from system prompt (~1410 tokens)
- GPU: RTX 3050 6GB

---

## Step 0 — Check the debug run from the previous session

First, check if the previous debug run (30 episodes) produced output:

```bash
ls -la outputs/ 2>/dev/null && \
ls -la outputs/debug_run/ 2>/dev/null && \
[ -f outputs/debug_run/reward_log.json ] && python -c "
import json
with open('outputs/debug_run/reward_log.json') as f: d = json.load(f)
print(f'Debug run episodes: {len(d)}')
rewards = [r[\"reward\"] for r in d]
print(f'Rewards: {rewards}')
print(f'All same: {len(set(round(r,3) for r in rewards))==1}')
" || echo "No debug_run/reward_log.json found"
```

**Interpret the result:**
- If `reward_log.json` exists with varying rewards → debug run worked. Skip to Step 2.
- If file doesn't exist → debug run crashed before finishing. Check `debug_run.log` if it exists: `tail -30 debug_run.log 2>/dev/null`
- If all rewards are exactly `-0.300` → model never learned to submit, see fix below

---

## Step 1 — Quick smoke test (2 episodes, confirms pipeline works end-to-end)

Run this before the full training:

```bash
TOKENIZERS_PARALLELISM=false .venv/bin/python -c "
from training.grpo_trainer import train
train(curriculum_level=1, num_episodes=2, output_dir='./outputs/smoke', seed=0)
print('SMOKE OK')
" 2>&1 | tee smoke.log
```

**Watch for these specific lines in the first 2 minutes:**
- `Unsloth: ...` — model loading (normal, takes ~60s)
- `Training log:` or a reward value printed — training actually ran
- Any traceback — fix it before full run

**If you see `past_key_values is None`:** The `use_cache=False` fix didn't stick. Open `training/grpo_trainer.py`, find the model load block, and confirm these two lines are there immediately after `FastLanguageModel.from_pretrained(...)`:
```python
model.config.use_cache = False
if hasattr(model, 'generation_config') and model.generation_config is not None:
    model.generation_config.use_cache = False
```

**If you see `Input length X exceeds max_length`:** The 4096 fix didn't stick. Confirm `max_seq_length=4096` in the `FastLanguageModel.from_pretrained(` call.

**If you see CUDA OOM:** Reduce at top of `training/grpo_trainer.py`:
```python
GRPO_BATCH_SIZE      = 2  # was 4
GRPO_NUM_GENERATIONS = 2  # was 4
```
Then retry.

---

## Step 2 — Start the full training run in tmux

Once smoke test passes (or if debug run already produced valid output), start full training:

```bash
# Create tmux session
tmux new-session -d -s crisisops \
  "cd $(pwd) && TOKENIZERS_PARALLELISM=false .venv/bin/python -c \
  'from training.grpo_trainer import train; train(curriculum_level=1, num_episodes=300, output_dir=\"./outputs/full_run\", seed=42)' \
  2>&1 | tee full_run.log; echo TRAINING_DONE >> full_run.log"

echo "Training running in tmux session 'crisisops'"
echo "Attach: tmux attach -t crisisops"
echo "Detach: Ctrl+B then D"
```

While it runs, open a second terminal and monitor:

```bash
# Monitor reward log as it builds (check every few minutes)
watch -n 30 'python -c "
import json, os
p = \"./outputs/full_run/reward_log.json\"
if os.path.exists(p):
    d = json.load(open(p))
    r = [x[\"reward\"] for x in d]
    if r:
        import statistics
        print(f\"Episodes: {len(r)}\")
        print(f\"Last 10 mean: {statistics.mean(r[-10:]):.4f}\")
        print(f\"Max so far: {max(r):.4f}\")
        print(f\"Levels: {sorted(set(x[\"level\"] for x in d))}\")
else:
    print(\"No reward_log.json yet\")
"'
```

---

## Step 3 — Capture results after training

When training finishes (or after at least 100 episodes), run the full results report:

```bash
.venv/bin/python - << 'EOF'
import json, os
import numpy as np

OUTPUT_DIR = './outputs/full_run'
log_path = os.path.join(OUTPUT_DIR, 'reward_log.json')

if not os.path.exists(log_path):
    print("ERROR: reward_log.json not found")
    exit(1)

with open(log_path) as f:
    log = json.load(f)

rewards = [r['reward'] for r in log]
levels  = [r['level']  for r in log]
n = len(rewards)

print("=== TRAINING RESULTS ===")
print(f"Episodes logged:     {n}")
print(f"Curriculum levels:   {sorted(set(levels))}")
print()
print(f"Mean (all):          {np.mean(rewards):.4f}")
print(f"Mean (first 25%):    {np.mean(rewards[:n//4]):.4f}")
print(f"Mean (last 25%):     {np.mean(rewards[3*n//4:]):.4f}")
print(f"Max reward:          {max(rewards):.4f}")
print(f"Min reward:          {min(rewards):.4f}")
print(f"Std dev:             {np.std(rewards):.4f}")
print()
first_half = np.mean(rewards[:n//2])
second_half = np.mean(rewards[n//2:])
print(f"First half mean:     {first_half:.4f}")
print(f"Second half mean:    {second_half:.4f}")
print(f"Trend:               {'IMPROVING' if second_half > first_half else 'FLAT/DECLINING'}")
print()
positive = sum(1 for r in rewards if r > 0)
print(f"Positive CF reward:  {positive}/{n} ({100*positive/n:.1f}%)")
print()
import os
checkpoints = sorted([d for d in os.listdir(OUTPUT_DIR) if d.startswith('checkpoint_ep')])
print(f"Checkpoints saved:   {checkpoints}")
print()
print("=== LAST 20 LINES OF LOG ===")
if os.path.exists('full_run.log'):
    lines = open('full_run.log').readlines()
    print(''.join(lines[-20:]))
EOF
```

---

## Step 4 — Generate final plots

```bash
.venv/bin/python - << 'EOF'
import json, os, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

OUTPUT_DIR = './outputs/full_run'
with open(os.path.join(OUTPUT_DIR, 'reward_log.json')) as f:
    log = json.load(f)

episodes = [r['episode'] for r in log]
rewards  = [r['reward']  for r in log]
levels   = [r['level']   for r in log]
n = len(rewards)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Reward curve
ax = axes[0]
ax.plot(episodes, rewards, alpha=0.2, color='#2563eb', linewidth=0.8, label='Raw reward')
if n > 5:
    sm = gaussian_filter1d(rewards, sigma=max(1, n//30))
    ax.plot(episodes, sm, color='#1d4ed8', linewidth=2.2, label='Smoothed')
ax.axhline(0, color='#dc2626', linestyle='--', linewidth=1.2, label='Greedy baseline (0)')
ax.set_xlabel('Training episode', fontsize=11)
ax.set_ylabel('Counterfactual reward (agent − greedy PM)', fontsize=11)
ax.set_title('CrisisOps — GRPO Training Curve', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Shade curriculum levels
level_colors = {1:'#eff6ff', 2:'#ecfdf5', 3:'#fefce8', 4:'#fff1f2'}
prev_ep, prev_lv = (episodes[0] if episodes else 0), (levels[0] if levels else 1)
for ep, lv in zip(episodes, levels):
    if lv != prev_lv:
        ax.axvspan(prev_ep, ep, alpha=0.3, color=level_colors.get(prev_lv,'#f5f5f5'))
        prev_ep, prev_lv = ep, lv
if episodes:
    ax.axvspan(prev_ep, episodes[-1], alpha=0.3, color=level_colors.get(prev_lv,'#f5f5f5'))

# Plot 2: Distribution shift
ax2 = axes[1]
first_q = rewards[:n//4]
last_q  = rewards[3*n//4:]
ax2.hist(first_q, bins=15, alpha=0.6, color='#dc2626',
         label=f'First 25% (mean={np.mean(first_q):.3f})')
ax2.hist(last_q,  bins=15, alpha=0.6, color='#2563eb',
         label=f'Last 25% (mean={np.mean(last_q):.3f})')
ax2.axvline(0, color='black', linestyle='--', linewidth=1.0, label='Greedy baseline')
ax2.set_xlabel('Counterfactual reward', fontsize=11)
ax2.set_ylabel('Episode count', fontsize=11)
ax2.set_title('Reward Distribution: Early vs Late Training', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, 'training_curve_final.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')

os.makedirs('plots', exist_ok=True)
shutil.copy(out, 'plots/reward_curve.png')
print('Copied to plots/reward_curve.png')
print('Commit with: git add plots/reward_curve.png && git commit -m "feat: training results"')
EOF
```

---

## Step 5 — What to paste back for diagnosis

After training (or after 50+ episodes if it's slow), paste back the **complete output of Step 3** plus:

```bash
# Also paste these two lines:
tail -5 full_run.log
ls -lh outputs/full_run/
```

---

## Failure modes and exact fixes

**All rewards exactly `-0.300` every episode:**
Model always exhausts budget without submitting. Print a raw completion to see what it outputs:
```python
# Add temporarily to reward_fn in grpo_trainer.py, first episode only:
if i == 0:
    print(f"[DEBUG] completion: {completion[:200]}")
```
If output is all freeform text (not JSON), the model isn't following the format. Fix: reduce `GRPO_TEMPERATURE` from 0.9 to 0.6 in `grpo_trainer.py`.

**All rewards exactly `0.000`:**
`_compute_reward` isn't running. Check that `env.reset(seed=ep_seed)` doesn't fail silently. Add `print(f"[DEBUG] reward_val={reward_val}")` after the while loop.

**Training stalls after episode 10 with no new log lines:**
The inner model.generate() call is hanging on a very long prompt. Check if any observation is > 3000 chars:
```python
import json
obs_str = format_observation_as_prompt(obs)
print(f"[DEBUG] obs chars: {len(obs_str)}")
```
If > 3000: use level 1 scenarios only (`from scenarios.level1 import get_random_level1_scenario`) and set `curriculum_level=1` for the full run.

**CUDA OOM mid-training:**
```python
GRPO_BATCH_SIZE      = 2
GRPO_NUM_GENERATIONS = 2
```
Restart training. Checkpoint from last save will not be auto-resumed — the curve will restart from 0 but the model weights are partially trained.

**Reward improves then collapses around episode 80-100:**
This is reward hacking — model learned to always submit immediately. Check if `submit_recovery_plan` frequency rises in action distribution. If yes, add to system prompt: "Never submit before at least 3 cross-verify actions."

---

## Interpreting results

| Pattern | Meaning | Action |
|---|---|---|
| Mean moves from -0.34 toward -0.10 over 300 eps | Learning is happening, curve is gradual | Normal for 1.5B — good result |
| Mean crosses 0 by episode 150+ | Model beat greedy PM baseline | Excellent — screenshot this |
| Flat at -0.30 entire run | Budget exhaustion every ep | Print raw completions, reduce temperature |
| Spiky with no trend | Training signal is too noisy | Increase `CURRICULUM_WINDOW` from 10 to 20 |
| Improves then collapses | Reward hacking | Strengthen system prompt with explicit anti-hack rules |

---

## After a successful run — commit everything

```bash
git add plots/reward_curve.png outputs/full_run/reward_log.json
git commit -m "feat: 300-episode GRPO training results"
git push origin main
git push hf main --force
```