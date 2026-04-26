# CrisisOps v2: Training an AI to Detect Deception in Failing Software Projects

OpenEnv Hackathon · Theme 3.1 + 1 + 2 + 4

> "Logs don't lie. Engineers do." — Training a small LLM to act as a crisis-mode project manager who can't trust what its own team tells it.

- Theme 1 · Multi-Agent
- Theme 2 · Long-Horizon
- Theme 3.1 · Professional Tasks
- Theme 4 · Self-Improvement

## Quick links

- Live demo: [HuggingFace Spaces](https://huggingface.co/spaces/aryannzzz/crisisops)
- Training notebook: [Google Colab](https://colab.research.google.com/github/aryannzzz/crisisops/blob/main/training/colab_notebook.ipynb)
- Codebase: [GitHub](https://github.com/aryannzzz/crisisops)

## Why this problem matters

Every few weeks, a high-stakes engineering project hits a crisis. Deadlines slip. Bugs compound. Pressure mounts. And in that pressure, team members start shading the truth — "I'm 90% done" becomes an aspirational statement, not a factual one.

A good project manager in this situation doesn't just take reports at face value. They cross-reference what they're told with observable evidence: *when did the ticket last update? have there been any commits? what do peers say?* They triangulate. They build a picture of truth from indirect signals — exactly the kind of multi-signal, adversarial reasoning that LLMs struggle with out of the box.

> "Standard RL environments (Atari, MuJoCo, Kube SRE) have no deceptive actors. CrisisOps is the first environment built explicitly to train an agent under information warfare."

CrisisOps v2 is a reinforcement learning environment that trains a small LLM (Qwen2.5-1.5B via GRPO) to do exactly this: act as a project manager recovering a failing software project while detecting and managing team members who actively lie about their progress.

| Distinct actions the PM agent can take | Novel mechanisms not in any prior RL env | Curriculum levels, from 1 liar to information war |
| --- | --- | --- |
| 16 | 6 | 4 |

## The environment: what does the agent see, do, and get rewarded for?

### Setup

Each episode, the agent manages a 4–6 person engineering team through a software crisis. The team has tasks, crises to resolve, a client watching the project, and executive stakeholders who react to how well the PM communicates. Some team members are honest. Some are not.

*Figure 1. One CrisisOps episode. The PM agent interacts with an adversarial team under partial observability. Reward is counterfactual: what the agent scored minus what a greedy baseline would have scored.*

### Partial observability and the deception problem

The core difficulty: the agent never sees the truth directly. Each team member has a hidden `candor` score (0–1). When queried, they report their progress not as it is, but as:

```text
reported = actual_completion + (1 - candor) × inflation_bias
```

A self-preservation member with candor=0.1 and inflation_bias=0.5 will report 45% done when they're actually 0% done. The agent has to figure this out by checking observable signals — commit activity, ticket staleness, peer mentions — and triangulating against the self-reports.

| Candor level | Range | Inflation bias | Typical lie magnitude |
| --- | --- | --- | --- |
| `honest` | 0.85–1.0 | N(0.02, 0.03) | ~2% over-report |
| `optimism_bias` | 0.50–0.70 | N(0.20, 0.08) | 10–30% over-report |
| `self_preservation` | 0.10–0.40 | N(0.50, 0.10) | 30–70% over-report |

## Six novel mechanisms

What makes CrisisOps genuinely different from existing environments isn't the premise — it's the mechanics that force the agent to reason carefully rather than exploit shallow patterns.

- **Dynamic candor evolution**: Deceptive members aren't static. When caught, their candor rises and inflation drops — they become more honest mid-episode. When ignored, they grow bolder. The environment is game-theoretic: the agent's investigation strategy shapes the adversary's future behavior.
- **Social testimony graph**: The `query_peer_opinion` action lets the PM ask one member about another's progress. Honest members give accurate estimates. Allied deceptive members cover for each other. A trained agent learns to use this as a cross-verification tool.
- **Alibi coordination**: Deceptive members with the same `alliance_id` coordinate alibis, pointing blame at each other's tasks. The agent must detect the circular dependency by querying observable signals on the "blamed" member — breaking the alibi chain.
- **Political capital**: A second resource alongside the action budget. Earned through proactive stakeholder communication and catching liars. Spent on high-power actions: `force_truth` compels a member to tell the truth; `trigger_whistleblower` exposes the worst liar to the team.
- **LLM-powered adversary**: One team member per episode runs a live LLM (Ollama qwen2.5:3b, with OpenAI and rule-based fallbacks). This adversary reads its true state and prior statements to generate contextually consistent, adaptive lies. Genuine multi-agent RL at zero marginal API cost when running locally.
- **Long-horizon memory buffer**: Every 8 steps (5 at level 3–4), the environment compresses episode history into a natural language summary injected into the observation as `agent_memory`. Forces the agent to reason about deception patterns across an extended horizon beyond the immediate context.

*Figure 2. The trained agent's deception-detection pipeline. Paid actions spend from the 20-point action budget (free queries do not), so the agent must decide how deeply to investigate each member.*

## Reward design: counterfactual scoring

A naive reward for this task would be: did the project succeed? But that reward is far too sparse — most episodes succeed or fail for reasons outside the agent's control (scenario difficulty, initial crisis severity). We need a reward that measures the agent's *relative contribution*.

CrisisOps uses a **counterfactual reward**: at the end of each episode, we run a greedy PM baseline on the same starting state. The reward is:

```text
reward = project_score(agent_final_state) − project_score(greedy_PM_final_state)

project_score = 0.5 × crisis_recovery_rate
              + 0.3 × client_satisfaction_normalized
              + 0.2 × team_morale_avg
```

All three components are computed from *actual state, not reported state*. The greedy baseline trusts every self-report uncritically — it reassigns when reports look bad, communicates on schedule, and takes the first available action. It scores around 0.50 on average.

> **Calibration targets**  
> The environment is calibrated so: Random agent ≈ −0.34 | Greedy PM ≈ +0.00 (baseline) | Oracle ≈ +0.34. A positive counterfactual reward means the agent outperformed the greedy PM. This range gives the training signal room to breathe.

*Figure 3. The counterfactual reward landscape. A score of 0 means the agent performed identically to the greedy baseline. Training pushes the distribution rightward.*

## Training pipeline: GRPO on Qwen2.5-1.5B

We train using **GRPO** (Group Relative Policy Optimization) via Hugging Face TRL and Unsloth, on Qwen2.5-1.5B-Instruct with LoRA (r=16, α=32). The training loop runs entirely inside the environment — no static dataset, no precomputed rewards.

> **Why GRPO?**  
> GRPO is ideal for single-agent RL with LLMs. Unlike PPO, it doesn't require a separate value network. It generates G=4 completions per prompt, ranks them by reward, and pushes the model toward higher-reward behaviors. The multi-step rollout inside the reward function gives multi-action trajectory feedback.

### The training loop, concretely

For each training step:

```text
1. Sample scenario from curriculum level 1–4 (CrisisGenerator)
2. Run the LLM agent for up to 30 steps per episode (inner rollout in reward_fn)
3. Compare final state to greedy PM on same starting scenario
4. Counterfactual reward → GRPO update
5. Every 10 episodes: check mean CF reward for curriculum unlock
```

The curriculum starts at level 1 (one honest member, one self-preservation liar, one crisis). As the agent's mean reward crosses thresholds, harder scenarios unlock automatically — up to level 4 (adversarial majority, cascading crises, information war).

### Model and training configuration

| Parameter | Value | Rationale |
| --- | --- | --- |
| Base model | Qwen2.5-1.5B-Instruct | Fits on T4 GPU; same family as adversary (qwen2.5:3b) |
| LoRA rank | r=16, α=32 | Balance between expressiveness and training speed |
| Max sequence length | 4096 | Fits system prompt + memory buffer + observation |
| Generations per prompt (G) | 4 | GRPO group size — enough variance for stable updates |
| Training episodes | 300 | ~6h on T4; sufficient for L1 → L2 curriculum unlock |
| Optimizer | AdamW, lr=2e-5 | Standard for LoRA fine-tuning |

*Figure 4. Curriculum levels, unlocked automatically as the agent's mean counterfactual reward crosses thresholds. Each level adds adversarial complexity.*

## What does the agent actually learn?

The most intuitive way to see the training effect is to compare the same scenario played by an untrained agent vs a trained one. The numbers below are from test episodes at level 1.

*Figure 5. Same scenario (Bella is a self-preservation liar, actual completion 15%, reported 85%). The trained agent cross-verifies with signals and peer testimony before acting. The untrained agent trusts the report.*

The critical behavioral shift is in step 2: the trained agent learns to follow up a suspicious self-report with `query_observable_signals` before taking any consequential action. This cross-verification rate — the ratio of signal queries to report queries — is the clearest behavioral signature of deception detection having been learned.

> **Note on model size**  
> At 1.5B parameters, Qwen2.5 is operating at the lower bound of capability for this task. Expect the reward curve to trend upward toward 0 to +0.10 range rather than reaching oracle performance. The behavioral signatures (higher cross-verify rate, faster liar identification) are clear even when the absolute reward improvement is modest. Upgrading to Qwen2.5-3B roughly doubles training VRAM requirements but substantially improves final performance.

## Architecture and OpenEnv compliance

CrisisOps exposes the standard OpenEnv interface via a FastMCP server, with wrappers to avoid reserved name conflicts:

```text
crisisops_reset()   → initial observation dict
crisisops_step()    → (observation, reward, done, info)
crisisops_state()   → full serializable state dict
```

The full action set covers 4 tiers of cost:

| Cost | Actions |
| --- | --- |
| Free (0) | query_status, query_member_report, query_observable_signals, query_ticket |
| 1 budget point | reassign_task, communicate, cut_scope, escalate_risk, request_resource, update_timeline, consult_expert, query_peer_opinion, force_truth (also 3 PC), trigger_whistleblower (also 6 PC) |
| 2 budget points | resolve_blocker |
| Terminal | submit_recovery_plan (ends episode; costs 1 budget point) |

Political capital (PC) is a separate meter from the action budget; it is earned via stakeholder updates and catching liars, and spent on truth-forcing actions.

*Figure 6. System architecture. Training and deployment are cleanly separated. The environment is shared; only the interface changes between training loop and MCP server.*

## Results

Below are the calibration numbers from a 20-episode calibration run (before full training):

| Metric | Value |
| --- | --- |
| Random agent mean CF reward | −0.34 |
| Greedy PM baseline (reference) | 0.00 |
| Oracle agent mean CF reward | +0.34 |

The 0.34 learning gap (oracle minus random) gives GRPO enough reward signal to drive meaningful behavioral change. Full training curves from the Colab run will be added here as the run completes.

> **Reward curve placeholder**  
> Training is ongoing. The Colab notebook (link above) saves training_curve_final.png and reward_log.json automatically. The expected trajectory: episodes 0–50 mostly negative, 50–150 crossing zero and stabilizing, 150–300 approaching +0.10 to +0.15 at level 1/2.

## How to try it

### Run the live demo

The [HuggingFace Spaces demo](https://huggingface.co/spaces/aryannzzz/crisisops) lets you step through an episode interactively. Choose an action from the dropdown, see the observation update in real time, and watch the reward accumulate.

### Train locally with Ollama (free, no API cost)

```bash
# 1. Pull the adversary model (free, runs locally)
ollama pull qwen2.5:3b

# 2. Clone and install
git clone https://github.com/aryannzzz/crisisops
pip install -r requirements_train.txt

# 3. Calibrate (sanity check)
python -m calibration.calibrate

# 4. Train
python training/grpo_trainer.py
```

### One-click Colab training

Open [the Colab notebook](https://colab.research.google.com/github/aryannzzz/crisisops/blob/main/training/colab_notebook.ipynb), select a T4 GPU runtime, and run all cells. Training takes ~4 hours for 300 episodes.

## Why this matters beyond the hackathon

The capability being trained here — reasoning about adversarial information in a high-stakes collaborative environment — is one of the core gaps in current LLM behavior. Models are good at following instructions when the information they receive is accurate. They are much worse at detecting when they're being misled by a strategic actor with a coherent motivation.

CrisisOps is the first RL environment specifically designed to close that gap. The skills it trains — cross-verifying reports against observable signals, building social testimony graphs, managing two scarce resources simultaneously, detecting coordinated deception networks — don't exist in Atari, MuJoCo, ALFWorld, or any standard benchmark.

This makes it both a training environment and a benchmark: you can evaluate any LLM's deception-detection capability by measuring its counterfactual reward on level 3–4 scenarios, and get a number that means something about real-world professional judgment under adversarial conditions.

> "A messy but ambitious environment with real training evidence beats a polished but boring one." — OpenEnv Hackathon judging guide

## All resources

- Live demo: [huggingface.co/spaces/aryannzzz/crisisops](https://huggingface.co/spaces/aryannzzz/crisisops)
- Codebase: [github.com/aryannzzz/crisisops](https://github.com/aryannzzz/crisisops)
- Training notebook: [One-click Colab](https://colab.research.google.com/github/aryannzzz/crisisops/blob/main/training/colab_notebook.ipynb)
- Demo video: Add YouTube URL after recording

Built for the OpenEnv Hackathon, April 2026 · IIT Madras · Qwen2.5-1.5B + GRPO + Unsloth + TRL