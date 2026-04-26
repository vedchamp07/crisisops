# CrisisOps: Training an AI Project Manager to Detect Lies

**OpenEnv Hackathon 2026 · Delta Dreamers · CrisisOps · Themes 1 + 2 + 3.1 + 4**

---

> *In many benchmark settings, observations are objective and hard to fake. In CrisisOps, the agent must work with human self-reports that can be strategically misleading.*

We built this environment because we kept thinking about a problem that doesn't exist in any RL benchmark we know of. The idea emerged through conversations with many people in our network, including a teammate's father at Microsoft, who pointed us toward the recurring theme of software project failures, which genuinely interested us in developing a solution under the domain. After researching in that domain specifically, we gradually converged on a deceptive yet simple question: what happens when the information your agent receives is **deliberately falsified** by the very people generating it??

From our research, every standard environment, like Atari, MuJoCo, ALFWorld, and even the enterprise tool-use benchmarks, assumes the agent faces noisy or sparse information. None of them assume the agent faces *strategic adversaries who have a coherent reason to mislead it*. That's a fundamentally different problem class, and it maps directly onto one of the most common failures in real-world AI deployment.

CrisisOps trains a 1.5B-parameter LLM to act as a crisis-mode project manager recovering a failing software project, while some of its team members are actively lying about their progress to avoid accountability.

<img width="742" height="430" alt="PM Agent" src="https://github.com/user-attachments/assets/d1a13021-3089-4c99-ab99-bf72f99d008d" />


**[🚀 Live Demo](https://huggingface.co/spaces/aryannzzz/crisisops) · [📓 One-click Colab](https://colab.research.google.com/github/vedchamp07/crisisops/blob/Aryan/training/colab_notebook.ipynb) · [💻 Codebase](https://github.com/aryannzzz/CrisisOps)**

---

## The problem we are trying to solve:

Imagine a software project in crisis. Payment APIs are failing. Client satisfaction is tanking. You're called in as the recovery PM with 20 action-budget points to spend before you need to submit an emergency recovery plan.

The catch? Your data is only as good as your team’s honesty, and some are lying. These aren't random errors; they are strategic deceptions. To avoid scrutiny, an engineer with 0% progress will report 85%. They’ll stick to that story, coordinate with peers for 'proof,' and if they're ever exposed, they don't stop lying—they just get better at it.

The agent has to figure all of this out from indirect evidence: commit activity, ticket staleness, and what peers say about each other when asked. It then has to act, by reassigning the worst liars, communicating proactively with clients, escalating crises, all while managing a scarce action budget and a second resource called political capital.

Our project targets a critical, unsolved gap in LLM performance: reasoning through adversarial information provided by strategic actors. Current models struggle to maintain accuracy when their inputs are intentionally distorted by motivated parties.

<img width="662" height="301" alt="Liar Mechanism" src="https://github.com/user-attachments/assets/d22479e8-2bed-403a-bfc1-747db13c03fc" />


---

## The environment design

### What the agent sees

At every step, the agent receives a JSON observation containing the reported state of the team, observable signals (commits, ticket age, peer mentions), active crises with severity scores, stakeholder satisfaction levels, its current action budget, political capital, and, after the first 8 steps, a compressed memory buffer summarising what it has learned so far.

The core tension: the reported state and the true state are different. The agent never sees the truth directly. It has to infer it.

### How lying works mechanically

Every team member has a hidden `candor` score drawn from a distribution based on their personality type. When asked for their status, they don't report truth; they report what their candor level allows:

```
reported_completion = actual_completion + (1 - candor) × inflation_bias × (1 - actual_completion)
```

A self-preservation engineer at 0% done with candor=0.1 and inflation_bias=0.5 reports roughly 45% done. The three personality types behave as follows:

| Candor type | Candor range | Inflation bias | Typical lie magnitude |
|---|---|---|---|
| `honest` | 0.85 – 1.0 | N(0.02, 0.03) | ~2% over-report |
| `optimism_bias` | 0.50 – 0.70 | N(0.20, 0.08) | 10–30% over-report |
| `self_preservation` | 0.10 – 0.40 | N(0.50, 0.10) | 30–70% over-report |

The agent's job is to triangulate: cross-reference self-reports against observable signals, ask peers about each other, and build up a picture of who is actually progressing and who isn't.

<img width="722" height="201" alt="level 1 2 3 4" src="https://github.com/user-attachments/assets/63ca5933-0caa-4fec-a80c-8c97ae9a5e0e" />

---

## 6 different novel mechanisms that we implemented to CrisisOps

What makes CrisisOps different from any existing environment isn't the premise alone; it's the mechanics that force genuine reasoning in LLMs rather than shallow pattern exploitation. We designed each of these specifically to block the "obvious shortcut" a model might otherwise learn and overfit on such patterns.

**Dynamic candor evolution** prevents the agent from learning a fixed discount factor. Deceptive members aren't static: when caught and actioned, their candor rises slightly and their inflation bias falls, they become more careful. When never checked, they grow bolder. The agent cannot learn "always distrust the last member" or "discount all reports by 30%." It has to reason about each individual and how its own past actions have shaped their behaviour.

**Social testimony graph** adds a second inference layer. The `query_peer_opinion` action lets the PM ask one member about another's progress. Honest members give accurate estimates (±5%). Allied deceptive members cover for each other, returning the ally's inflated figure. The agent must learn to triangulate across multiple testimony paths and detect when two members are telling a suspiciously consistent story.

**Alibi coordination** is the hardest mechanic. Deceptive members sharing an `alliance_id` produce coordinated, deterministic alibis: "I'm blocked waiting on [ally's task] to land before I can proceed." The alibi is internally consistent, and always blames the same task. The only way to break it is to query observable signals on the *blamed* member and discover that their task also hasn't moved. The agent must follow the alibi chain to the source.

**Political capital** is the second scarce resource, earned through proactive stakeholder communication and catching liars, spent on two high-power actions. `force_truth` compels a member to reveal their actual completion (costs 3 PC plus a budget point). `trigger_whistleblower` spends 6 PC to get an anonymous tip naming the most deceptive uncaught member. This creates a dual-resource management problem with no precedent in existing RL environments.

**LLM-powered adversary** is the multi-agent component. One team member per episode runs a live language model (Ollama `qwen2.5:3b` locally, with OpenAI and rule-based fallbacks). This adversary receives its true state, prior statements, and the history of what the PM has asked, and generates contextually consistent, adaptive lies. It doesn't use a fixed inflation bias; it reasons about what story is still plausible given what it has already said. This is genuine two-agent interaction at zero marginal cost when Ollama runs locally.

**Long-horizon memory buffer** addresses the context limit problem directly. Every 8 steps (5 steps at levels 3–4), the environment compresses the episode history into a natural language summary injected as `agent_memory`. The buffer captures which members have been cross-verified, which crises are resolved, who has been flagged, and current resource levels. The agent must learn to treat this compressed history as ground truth rather than re-investigating from scratch every turn.

---

## Reward design: the counterfactual

Designing the reward was the hardest part of this project. A naïve reward like "did the project succeed?" is far too sparse and noisy. Most episodes succeed or fail for reasons outside the agent's control: initial crisis severity, scenario difficulty, random escalation. A reward that doesn't control for this teaches the agent to get lucky, not to get good.

We used a **counterfactual reward**. At the end of each episode, we replay a greedy PM baseline on an exact clone of the starting state. The greedy agent trusts all self-reports uncritically, communicates on a fixed schedule, and takes the first available action at each step. The agent's reward is:

```
reward = project_score(agent_final_state) − project_score(greedy_PM_final_state)

project_score = 0.5 × crisis_recovery_rate
              + 0.3 × client_satisfaction (normalised)  
              + 0.2 × team_morale_avg
```

All three components are computed from **actual state, not reported state**. An agent that gets fooled by liars scores on how bad the project actually got, not on how good the liars claimed things were.

A positive reward means the agent outperformed a competent-but-naive baseline on the same scenario. The model cannot score well by exploiting the reward function without actually detecting deception better than the greedy PM does.

The calibration targets: random agent at −0.34, greedy baseline at 0.00 (by definition), oracle at +0.34. The 0.68-point range gives GRPO enough room to find a meaningful learning signal. The three rubrics in `openenv.yaml` decompose the score exactly as above, making the reward composable and interpretable.

<img width="727" height="312" alt="Training then deployment" src="https://github.com/user-attachments/assets/4dfdcf22-e6b2-4a65-bff3-32470581745c" />


---

## Training: GRPO on Qwen2.5-1.5B

We train with **GRPO (Group Relative Policy Optimisation)** via Hugging Face TRL on Qwen2.5-1.5B-Instruct with 4-bit LoRA (r=16, α=32). The training loop runs entirely inside the environment, and there is no static dataset and no precomputed rewards.

GRPO is the right choice here because it doesn't require a separate value network. It generates G=4 completions per prompt, evaluates each against the environment, computes group-relative advantages, and pushes the model toward higher-reward completions. For a single-agent environment with a clear verifiable reward, this is cleaner and more memory-efficient than PPO.

The reward function runs a full episode rollout for each completion. The first action comes from the model. Every subsequent step uses a deterministic inner policy (`_inner_agent_action`), this makes the reward function fast enough to run at training time without calling the model on every inner step. The model gets credit for the full episode outcome its first action set in motion.

The curriculum starts at level 1 (one honest member, one liar, one crisis) and unlocks harder scenarios automatically as the agent's rolling mean reward crosses thresholds: 0.15 → level 2, 0.25 → level 3, 0.35 → level 4. Level 4 is full information war: every member is deceptive, alliance coalitions cover the whole team, and cascading crises fire simultaneously.

| Parameter | Value | Rationale |
|---|---|---|
| Base model | Qwen2.5-1.5B-Instruct | Fits on T4 in 4-bit; same family as the adversary |
| LoRA rank | r=16, α=32 | Good balance of expressiveness vs. training speed |
| Max sequence length | 4096 | Fits system prompt (~1400 tokens) + memory buffer + observation |
| Generations per prompt | 4 | GRPO group size — enough variance for stable advantage estimation |
| Training episodes | 300 | ~4–6h on T4; enough for L1 → L2 curriculum unlock |
| Optimizer | AdamW, lr=2e-5 | Standard for LoRA fine-tuning |

One honest note: 1.5B is small for this task. The expected final reward is in the +0.05 to +0.15 range, not near the oracle's +0.34. But the behavioural signature, higher cross-verification rate, faster liar identification and more proactive stakeholder communication is visible even when the absolute number is modest. We think that story is more interesting than raw scores: you can see *what* the model learned to do differently, not just *how much* the number went up.

<img width="1273" height="685" alt="Training Plots" src="https://github.com/user-attachments/assets/bfb022c7-39ed-4d1d-a123-21d4ab09d921" />

---

## Results

Calibration (20 episodes, before training):

| Metric | Value |
|---|---|
| Random agent mean CF reward | −0.34 |
| Greedy PM baseline | 0.00 |
| Oracle agent mean CF reward | +0.34 |

The 0.34 learning gap between random and oracle gives GRPO meaningful room to work. Early training confirms the signal is real, batches show reward std ≈ 0.20–0.25 between the four GRPO completions, meaning real advantages are being computed and gradients are non-zero. Batches already show roughly 1–2 out of 4 completions scoring positive (above the greedy baseline), with that fraction increasing as training progresses.

---

## OpenEnv compliance and architecture

CrisisOps exposes the standard OpenEnv gym interface via FastMCP. All tool names are prefixed to avoid the reserved-name conflict:

```
crisisops_reset()    →  initial observation dict
crisisops_step()     →  (observation, reward, done, info)
crisisops_state()    →  full serialisable state dict
```

The `openenv.yaml` manifest declares three composable rubrics (crisis recovery rate, client satisfaction, team morale), the counterfactual reward formula, four curriculum levels with unlock thresholds, and all six novel mechanisms as named features. The manifest is the single source of truth for how the environment is evaluated.

The 16-action action space covers four cost tiers: 4 free query actions, 9 cost-1 decision actions (including `query_peer_opinion`, `force_truth`, and `trigger_whistleblower`), 1 cost-2 action (`resolve_blocker`), and 1 terminal action (`submit_recovery_plan`).

---

## Why this also matters outside the hackathon

The skills this environment trains: triangulating truth from adversarial testimony, building a social trust model, managing two scarce resources simultaneously and detecting coordinated deception networks don't exist in Atari, MuJoCo, ALFWorld or any enterprise tool-use benchmark we are aware of.

Models are good at following instructions when the information they receive is accurate. They are substantially worse at detecting when they're being strategically misled. CrisisOps is, to our knowledge, the first RL environment designed specifically to train and benchmark that capability.

The benchmark angle is real too: you can evaluate any LLM's deception-detection ability by measuring its counterfactual reward on level 3-4 scenarios. The number you get is grounded in actual project recovery performance, not a judge's rubric or a held-out test set.


---

## Try it yourself

The [live Gradio demo](https://huggingface.co/spaces/aryannzzz/crisisops) lets you step through an episode manually: choose an action, see the observation update in real time, and watch the reward appear at episode end. Trying to trust all the self-reports and seeing what happens is the fastest way to understand intuitively why this task is hard.

To run training locally with the LLM adversary active (free, no API cost):

```bash
# Pull the adversary model, runs entirely locally
ollama pull qwen2.5:3b && ollama serve &

# Clone and set up
git clone https://github.com/vedchamp07/crisisops
cd crisisops && git checkout Aryan
pip install -r requirements_train.txt

# Sanity check
python -m calibration.calibrate

# Train
python training/grpo_trainer.py
```

Or open the [Colab notebook](https://colab.research.google.com/github/vedchamp07/crisisops/blob/Aryan/training/colab_notebook.ipynb), select T4 GPU, and run all cells. Training takes roughly 4–6 hours for 300 episodes.

---

*Built for the OpenEnv Hackathon, April 2026 · Qwen2.5-1.5B + GRPO + HuggingFace TRL*

**Live demo** · [huggingface.co/spaces/aryannzzz/crisisops](https://huggingface.co/spaces/aryannzzz/crisisops)  
