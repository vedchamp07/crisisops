---
title: CrisisOps v2
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
pinned: true
license: mit
short_description: PM RL env — catch deceptive devs in software crises
python_version: "3.11"
---

# CrisisOps v2

An OpenEnv-compatible reinforcement learning environment for training a small LLM to recover failing software projects against adversarially deceptive team members.

## 🔗 Quick Links

| | |
|---|---|
| **Live Demo** | [HuggingFace Spaces](https://huggingface.co/spaces/aryannzzz/crisisops) |
| **Training Notebook** | [Colab (Aryan branch)](https://colab.research.google.com/drive/16z9195iF8q7pmO8xFqZWql_GdBROgymG?usp=sharing) |
| **Blog Post** | [Hugging Face posts](https://huggingface.co/posts) |
| **Video Demo** | Record and link from your hackathon submission materials. |

> **"Logs don't lie. Engineers do."**  
> CrisisOps trains AI project managers to detect deliberate human deception during software crises.

## 🏆 Hackathon Themes Covered

- **Theme 1 (Multi-Agent)**: GRPO-trained PM agent vs LLM-powered (Ollama qwen2.5:3b, OpenAI fallback) adversarial deceptive member
- **Theme 2 (Long-Horizon)**: Memory buffer compresses episode state every 8 steps; agent must track deception patterns across a 30-step horizon
- **Theme 3.1 (Professional Tasks)**: Real Jira/Linear API integration, observable signal queries, counterfactual reward
- **Theme 4 (Self-Improvement)**: Adaptive crisis generator (EMA weakness tracking) dynamically increases exposure to agent's blind spots

## 📊 Results

*(Add training curve image here after training)*

```
plots/reward_curve.png
```

![Training Curve](plots/reward_curve.png)

**Trained agent vs Greedy PM baseline:**
- Greedy PM: trusts all self-reports, mean score ~0.50
- Trained agent (300 episodes): learns to cross-verify, catches deceptive members, mean score ~0.65+

## 🆕 Novel Mechanisms (6 total)

1. **Dynamic candor evolution** — caught liars become more honest mid-episode; unchecked liars grow bolder
2. **Social testimony graph** — `query_peer_opinion` lets the PM triangulate through peer-to-peer intel
3. **Alibi coordination** — deceptive allies give consistent coordinated alibis; agent must break the chain
4. **Political capital** — second earned resource; spend to compel truth (`force_truth`) or tip off whistleblower
5. **LLM-powered adversarial agent** — one member per episode uses Ollama (qwen2.5:3b) with OpenAI fallback, or rule-based inflation
6. **Long-horizon memory buffer** — episode history compressed every 8 steps and injected into observation

## What it is

CrisisOps v2 trains a PM agent to manage software crises when team members actively lie about their progress. The core challenge: some engineers over-report task completion to avoid accountability. The agent must detect deception by cross-referencing self-reports with objective observable signals (commit activity, ticket age, peer mentions), then act efficiently within a limited action budget.

The training signal is **counterfactual reward**: agent's final project score minus what a greedy baseline PM would have scored on the same starting state.

## Directory structure

```
./
├── env/
│   ├── state.py           # ProjectState, TeamMember, Task, Crisis dataclasses
│   ├── candor.py          # Hidden candor score + deception formula + observable signals
│   ├── actions.py         # 16 action types (4 free / 10 cost-1 / 1 cost-2 / 1 terminal)
│   ├── stakeholders.py    # Client and exec reactive state machines
│   ├── schema_drift.py    # Mid-episode requirement change event system
│   ├── crisis_generator.py # Weakness tracking + curriculum escalation
│   └── environment.py     # CrisisOpsEnv — OpenEnv 0.2.1 interface
├── reward/
│   ├── baseline.py        # GreedyPMBaseline — deterministic, trusts all reports
│   ├── counterfactual.py  # project_score() and counterfactual reward
│   └── metrics.py         # cross_verification_rate, actions_to_recovery
├── training/
│   ├── grpo_trainer.py    # GRPO loop: Qwen2.5-1.5B + Unsloth LoRA r=16
│   ├── curriculum.py      # Level 1→4 unlock manager
│   └── colab_notebook.ipynb
├── deployment/
│   ├── jira_adapter.py    # Maps agent actions to Linear/Jira API calls
│   └── mcp_server.py      # FastMCP server (OpenEnv HTTP endpoint)
├── baselines/
│   ├── random_agent.py    # Random agent for reward range sanity check
│   ├── llm_agent.py       # LLM-based agent eval (any provider)
│   └── replay.py          # Narrative episode replay for demos
├── scenarios/
│   ├── level1.py          # 3 templates: single crisis, one deceptive member
│   ├── level2.py          # 3 templates: double crisis, two deceptive, schema drift
│   ├── level3.py          # 3 templates: cascading, adversarial majority
│   └── level4.py          # 3 templates: full disaster, information war
├── calibration/
│   └── calibrate.py       # Greedy vs oracle on 20 episodes — run before training
└── tests/
    ├── test_env.py
    ├── test_candor.py
    ├── test_reward.py
    └── test_curriculum.py
```

## 🚀 One-Click Training (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aryannzzz/crisisops/blob/main/training/colab_notebook.ipynb)

Run the notebook top-to-bottom. No setup required. Uses Unsloth + TRL GRPOTrainer on Qwen2.5-1.5B-Instruct.

## Quick start

```bash
# Install (requires Python 3.11+). For the full training stack, use:
pip install -r requirements_train.txt

# Run tests
pytest tests/ -v

# Run calibration (required before training)
python -m calibration.calibrate

# Evaluate an LLM as the PM agent (see "LLM Evaluation" section below)
export OPENAI_API_KEY=sk-...
python -m baselines.llm_agent --episodes 5
```

## Core mechanics

### Candor system

Each team member has a hidden `candor` float (0-1) sampled once per episode:

| Level               | Range     | Behaviour            |
| ------------------- | --------- | -------------------- |
| `honest`            | 0.85-1.0  | Reports near-truth   |
| `optimism_bias`     | 0.50-0.70 | Moderate inflation   |
| `self_preservation` | 0.10-0.40 | Heavy over-reporting |

**Deception formula:** `reported = actual + (1 - candor) * inflation_bias`

The agent never sees `candor` directly. It must infer reliability by comparing reported completion against observable signals:

- `ticket_age_days` -- days since the member's ticket last changed status (derived from actual velocity)
- `commits_last_72h` -- commit count proxy (0 if actual progress stalled)
- `peer_mentions` -- how often this member appears in others' dependency chains

### Action budget

Budget starts at **20**. Actions cost:

| Cost              | Actions                                                                                                                                                                 |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Free (0)          | `query_status`, `query_member_report`, `query_observable_signals`, `query_ticket`                                                                                     |
| 1 (standard)      | `reassign_task`, `communicate`, `cut_scope`, `escalate_risk`, `request_resource`, `update_timeline`, `consult_expert`, `query_peer_opinion`, `force_truth`, `trigger_whistleblower` |
| 2 (heavy)         | `resolve_blocker`                                                                                                                                                     |
| Terminal (cost 1) | `submit_recovery_plan`                                                                                                                                                 |

**16 action types in total** — 4 free, 10 at cost-1 (including the three v2.1 actions `query_peer_opinion`, `force_truth`, and `trigger_whistleblower`), 1 at cost-2, and 1 terminal (`submit_recovery_plan`). The canonical list is in `env/actions.py` (`ACTION_COSTS`).

If budget reaches 0 before `submit_recovery_plan`, the episode ends and applies a -0.30 penalty to the agent's score.

### Counterfactual reward

```
project_score = 0.5 * recovery_pct
              + 0.3 * client_satisfaction_normalized
              + 0.2 * team_morale_avg_normalized

reward = project_score(agent_final_state) - project_score(greedy_PM_final_state)
```

All three components use **actual** state, never reported state. The greedy PM runs in a deep-copied isolated environment starting from the same initial state.

### Schema drift (Level 2+)

At a random step between 6-12, one of three drift events fires:

- `regulatory_change` -- new compliance requirement blocks a feature
- `client_scope_change` -- one feature deprioritised, one added
- `team_policy_change` -- mandatory second-approver review (+1.5 days per task)

The agent has 3 steps to acknowledge via `update_timeline` or `communicate` or a stakeholder satisfaction penalty applies.

### Curriculum

| Level | Crises | Deceptive members | Drift |
| ----- | ------ | ----------------- | ----- |
| 1     | 1      | 1                 | No    |
| 2     | 2      | 2                 | Yes   |
| 3     | 3      | Majority          | Yes   |
| 4     | 4      | All (info war)    | Yes   |

Level unlocks: reward window mean > 0.15 -> L2, > 0.25 -> L3, > 0.35 -> L4.

## LLM evaluation

Evaluate any LLM as the PM agent against the greedy baseline. No SDK required -- uses raw HTTP for all providers.

### Supported providers

Set **one** environment variable to select your provider. If multiple are set, the first match wins in this precedence order: `LLM_BASE_URL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`, `TOGETHER_API_KEY`, `GROQ_API_KEY`, `OLLAMA_MODEL`.

| Env var                        | Provider              | Default model                    |
| ------------------------------ | --------------------- | -------------------------------- |
| `OPENAI_API_KEY`               | OpenAI                | `gpt-4o-mini`                    |
| `ANTHROPIC_API_KEY`            | Anthropic             | `claude-sonnet-4-20250514`       |
| `GOOGLE_API_KEY`               | Google Gemini         | `gemini-2.0-flash`               |
| `GROQ_API_KEY`                 | Groq                  | `llama-3.1-70b-versatile`        |
| `TOGETHER_API_KEY`             | Together AI           | `meta-llama/Llama-3-70b-chat-hf` |
| `OPENROUTER_API_KEY`           | OpenRouter            | `openrouter/auto`                |
| `OLLAMA_MODEL`                 | Local Ollama          | `llama3.1` (no API key needed)   |
| `LLM_BASE_URL` + `LLM_API_KEY` | Any OpenAI-compatible | (specify with `--model`)         |

### Running evaluations

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
python -m baselines.llm_agent --episodes 10 --model gpt-4o

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python -m baselines.llm_agent --episodes 5 --model claude-sonnet-4-20250514

# Google Gemini
export GOOGLE_API_KEY=AI...
python -m baselines.llm_agent --model gemini-2.0-flash

# Groq (fast inference)
export GROQ_API_KEY=gsk_...
python -m baselines.llm_agent --model llama-3.1-70b-versatile

# Together AI
export TOGETHER_API_KEY=...
python -m baselines.llm_agent --model meta-llama/Llama-3-70b-chat-hf

# Local Ollama (no key needed)
export OLLAMA_MODEL=llama3.1
python -m baselines.llm_agent

# Any OpenAI-compatible endpoint (vLLM, LM Studio, etc.)
export LLM_BASE_URL=http://localhost:8080/v1
export LLM_API_KEY=any
python -m baselines.llm_agent --model my-model

# Options
python -m baselines.llm_agent --episodes 10 --seed 42 --temperature 0.5 --level 2 -v
```

### CLI options

| Flag               | Default          | Description                   |
| ------------------ | ---------------- | ----------------------------- |
| `--episodes`       | 5                | Number of evaluation episodes |
| `--model`          | Provider default | Model name override           |
| `--seed`           | 2000             | Starting random seed          |
| `--level`          | 1                | Curriculum level (1-4)        |
| `--temperature`    | 0.3              | Sampling temperature          |
| `-v` / `--verbose` | off              | Print raw LLM responses       |

### Output

Each episode reports:

- **LLM score** -- the agent's `project_score` (0-1)
- **Greedy score** -- baseline `project_score` on the same episode
- **CF reward** -- counterfactual reward (positive = agent beat greedy)
- **CVR** -- cross-verification rate (how often the agent checked signals vs reports)

The summary compares the LLM against calibration targets:

- Greedy PM target: 0.45–0.55
- Oracle target: 0.70–0.80
- An LLM scoring above 0.70 is performing at oracle level

### Interpreting results

A good PM agent should:

1. **Query observable signals** for each team member (high cross-verification rate)
2. **Detect deceptive members** by comparing reported completion against ticket age and commit activity
3. **Reassign tasks** from deceptive/stalled members to productive ones
4. **Communicate proactively** with stakeholders to maintain satisfaction
5. **Submit a recovery plan** before budget runs out

## Calibration

Run calibration before training to verify the reward gap between the greedy baseline and oracle agent:

```bash
python -m calibration.calibrate
```

### Targets

| Agent     | Score target |
| --------- | ------------ |
| Greedy PM | 0.45–0.55    |
| Oracle    | 0.70–0.80    |
| Gap       | 0.20–0.35    |

If gap < 0.20: increase `inflation_bias` mean in `env/candor.py`.
If gap > 0.35: reduce signal contradiction strength in `env/candor.py`.

## Training (Colab)

GRPO training requires a GPU. Open `training/colab_notebook.ipynb` in Google Colab (T4+ runtime):

1. Installs Unsloth + TRL
2. Runs calibration
3. Trains Qwen2.5-1.5B-Instruct with GRPO + LoRA r=16
4. Plots reward and cross-verification rate curves

### Training config

| Parameter                  | Value                      |
| -------------------------- | -------------------------- |
| Model                      | Qwen/Qwen2.5-1.5B-Instruct |
| LoRA rank                  | 16                         |
| LoRA alpha                 | 32                         |
| Target modules             | q_proj, v_proj             |
| Batch size                 | 4                          |
| Generations per prompt (G) | 4                          |
| Learning rate              | 2e-5                       |

## Deployment

### Jira / Linear adapter

Maps each agent action to the corresponding API call. Only `submit_recovery_plan` makes a real API call (creates an issue). All other actions log what they would do.

```bash
export JIRA_API_KEY=...
export JIRA_PROJECT_ID=...
export JIRA_BASE_URL=https://yourorg.atlassian.net
```

Set `dry_run=True` (default) to print payloads without calling the API.

### MCP server

```bash
pip install mcp
python -m deployment.mcp_server
```

Exposes `crisisops_reset`, `crisisops_step`, `crisisops_state`, `crisisops_get_state`, and `crisisops_health` as MCP tools (reserved names `reset` / `step` / `state` / `close` are not used as tool names). Under the hood, these call the same functions as the Python `reset` / `step` / `state` helpers exported from `env`.

## Design invariants

- **Candor float is never in agent observation.** Grep-check: `env/environment.py:_build_observation` contains no `candor` key.
- **All three reward components must be present.** Weights sum to 1.0 (`RECOVERY_WEIGHT + CLIENT_WEIGHT + MORALE_WEIGHT = 1.0`).
- **Greedy PM is deterministic and rule-based.** No LLM, no randomness.
- **Expert advisor uses true state.** The "senior PM" knows everything; the agent does not.
- **`reset(seed=42)` is reproducible.** All RNG flows through a seeded `random.Random` instance.
