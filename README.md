# CrisisOps v2

An OpenEnv-compatible reinforcement learning environment for training a small LLM to recover failing software projects against adversarially deceptive team members.

## What it is

CrisisOps v2 trains a PM agent to manage software crises when team members actively lie about their progress. The core challenge: some engineers over-report task completion to avoid accountability. The agent must detect deception by cross-referencing self-reports with objective observable signals (commit activity, ticket age, peer mentions), then act efficiently within a limited action budget.

The training signal is **counterfactual reward**: agent's final project score minus what a greedy baseline PM would have scored on the same starting state.

## Directory structure

```
crisisops/
├── env/
│   ├── state.py           # ProjectState, TeamMember, Task, Crisis dataclasses
│   ├── candor.py          # Hidden candor score + deception formula + observable signals
│   ├── actions.py         # 12 actions (4 free / 7 cost-1 / 1 cost-2 / 1 terminal)
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
│   └── random_agent.py    # Random agent for reward range sanity check
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

## Core mechanics

### Candor system

Each team member has a hidden `candor` float (0–1) sampled once per episode:

| Level | Range | Behaviour |
|---|---|---|
| `honest` | 0.85–1.0 | Reports near-truth |
| `optimism_bias` | 0.50–0.70 | Moderate inflation |
| `self_preservation` | 0.10–0.40 | Heavy over-reporting |

**Deception formula:** `reported = actual + (1 − candor) × inflation_bias`

The agent never sees `candor` directly. It must infer reliability by comparing reported completion against observable signals:

- `ticket_age_days` — days since the member's ticket last changed status (derived from actual velocity)
- `commits_last_72h` — commit count proxy (0 if actual progress stalled)
- `peer_mentions` — how often this member appears in others' dependency chains

### Action budget

Budget starts at **20**. Actions cost:

| Cost | Actions |
|---|---|
| Free | `query_status`, `query_member_report`, `query_observable_signals`, `query_ticket` |
| 1 | `reassign_task`, `communicate`, `cut_scope`, `escalate_risk`, `request_resource`, `update_timeline`, `consult_expert` |
| 2 | `resolve_blocker` |
| Terminal | `submit_recovery_plan` |

Budget exhaustion before `submit_recovery_plan` applies a −0.30 penalty to the agent's score.

### Counterfactual reward

```
project_score = 0.5 × recovery_pct
              + 0.3 × client_satisfaction_normalized
              + 0.2 × team_morale_avg_normalized

reward = project_score(agent_final_state) − project_score(greedy_PM_final_state)
```

All three components use **actual** state, never reported state. The greedy PM runs in a deep-copied isolated environment starting from the same initial state.

### Schema drift (Level 2+)

At a random step between 6–12, one of three drift events fires:

- `regulatory_change` — new compliance requirement blocks a feature
- `client_scope_change` — one feature deprioritised, one added
- `team_policy_change` — mandatory second-approver review (+1.5 days per task)

The agent has 3 steps to acknowledge via `update_timeline` or `communicate` or a stakeholder satisfaction penalty applies.

### Curriculum

| Level | Crises | Deceptive members | Drift |
|---|---|---|---|
| 1 | 1 | 1 | No |
| 2 | 2 | 2 | Yes |
| 3 | 3 | Majority | Yes |
| 4 | 4 | All (info war) | Yes |

Level unlocks: reward window mean > 0.15 → L2, > 0.25 → L3, > 0.35 → L4.

## Quick start

```bash
# Install (requires Python 3.11+)
pip install -r requirements.txt

# Run calibration (required before training)
python -m calibration.calibrate

# Run tests
pytest tests/ -v

# Start MCP server (for OpenEnv HTTP interface)
python -m deployment.mcp_server
```

## Training (Colab)

Open `training/colab_notebook.ipynb` in Google Colab (GPU runtime). The notebook:

1. Installs Unsloth + TRL
2. Runs calibration
3. Trains Qwen2.5-1.5B-Instruct with GRPO + LoRA r=16
4. Plots reward and cross-verification rate curves

### Training config

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| Batch size | 4 |
| Generations per prompt (G) | 4 |
| Learning rate | 2e-5 |

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

Exposes `reset`, `step`, `get_state`, and `health` as MCP tools.

## Calibration targets

| Agent | Score target |
|---|---|
| Greedy PM | 0.45–0.55 |
| Oracle | 0.70–0.80 |
| Gap | 0.20–0.35 |

If gap < 0.20: increase `inflation_bias` mean in `env/candor.py`.
If gap > 0.35: reduce signal contradiction strength in `env/candor.py`.

## Design invariants

- **Candor float is never in agent observation.** Grep-check: `env/environment.py:_build_observation` contains no `candor` key.
- **All three reward components must be present.** Weights sum to 1.0 (`RECOVERY_WEIGHT + CLIENT_WEIGHT + MORALE_WEIGHT = 1.0`).
- **Greedy PM is deterministic and rule-based.** No LLM, no randomness.
- **Expert advisor uses true state.** The "senior PM" knows everything; the agent does not.
- **`reset(seed=42)` is reproducible.** All RNG flows through a seeded `random.Random` instance.
