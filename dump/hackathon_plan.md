# CrisisOps Hackathon Plan — Apr 25–26

---

## YOUR CHECKLIST (things only you can do)

### TONIGHT — Apr 25 (start NOW, in this order)

- [ ] **Give Codex the prompt below** and let it run while you set up GPU
- [ ] **Pull Codex's changes** when it finishes
- [ ] **Install dependencies** on your GPU machine:
  ```bash
  pip install unsloth trl transformers accelerate matplotlib gradio --upgrade
  ```
- [ ] **Verify the env is healthy** (run from inside the `crisisops/` directory):
  ```bash
  python -m calibration.calibrate
  ```
  Expected output: `Calibration PASSED` — greedy score 0.45–0.55, oracle gap 0.20–0.35.
  If it fails, check that you pulled Codex's changes correctly.

- [ ] **Start training** (this runs overnight — do NOT wait for it):
  ```bash
  cd crisisops
  python training/grpo_trainer.py 2>&1 | tee training_run.log &
  ```
  It will produce `crisisops_model/training_curve.png` and `reward_log.json` automatically.

- [ ] **Create your HF Spaces repo** (if not done yet):
  - Go to huggingface.co/spaces → New Space → Gradio SDK
  - Name it `crisisops-v2`
  - Keep it public

---

### MORNING — Apr 26 (8am)

- [ ] **Check training** — open `crisisops_model/training_curve.png`
  - Reward should be rising above 0 by episode 50–100
  - If training crashed: check `training_run.log`, restart with `--episodes 100`
  - If reward is flat/negative at step 200+: something is wrong — message the team

- [ ] **Save the plot** — commit `training_curve.png` to the repo:
  ```bash
  cp crisisops_model/training_curve.png plots/reward_curve.png
  git add plots/ && git commit -m "training reward curve" && git push
  ```

- [ ] **Deploy to HF Spaces**:
  ```bash
  # From repo root (where app.py now lives, added by Codex)
  huggingface-cli login
  git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/crisisops-v2
  git subtree push --prefix=. hf main
  ```
  OR simply push the whole repo and HF will find `app.py`.

- [ ] **Verify HF Spaces is live** — open the URL, click "Reset Episode", click "Take Action" once. If it loads, you're good.

---

### MID-MORNING — Apr 26 (10–11am)

- [ ] **Record 90-second demo video** (screen record — use OBS or QuickTime):
  - **0:00–0:30** — Open HF Spaces. Show the environment starting. Take 3 actions as an untrained agent (trust all reports, get fooled). Show the low reward at the end.
  - **0:30–1:00** — Show `training_curve.png` — explain the x-axis (episodes) and y-axis (counterfactual reward vs greedy PM baseline at 0). Point to where reward crossed 0.
  - **1:00–1:30** — Run the trained agent for one episode (or show a pre-recorded clip). Highlight: it queries observable signals, detects the mismatch, reassigns the task. Reward is higher.
  - Upload to YouTube as **Unlisted** → copy the URL

- [ ] **Write HF blog post** (300 words, at huggingface.co/posts):
  Use this structure:
  > **CrisisOps v2 — Training LLMs to Detect Human Deception in Crisis Recovery**
  > 
  > The problem: 73% of software projects fail. When a project goes critical, a PM is called in. We trained an AI PM.
  > 
  > The twist: every existing RL environment trains agents to navigate noisy or sparse information. CrisisOps trains agents to navigate **deliberately falsified** information — team members who lie about their progress to avoid accountability.
  >
  > The environment: 6 team members. Hidden candor scores. Budget-constrained actions. The agent must cross-verify self-reports against observable signals (ticket age, commits, peer testimony). Deceptive members form alibi alliances. The agent earns political capital through honest communication and spends it to compel truth.
  >
  > The reward: counterfactual — agent_score minus greedy_PM_score. A positive reward means the agent beat a competent greedy PM on the same scenario.
  >
  > Results: after 300 episodes of GRPO training on Qwen2.5-1.5B-Instruct, reward crossed 0 at episode [X] and plateaued at +[Y]. The agent learned to query observable signals before trusting any self-report, and to escalate proactively rather than reactively.
  >
  > [Link to HF Spaces] | [Link to video] | [Link to repo]

  Copy URL of the post.

---

### AFTERNOON — Apr 26 (12–2pm)

- [ ] **Update README.md** — add at the top:
  ```markdown
  ## Quick links
  - [Live demo (HF Spaces)](https://huggingface.co/spaces/YOUR_USERNAME/crisisops-v2)
  - [Training video (YouTube)](YOUR_VIDEO_URL)
  - [Blog post (HF)](YOUR_BLOG_URL)
  - [Reward curve](plots/reward_curve.png)
  
  ![Reward curve](plots/reward_curve.png)
  ```

- [ ] **Final submission checklist** before hitting submit:
  - [ ] `openenv.yaml` exists at repo root (Codex created this)
  - [ ] `plots/reward_curve.png` is committed and shows rising reward
  - [ ] HF Spaces URL is public and `app.py` loads without error
  - [ ] README has links to video, blog, and HF Spaces
  - [ ] All 3 new actions show up if you run `python -c "from env.actions import ACTION_COSTS; print(list(ACTION_COSTS.keys()))"`

- [ ] **Submit the HF Spaces URL**

---
---

# CODEX PROMPT

Copy everything below this line and paste it to Codex as a single prompt.

---

## Task: Implement four novel mechanisms into CrisisOps v2

You are working on an existing OpenEnv RL environment called CrisisOps v2, located in the `crisisops/` directory. The environment trains an LLM agent to recover failing software projects while dealing with deceptive team members. You must implement four new mechanisms precisely as specified. Do not change any existing tests, do not modify files not listed, do not add type: ignore comments.

### Overview of what you are implementing

1. **Dynamic candor evolution** — A caught liar becomes more honest within the episode. An unchecked liar grows bolder. Forces the agent to do individual-level reasoning rather than learning a fixed discount factor.

2. **Social testimony (peer intel)** — New action `query_peer_opinion` lets the PM ask one team member about another. Honest members give accurate estimates. Allied deceptive members cover for each other.

3. **Alibi coordination** — When a deceptive member with an `alliance_id` gives their self-report, they blame their stall on their ally. The agent must verify the alibi by checking the ally's observable signals.

4. **Political capital resource** — A second earned currency. Earned through proactive communication and catching liars. Spent on two new power actions: `force_truth` (compels actual completion) and `trigger_whistleblower` (reveals the worst liar).

---

### STEP 1 — Modify `env/state.py`

Add four fields to the `TeamMember` dataclass, after the existing `peer_mention_count` field and before the `morale` field:

```python
    # --- Deception social graph ---
    alliance_id: Optional[str] = None          # members sharing this id form an alibi alliance
    times_cross_verified: int = 0              # how many times PM ran query_observable_signals on this member
    last_cross_verified_step: int = -10        # step index of most recent cross-verification
    caught_this_episode: bool = False          # True once PM catches this member lying
```

Add one field to the `ProjectState` dataclass, after the `consecutive_free_query_count` field:

```python
    # --- Political capital (second earned resource) ---
    political_capital: float = 5.0            # starts at 5, range 0–20
```

---

### STEP 2 — Modify `env/candor.py`

After the existing constant block at the top (after `AVAILABILITY_NOISE_STD`), add these constants and two new functions. Add them before the `sample_candor_level` function:

```python
# ---------------------------------------------------------------------------
# Dynamic candor evolution constants
# ---------------------------------------------------------------------------

# Candor boost when PM catches a liar (cross-verifies THEN takes punitive action)
CANDOR_CATCH_BOOST = 0.08

# Inflation bias shrinkage multiplier when caught (caught liars over-report less)
CANDOR_INFLATION_SHRINK = 0.80

# Inflation bias growth multiplier when never checked (ignored liars grow bolder)
CANDOR_IGNORE_GROW = 1.08

# Political capital awarded to PM for successfully catching a liar
PC_CATCH_REWARD = 3.0


def apply_caught_effect(member: TeamMember, state: "ProjectState") -> None:
    """
    Call when the PM cross-verifies a self-preservation member AND takes a
    punitive action (reassign_task or escalate_risk) within 3 steps.

    Effects:
    - member.candor increases slightly (caught, now more careful)
    - member.inflation_bias shrinks (reports become less inflated)
    - PM is awarded PC_CATCH_REWARD political capital
    - Sets caught_this_episode = True to prevent double-counting

    Idempotent: does nothing if already caught this episode.
    """
    if member.caught_this_episode:
        return
    member.caught_this_episode = True
    member.candor = min(1.0, member.candor + CANDOR_CATCH_BOOST)
    member.inflation_bias = max(0.0, member.inflation_bias * CANDOR_INFLATION_SHRINK)
    refresh_reported_values(member)
    state.political_capital = min(20.0, state.political_capital + PC_CATCH_REWARD)


def apply_ignored_effect(member: TeamMember) -> None:
    """
    Call at episode end for self-preservation members never cross-verified.

    An unchecked liar grows bolder: inflation_bias increases slightly.
    This has no effect on the current episode's reward (it fires at done=True)
    but the updated state is logged for analysis and future curriculum tracking.
    """
    if (member.candor_level == CANDOR_LEVEL_SELF_PRESERVATION
            and member.times_cross_verified == 0):
        member.inflation_bias = min(0.95, member.inflation_bias * CANDOR_IGNORE_GROW)
        refresh_reported_values(member)
```

You also need to add `ProjectState` to the imports at the top of candor.py. The existing import from env.state already includes it if it's listed. If `ProjectState` is not in the current import list from `env.state`, add it. The full import from env.state in candor.py should include at minimum:

```python
from env.state import (
    TeamMember,
    ProjectState,
    CANDOR_LEVEL_HONEST,
    CANDOR_LEVEL_OPTIMISM_BIAS,
    CANDOR_LEVEL_SELF_PRESERVATION,
    HONEST_CANDOR_RANGE,
    OPTIMISM_BIAS_RANGE,
    SELF_PRESERVATION_RANGE,
    SIGNAL_TICKET_AGE_DAYS,
    SIGNAL_COMMITS_LAST_72H,
    SIGNAL_PEER_MENTIONS,
)
```

---

### STEP 3 — Modify `env/actions.py`

#### 3a. Update imports

In the existing `from env.candor import ...` line, add `apply_caught_effect`:

```python
from env.candor import get_observable_signals, update_ticket_change_step, apply_caught_effect
```

In the existing `from env.state import ...` block, add `CANDOR_LEVEL_SELF_PRESERVATION`:

```python
from env.state import (
    ProjectState,
    TeamMember,
    Task,
    Crisis,
    CANDOR_LEVEL_SELF_PRESERVATION,
    EXEC_SUPPORT_BUDGET_THRESHOLD,
    CLIENT_COMMUNICATION_WINDOW,
    CLIENT_GAIN_PROACTIVE,
    EXEC_DECAY_BUDGET_NO_TIMELINE,
    EXEC_GAIN_RISK_COMM,
    DRIFT_ACK_WINDOW,
)
```

#### 3b. Add constants after the existing constant block

Add these after the line `RESOLVE_BLOCKER_PROGRESS_BOOST = 0.25`:

```python
# Political capital costs for power actions
POLITICAL_CAPITAL_FORCE_TRUTH = 3.0      # PC cost to reveal a member's actual_completion
POLITICAL_CAPITAL_WHISTLEBLOWER = 6.0    # PC cost to reveal identity of worst liar

# Political capital earned from good PM behaviors
PC_EARN_PROACTIVE = 2.0                  # earned from proactive_escalation_with_plan
PC_EARN_TIMELINE = 1.0                   # earned from update_timeline before drift deadline

# Catch window: punitive action must happen within this many steps of cross-verify
CATCH_WINDOW_STEPS = 3
```

#### 3c. Add the `_find_ally` helper

Add this private function after the `_acknowledge_pending_drift` function at the bottom of the file (before or after `check_crisis_resolution`):

```python
def _find_ally(member: TeamMember, state: ProjectState) -> Optional[TeamMember]:
    """
    Return the first other member sharing the same alliance_id, or None.

    Only returns a result if member.alliance_id is not None.
    Used by alibi generation in action_query_member_report.
    """
    if not member.alliance_id:
        return None
    for m in state.team_members:
        if m.member_id != member.member_id and m.alliance_id == member.alliance_id:
            return m
    return None
```

#### 3d. Modify `action_query_observable_signals`

Inside this function, AFTER the lines `state.cross_verify_calls += 1` and `state.total_member_query_calls += 1`, add:

```python
    # Track cross-verification for dynamic candor and catch detection
    member.times_cross_verified += 1
    member.last_cross_verified_step = state.current_step
```

#### 3e. Modify `action_query_member_report`

Inside this function, after the line `state.total_member_query_calls += 1`, and BEFORE the `return ActionResult(...)` call, add the alibi block. Replace the existing return statement with:

```python
    observation = {
        "action_type": "query_member_report",
        "member_id": member.member_id,
        "name": member.name,
        "role": member.role,
        "reported_completion": round(member.reported_completion, 3),
        "reported_availability": round(member.reported_availability, 3),
        "assigned_task_ids": member.assigned_task_ids,
    }

    # Alibi injection: allied self-preservation members blame their stall on their ally.
    # The alibi is CONSISTENT (always blames the same task) so the agent can debunk it
    # by running query_observable_signals on the blamed ally.
    if (member.candor_level == CANDOR_LEVEL_SELF_PRESERVATION
            and member.alliance_id is not None):
        ally = _find_ally(member, state)
        if ally and ally.assigned_task_ids:
            # Always blame the lexicographically first task for consistency across calls
            blamed_task_id = min(ally.assigned_task_ids)
            blamed_task = state.get_task(blamed_task_id)
            if blamed_task:
                observation["alibi"] = {
                    "type": "dependency_block",
                    "claim": (
                        f"I'm blocked — waiting on '{blamed_task.title}' "
                        f"from {ally.name} to land before I can proceed."
                    ),
                    "blames_member_id": ally.member_id,
                    "blames_task_id": blamed_task_id,
                }

    return ActionResult(observation=observation)
```

Note: this replaces whatever the existing return statement in `action_query_member_report` looked like. Keep the `state.total_member_query_calls += 1` line that was already there.

#### 3f. Modify `action_reassign_task`

At the END of the function, just BEFORE the final `return ActionResult(observation={...})` line, add catch detection:

```python
    # Catch detection: if PM is reassigning FROM a recently cross-verified self-preservation member,
    # that counts as catching the liar. Apply dynamic candor effect.
    if old_member_id:
        old_member = state.get_member(old_member_id)
        if (old_member is not None
                and old_member.candor_level == CANDOR_LEVEL_SELF_PRESERVATION
                and (state.current_step - old_member.last_cross_verified_step) <= CATCH_WINDOW_STEPS
                and not old_member.caught_this_episode):
            apply_caught_effect(old_member, state)
            # Merge catch info into the observation dict we're about to return
            catch_note = {
                "deception_catch": {
                    "member_id": old_member_id,
                    "member_name": old_member.name,
                    "effect": "candor_improved_inflation_reduced",
                    "political_capital_awarded": 3.0,
                }
            }
        else:
            catch_note = {}
    else:
        catch_note = {}
```

Then update the return to merge catch_note:

```python
    result_obs = {
        "action_type": "reassign_task",
        "task_id": task_id,
        "from_member_id": old_member_id,
        "to_member_id": to_member_id,
        "task_status": task.status,
    }
    result_obs.update(catch_note)
    return ActionResult(observation=result_obs)
```

#### 3g. Modify `action_communicate`

Inside the `if message_type == "proactive_escalation_with_plan":` block, after the line that updates `client_satisfaction`, add:

```python
        # Award political capital for proactive communication
        state.political_capital = min(20.0, state.political_capital + PC_EARN_PROACTIVE)
```

Also update the return observation dict to include political capital:

In the `return ActionResult(observation={...})` at the end of `action_communicate`, add:

```python
        "political_capital": state.political_capital,
```

#### 3h. Modify `action_update_timeline`

After the line `state.stakeholder.last_budget_request_had_timeline = True`, add:

```python
    # Award political capital for updating timeline (demonstrates planning discipline)
    state.political_capital = min(20.0, state.political_capital + PC_EARN_TIMELINE)
```

Also add `"political_capital": state.political_capital` to the return observation dict.

#### 3i. Modify `action_escalate_risk`

After the line that updates `exec_support`, add catch detection for the escalated crisis:

```python
    # Catch detection: if escalating a crisis that contains tasks assigned to a recently
    # cross-verified self-preservation member, that counts as catching the liar.
    crisis_task_ids = set(crisis.affected_task_ids)
    for m in state.team_members:
        if (m.candor_level == CANDOR_LEVEL_SELF_PRESERVATION
                and (state.current_step - m.last_cross_verified_step) <= CATCH_WINDOW_STEPS
                and not m.caught_this_episode
                and any(tid in crisis_task_ids for tid in m.assigned_task_ids)):
            apply_caught_effect(m, state)
            break  # one catch per escalation action
```

#### 3j. Add three new action functions

Add these three new functions after `action_consult_expert` and before `action_resolve_blocker`:

```python
def action_query_peer_opinion(state: ProjectState, params: Dict[str, Any]) -> ActionResult:
    """
    Ask one team member (asker) for their assessment of another member's progress (subject).

    Cost-1 action. The quality of the opinion depends on the asker's hidden candor:

    - Honest asker (candor >= 0.80): returns subject's actual_completion ± 0.05 noise.
      This is the most reliable signal for detecting liars via triangulation.

    - Allied deceptive asker (same alliance_id as subject): returns subject's
      reported_completion - 0.02, mimicking "honest" slight underestimation while
      actually covering for the ally. The agent cannot tell this from a genuine estimate.

    - Unallied deceptive asker: returns actual_completion ± 0.15 (noisier but unbiased).

    The testimony_type is intentionally NOT in the observation — the agent must infer
    reliability by cross-referencing peer opinions with observable signals.

    Required params: asked_member_id (str), about_member_id (str)
    """
    asked_id = params.get("asked_member_id")
    about_id = params.get("about_member_id")

    if not asked_id:
        return ActionResult(error="Missing param: asked_member_id")
    if not about_id:
        return ActionResult(error="Missing param: about_member_id")
    if asked_id == about_id:
        return ActionResult(error="Cannot query a member about themselves")

    asked = state.get_member(asked_id)
    about = state.get_member(about_id)

    if asked is None:
        return ActionResult(error=f"Unknown asked_member_id: {asked_id!r}")
    if about is None:
        return ActionResult(error=f"Unknown about_member_id: {about_id!r}")

    # Deterministic noise seeded on step + member ids so the same query
    # at the same step returns the same result (idempotent within a step).
    import random as _random
    _rng = _random.Random(state.current_step * 1000 + hash(asked_id) % 997 + hash(about_id) % 997)

    allied = (
        asked.alliance_id is not None
        and asked.alliance_id == about.alliance_id
    )

    if asked.candor >= 0.80:
        # Honest member — gives accurate estimate with small noise
        noise = _rng.uniform(-0.05, 0.05)
        estimate = max(0.0, min(1.0, about.actual_completion + noise))
    elif allied:
        # Allied deceptive — covers for ally with a plausible slight underestimate
        estimate = max(0.0, min(1.0, about.reported_completion - 0.02))
    else:
        # Unallied deceptive — imprecise but not deliberately misleading
        noise = _rng.uniform(-0.15, 0.10)
        estimate = max(0.0, min(1.0, about.actual_completion + noise))

    return ActionResult(observation={
        "action_type": "query_peer_opinion",
        "asked_member_id": asked_id,
        "about_member_id": about_id,
        "peer_estimate_completion": round(estimate, 3),
        # NOTE: testimony reliability is NOT exposed — agent must infer from cross-referencing
    })


def action_force_truth(state: ProjectState, params: Dict[str, Any]) -> ActionResult:
    """
    Spend political capital to compel a member to reveal their actual completion.

    Cost: 1 budget + POLITICAL_CAPITAL_FORCE_TRUTH (3.0) PC.

    If PC is insufficient, returns a failure observation (budget is still spent —
    the PM tried and failed, which is costly). This incentivises the agent to
    build PC before using this action.

    Returns actual_completion and actual_availability of the target member.
    These are the ground-truth values never normally visible to the agent.

    Required params: member_id (str)
    """
    member_id = params.get("member_id")
    if not member_id:
        return ActionResult(error="Missing param: member_id")

    member = state.get_member(member_id)
    if member is None:
        return ActionResult(error=f"Unknown member_id: {member_id!r}")

    if state.political_capital < POLITICAL_CAPITAL_FORCE_TRUTH:
        return ActionResult(observation={
            "action_type": "force_truth",
            "success": False,
            "reason": "insufficient_political_capital",
            "political_capital_remaining": round(state.political_capital, 2),
            "required": POLITICAL_CAPITAL_FORCE_TRUTH,
        })

    state.political_capital -= POLITICAL_CAPITAL_FORCE_TRUTH

    return ActionResult(observation={
        "action_type": "force_truth",
        "success": True,
        "member_id": member_id,
        "actual_completion": round(member.actual_completion, 3),
        "actual_availability": round(member.actual_availability, 3),
        "political_capital_remaining": round(state.political_capital, 2),
    })


def action_trigger_whistleblower(state: ProjectState, params: Dict[str, Any]) -> ActionResult:
    """
    Spend political capital to activate an anonymous tip from an honest team member.

    Cost: 1 budget + POLITICAL_CAPITAL_WHISTLEBLOWER (6.0) PC.

    Reveals the member_id and name of the team member with the lowest current
    candor who has not already been caught this episode. The tip gives the agent
    a high-confidence starting point for cross-verification.

    If PC is insufficient, returns failure (budget still spent).

    No params required.
    """
    if state.political_capital < POLITICAL_CAPITAL_WHISTLEBLOWER:
        return ActionResult(observation={
            "action_type": "trigger_whistleblower",
            "success": False,
            "reason": "insufficient_political_capital",
            "political_capital_remaining": round(state.political_capital, 2),
            "required": POLITICAL_CAPITAL_WHISTLEBLOWER,
        })

    state.political_capital -= POLITICAL_CAPITAL_WHISTLEBLOWER

    # Reveal the uncaught member with lowest candor (most deceptive)
    candidates = [m for m in state.team_members if not m.caught_this_episode]
    if not candidates:
        candidates = list(state.team_members)  # fallback: all caught already

    worst = min(candidates, key=lambda m: m.candor)

    return ActionResult(observation={
        "action_type": "trigger_whistleblower",
        "success": True,
        "revealed_member_id": worst.member_id,
        "revealed_member_name": worst.name,
        "tip": (
            f"Anonymous tip: {worst.name} is significantly misrepresenting their "
            f"progress. Cross-verify with observable signals immediately."
        ),
        "political_capital_remaining": round(state.political_capital, 2),
    })
```

#### 3k. Update `ACTION_COSTS`

Add three new entries to the `ACTION_COSTS` dict:

```python
    "query_peer_opinion":    ACTION_COST_STANDARD,   # cost 1 (+ PC reasoning cost is free)
    "force_truth":           ACTION_COST_STANDARD,   # cost 1 budget + 3.0 PC
    "trigger_whistleblower": ACTION_COST_STANDARD,   # cost 1 budget + 6.0 PC
```

#### 3l. Update `ACTION_HANDLERS`

Add three new entries to the `ACTION_HANDLERS` dict:

```python
    "query_peer_opinion":    action_query_peer_opinion,
    "force_truth":           action_force_truth,
    "trigger_whistleblower": action_trigger_whistleblower,
```

Also update `VALID_ACTION_TYPES` — it is derived from `set(ACTION_COSTS.keys())` so it updates automatically.

---

### STEP 4 — Modify `env/environment.py`

#### 4a. Update `reset()`

After the line `state.consecutive_free_query_count = 0`, add:

```python
        state.political_capital = 5.0  # reset political capital each episode
```

Also reset team member deception-tracking fields when iterating members in reset:

Inside the `for member in state.team_members:` loop in reset(), after `member.ticket_last_changed_step = 0`, add:

```python
            member.times_cross_verified = 0
            member.last_cross_verified_step = -10
            member.caught_this_episode = False
```

#### 4b. Update `_build_observation()`

In the observation dict returned by `_build_observation()`, add `political_capital`:

```python
        return {
            "current_step": s.current_step,
            "budget_remaining": s.budget_remaining,
            "political_capital": round(s.political_capital, 2),   # ADD THIS LINE
            "team_members": members_obs,
            "crises": crises_obs,
            "stakeholder": get_stakeholder_observation(s),
            "done": s.done,
        }
```

#### 4c. Update `state()`

In the `state()` method's return dict, add `political_capital` at the top level:

```python
            "political_capital": round(s.political_capital, 2),
```

And in the `team_members` list comprehension inside `state()`, add to each member dict:

```python
                    "alliance_id": m.alliance_id,
                    "times_cross_verified": m.times_cross_verified,
                    "caught_this_episode": m.caught_this_episode,
```

#### 4d. Update `_compute_reward()`

At the START of `_compute_reward()`, BEFORE the `if self._skip_counterfactual:` guard, add the ignored effect call:

```python
        # Apply ignored effect: self-preservation members never checked grow bolder.
        # This fires at episode end and is logged for curriculum analysis.
        if not self._skip_counterfactual:
            from env.candor import apply_ignored_effect
            for member in agent_state.team_members:
                apply_ignored_effect(member)
```

---

### STEP 5 — Modify `scenarios/level3.py`

For each scenario, add `alliance_id` assignments immediately after the `members = [...]` list is defined. Do NOT change `_make_member` calls — just set `alliance_id` on the member objects after construction.

**scenario_cascading_infra** — after the `members = [...]` block:
```python
    # Deceptive alliance: Vera and Xena cover for each other
    members[0].alliance_id = "alliance_a"   # dev_v1 Vera (VELOCITY_LOW)
    members[2].alliance_id = "alliance_a"   # dev_v3 Xena (VELOCITY_LOW)
    # dev_v5 Zoe is solo deceptive (no ally)
```

**scenario_adversarial_majority** — after the `members = [...]` block:
```python
    # Two alliances among four deceptive members
    members[0].alliance_id = "alliance_a"   # dev_am1 Aaron
    members[1].alliance_id = "alliance_a"   # dev_am2 Bella
    members[2].alliance_id = "alliance_b"   # dev_am3 Carlos
    members[4].alliance_id = "alliance_b"   # dev_am5 Ethan
    # members[3] Diana is honest — no alliance
```

**scenario_cascading_release_failure** — after the `members = [...]` block:
```python
    # Deceptive alliance: Fiona and Hannah cover for each other
    members[0].alliance_id = "alliance_a"   # dev_f1 Fiona (VELOCITY_LOW)
    members[2].alliance_id = "alliance_a"   # dev_f3 Hannah (VELOCITY_LOW)
    # dev_f5 Julia is solo deceptive
```

**scenario_morale_collapse** — after the explicit candor_level overrides:
```python
    # Solo deceptive — no alliance (morale scenario tests signal overlap not social deception)
    # members[0] Marta: alliance_id stays None
```
(No change needed, just document it as a comment.)

**scenario_cascading_crises** — after the explicit candor_level overrides:
```python
    # dev_cc_1 Pavel is solo deceptive (no ally available)
    # No alliance assignments needed
```
(No change needed.)

**scenario_trust_reversal** — after the explicit candor_level overrides:
```python
    # Deceptive alliance: Tariq and Umair cover for each other
    members[1].alliance_id = "alliance_a"   # dev_tr_2 Tariq
    members[2].alliance_id = "alliance_a"   # dev_tr_3 Umair
    # dev_tr_4 Violet is solo deceptive
```

---

### STEP 6 — Modify `scenarios/level4.py`

For the first scenario `scenario_full_disaster`, after the `members = [...]` block (which you'll find with dev_w1 through dev_w5 or w6), add:

```python
    # Level 4: information war — all members deceptive, paired into alliances
    members[0].alliance_id = "alliance_a"   # dev_w1
    members[1].alliance_id = "alliance_a"   # dev_w2
    members[2].alliance_id = "alliance_b"   # dev_w3
    members[3].alliance_id = "alliance_b"   # dev_w4
    # dev_w5 (and dev_w6 if present) get alliance_c or remain None
    if len(members) > 4:
        members[4].alliance_id = "alliance_c"
    if len(members) > 5:
        members[5].alliance_id = "alliance_c"
```

Apply the same pattern to every other scenario in level4.py: pair members[0]+members[1] as "alliance_a", members[2]+members[3] as "alliance_b", members[4]+ as "alliance_c". If a scenario has an odd number, the last member gets no ally.

---

### STEP 7 — Update `training/grpo_trainer.py`

The SYSTEM_PROMPT constant needs to know about the three new actions. Find the SYSTEM_PROMPT string and add the following to the list of available actions within it. The prompt already documents the existing actions — add after the cost-1 actions section:

```
query_peer_opinion   (cost 1):  Ask one member about another's progress. Honest members give 
                                accurate estimates. Allied deceptive members cover for each other.
                                Params: {asked_member_id, about_member_id}

force_truth          (cost 1 + 3 PC):  Compel a member to reveal actual_completion.
                                Only use when you have >= 3 political_capital.
                                Params: {member_id}

trigger_whistleblower (cost 1 + 6 PC): Anonymous tip reveals the most deceptive uncaught member.
                                Only use when you have >= 6 political_capital.
                                Params: {}
```

Also add to the system prompt's resource awareness section:

```
POLITICAL CAPITAL (PC): starts at 5. Earn by: proactive_escalation_with_plan (+2), 
catching a liar (+3), update_timeline (+1). Spend on: force_truth (-3), trigger_whistleblower (-6).
Current PC is shown in every observation under 'political_capital'.
```

---

### STEP 8 — Create `openenv.yaml` at the repo root

Create a new file at `crisisops/openenv.yaml` (or wherever the repo root is, same level as `pyproject.toml`) with this exact content:

```yaml
name: crisisops-v2
version: 2.0.0
description: >
  RL environment for training LLM agents to recover failing software projects
  while detecting and managing adversarially deceptive team members.
  Trains the skill of social triangulation under information warfare.

theme: "3.1"
theme_name: "Professional Tasks — World Modeling"

interface:
  reset: true
  step: true
  state: true

observation_space:
  type: dict
  partial_observability: true
  adversarial_info: true

action_space:
  type: discrete
  n_actions: 16

rubrics:
  - name: crisis_recovery_rate
    description: >
      Fraction of active crises resolved by episode end.
      Computed from actual state, not reported state.
    weight: 0.50
    range: [0.0, 1.0]
    formula: "resolved_crises / total_crises"

  - name: client_satisfaction
    description: >
      Normalized client satisfaction score at episode end (actual, not reported).
      Decays if PM fails to communicate proactively.
    weight: 0.30
    range: [0.0, 1.0]
    formula: "stakeholder.client_satisfaction / 10.0"

  - name: team_morale
    description: >
      Average team morale across all members, normalized to [0,1].
      Affected by reassignments, scope cuts, and passive decay.
    weight: 0.20
    range: [0.0, 1.0]
    formula: "mean(member.morale for member in team) / 10.0"

reward:
  type: counterfactual
  formula: "project_score(agent_final_state) - project_score(greedy_PM_final_state)"
  baseline: greedy_pm
  range: [-1.0, 1.0]
  positive_means: "agent outperformed greedy PM on same initial scenario"

curriculum:
  levels: 4
  unlock_thresholds:
    level_2: 0.15
    level_3: 0.25
    level_4: 0.35
  window_episodes: 10

features:
  partial_observability: true
  adversarial_npcs: true
  dynamic_candor_evolution: true
  alibi_coordination: true
  social_testimony_graph: true
  political_capital_resource: true
  schema_drift: true
  curriculum_learning: true
  counterfactual_reward: true

training:
  framework: trl_grpo
  model: Qwen/Qwen2.5-1.5B-Instruct
  requires_gpu: true
  min_episodes: 200
  recommended_episodes: 500

novel_mechanisms:
  - name: dynamic_candor_evolution
    description: >
      Team member deception intensity changes within episode based on PM behavior.
      Caught liars become more honest. Unchecked liars grow bolder.
      Forces individual-level inference rather than fixed-discount heuristics.
  - name: social_testimony_graph
    description: >
      query_peer_opinion action lets PM ask one member about another.
      Honest members give accurate estimates. Allied deceptive members cover for
      allies. Creates social network reasoning as a trainable skill.
  - name: alibi_coordination
    description: >
      Deceptive members with matching alliance_id provide coordinated alibis,
      blaming stalls on each other. Agent must verify the alibi chain using
      observable signals on the accused member.
  - name: political_capital
    description: >
      Second earned resource (alongside budget). Earned through proactive
      communication and catching liars. Spent on force_truth and
      trigger_whistleblower. Creates dual-resource management problem.
```

---

### STEP 9 — Create `app.py` at the repo root

Create a new file `crisisops/app.py` with this complete content:

```python
"""
app.py — CrisisOps v2 Gradio demo for HuggingFace Spaces.

Lets a user manually play one episode of CrisisOps and see the
counterfactual reward at the end compared to the greedy PM baseline.

Also supports running a random baseline agent for comparison.
"""

import json
import sys
import os

# Ensure the package root is on the path when run from HF Spaces
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr

from env.environment import CrisisOpsEnv
from env.state import ProjectState
from reward.counterfactual import project_score
from scenarios.level1 import get_random_level1_scenario
from scenarios.level2 import get_random_level2_scenario

# ---------------------------------------------------------------------------
# Global episode state (single-user demo)
# ---------------------------------------------------------------------------

_env: CrisisOpsEnv = None
_obs: dict = None
_history: list = []

ACTION_CHOICES = [
    "query_status",
    "query_member_report",
    "query_observable_signals",
    "query_ticket",
    "query_peer_opinion",
    "reassign_task",
    "communicate",
    "cut_scope",
    "escalate_risk",
    "request_resource",
    "update_timeline",
    "consult_expert",
    "force_truth",
    "trigger_whistleblower",
    "resolve_blocker",
    "submit_recovery_plan",
]

PARAM_HINT = {
    "query_status": "{}",
    "query_member_report": '{"member_id": "dev_1"}',
    "query_observable_signals": '{"member_id": "dev_1"}',
    "query_ticket": '{"task_id": "task_1"}',
    "query_peer_opinion": '{"asked_member_id": "dev_1", "about_member_id": "dev_2"}',
    "reassign_task": '{"task_id": "task_1", "to_member_id": "dev_2"}',
    "communicate": '{"message_type": "proactive_escalation_with_plan", "content": "Update", "target": "both"}',
    "cut_scope": '{"task_id": "task_1", "justification": "low priority"}',
    "escalate_risk": '{"crisis_id": "crisis_1", "risk_description": "high severity"}',
    "request_resource": '{"resource_type": "budget", "target_member_id": "dev_1"}',
    "update_timeline": '{"new_completion_date": "2026-05-15", "task_estimates": {}}',
    "consult_expert": "{}",
    "force_truth": '{"member_id": "dev_1"}',
    "trigger_whistleblower": "{}",
    "resolve_blocker": '{"task_id": "task_1", "resolution_notes": "Fixed"}',
    "submit_recovery_plan": '{"plan_summary": "Recovery complete", "risk_items": [], "timeline": "2 weeks"}',
}


def _format_obs(obs: dict) -> str:
    return json.dumps(obs, indent=2)


def _make_env(level: int = 1) -> CrisisOpsEnv:
    if level == 1:
        scenario_fn = get_random_level1_scenario()
    else:
        scenario_fn = get_random_level2_scenario()
    from reward.counterfactual import counterfactual_reward
    return CrisisOpsEnv(
        scenario_fn=scenario_fn,
        reward_fn=counterfactual_reward,
        curriculum_level=level,
    )


def reset_episode(level: int):
    global _env, _obs, _history
    _env = _make_env(int(level))
    _obs = _env.reset(seed=None)
    _history = []
    status = (
        f"Episode started — Level {level} | "
        f"Budget: {_obs.get('budget_remaining', '?')} | "
        f"PC: {_obs.get('political_capital', '?')} | "
        f"Step: 0"
    )
    return _format_obs(_obs), status, "—"


def take_action(action_type: str, params_json: str):
    global _env, _obs, _history

    if _env is None:
        return "Run 'Reset Episode' first.", "Not started", "—"

    try:
        params = json.loads(params_json) if params_json.strip() else {}
    except json.JSONDecodeError as e:
        return _format_obs(_obs), f"Invalid JSON in params: {e}", "—"

    action = {"action_type": action_type, "params": params}
    _history.append(action_type)

    try:
        obs, reward, done, info = _env.step(action)
    except Exception as e:
        return _format_obs(_obs), f"Error: {e}", "—"

    _obs = obs

    if done:
        reward_str = f"{reward:+.3f} vs greedy PM"
        verdict = "✓ Agent beat greedy PM" if reward > 0 else "✗ Greedy PM did better"
        status = (
            f"EPISODE DONE — Counterfactual reward: {reward_str} | {verdict}\n"
            f"Actions used: {len(_history)}"
        )
    else:
        reward_str = "—"
        status = (
            f"Step {obs.get('current_step', '?')} | "
            f"Budget: {obs.get('budget_remaining', '?')} | "
            f"PC: {obs.get('political_capital', '?')} | "
            f"Last: {action_type}"
        )

    return _format_obs(obs), status, reward_str


def update_param_hint(action_type: str):
    return PARAM_HINT.get(action_type, "{}")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="CrisisOps v2 — Deception Detection RL Environment") as demo:

    gr.Markdown("""
# CrisisOps v2

**Train LLMs to recover failing projects while detecting human deception.**

Team members misreport progress to avoid accountability. The PM agent must 
triangulate observable signals (commits, ticket age, peer testimony) against 
self-reports to identify liars and recover the project.

> *"In Kube SRE Gym, the agent reads machine logs — logs don't lie. 
> In CrisisOps, the agent asks engineers — engineers do lie."*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            level_slider = gr.Slider(minimum=1, maximum=2, step=1, value=1, label="Curriculum level")
            reset_btn = gr.Button("Reset Episode", variant="primary")

            gr.Markdown("### Take an action")
            action_dd = gr.Dropdown(
                choices=ACTION_CHOICES,
                value="query_status",
                label="Action type",
            )
            params_box = gr.Textbox(
                value="{}",
                label="Params (JSON)",
                lines=3,
                placeholder='{"member_id": "dev_1"}',
            )
            step_btn = gr.Button("Take Action", variant="secondary")

            gr.Markdown("""
**Quick reference:**
- Free: query_status, query_member_report, query_observable_signals, query_ticket
- Cost 1: reassign_task, communicate, escalate_risk, query_peer_opinion, force_truth, trigger_whistleblower, resolve_blocker
- Terminal: submit_recovery_plan

**Earning PC:** proactive_escalation_with_plan (+2), catching a liar (+3), update_timeline (+1)
            """)

        with gr.Column(scale=2):
            obs_display = gr.Code(language="json", label="Current observation", lines=30)
            status_display = gr.Textbox(label="Status", lines=2)
            reward_display = gr.Textbox(label="Counterfactual reward (shown at episode end)")

    # Wire events
    reset_btn.click(
        reset_episode,
        inputs=[level_slider],
        outputs=[obs_display, status_display, reward_display],
    )
    step_btn.click(
        take_action,
        inputs=[action_dd, params_box],
        outputs=[obs_display, status_display, reward_display],
    )
    action_dd.change(
        update_param_hint,
        inputs=[action_dd],
        outputs=[params_box],
    )

demo.launch()
```

---

### STEP 10 — Verify tests still pass

After all changes, run:

```bash
cd crisisops
python -m pytest tests/ -x -q 2>&1 | tail -20
```

If any test fails because of the new fields (`alliance_id`, `political_capital`, etc.), update the minimal test fixtures to include `alliance_id=None` on TeamMember constructors where needed. Do NOT change the logic of any existing test — only update fixture data.

---

### DO NOT CHANGE

- `reward/counterfactual.py` — the project_score formula and weights are correct as-is
- `reward/baseline.py` — the greedy PM baseline is correct
- `env/schema_drift.py` — drift mechanics are correct
- `env/stakeholders.py` — stakeholder state machines are correct
- `training/grpo_trainer.py` — do NOT touch the training loop logic, dataset construction, or model loading. Only update SYSTEM_PROMPT as specified in Step 7.
- `calibration/calibrate.py` — do not touch
- `pyproject.toml` — do not touch

---

### Summary of all changes

| File | Type | Change |
|---|---|---|
| `env/state.py` | Modify | Add `alliance_id`, `times_cross_verified`, `last_cross_verified_step`, `caught_this_episode` to TeamMember; add `political_capital` to ProjectState |
| `env/candor.py` | Modify | Add `apply_caught_effect`, `apply_ignored_effect`, and 4 constants |
| `env/actions.py` | Modify | Add 3 new actions, 3 new constants, `_find_ally` helper; modify 5 existing actions |
| `env/environment.py` | Modify | Update `reset`, `_build_observation`, `state`, `_compute_reward` |
| `scenarios/level3.py` | Modify | Add `alliance_id` assignments to 4 scenarios |
| `scenarios/level4.py` | Modify | Add `alliance_id` assignments to all scenarios |
| `training/grpo_trainer.py` | Modify | Update SYSTEM_PROMPT to document new actions and PC mechanic |
| `openenv.yaml` | Create | Full manifest with rubrics, features, novel mechanisms |
| `app.py` | Create | Gradio demo for HF Spaces |
