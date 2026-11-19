# Stable off-policy safe RL on SafetyPointGoal1-v0

Configurations that train reward **and** satisfy the cost constraint, found 2026-11-19.

The headline is negative and worth stating first: **the working configuration is OmniSafe's
stock `SafetyPointGoal1-v0` block with nothing changed but `cost_limit`.** Four attempted
improvements — an upper bound on the multiplier, a 20x faster `lambda_lr`, a 4x faster
`lambda_lr`, and a longer warmup — all performed worse.

## The configs

| file | `cost_limit` | EpRet | EpCost | λ |
| --- | --- | --- | --- | --- |
| `saclag_pointgoal1_cl10.yaml` | 10 | 12.0 | 5.4 | 0.55 |
| `saclag_pointgoal1_cl25.yaml` | 25 | 11.6 | 13.7 | 0.45 |

Means over the last 50 of 1000 epochs (2M steps). Both satisfy their constraint.
OmniSafe's published reference at `cost_limit=25` and 3M steps is reward 21.45, cost 19.17.

Reproduce with `launch.py`, or equivalently:

```python
omnisafe.Agent('SACLag', 'SafetyPointGoal1-v0', seed, custom_cfgs={
    'train_cfgs': {'total_steps': 2_000_000},
    'lagrange_cfgs': {'cost_limit': 10.0},
}).learn()
```

Everything else comes from the env block and must not be overridden: `alpha 1e-5`,
`actor lr 5e-6`, `critic lr 1e-3`, `lambda_lr 5e-7`, `warmup_epochs 100`,
`cost_normalize False`. These are interdependent — see below.

## Why the defaults are not negotiable

The PointGoal1 block looks badly tuned (an actor at `lr 5e-6`, entropy effectively off at
`alpha 1e-5`, a multiplier at `lambda_lr 5e-7`) and is not. It is a slow-dual schedule:
λ is frozen for 100 epochs, then rises at a measured **0.00100 per epoch**, so the policy
learns to navigate before any constraint pressure arrives. Overriding these to "sensible"
base values and setting `warmup_epochs: 0` produces a policy that never learns at all —
an 18-arm screen under those overrides reached EpRet between −0.58 and 0.56, against the
17–25 the published benchmark attains.

Note also that `update_cycle: 1` means λ is updated once per environment **step**, 2000
times per epoch, against a `Jc` that refreshes once per epoch. That is why `lambda_lr`
must be so small; it is not a typo.

## What did not work

| intervention | EpRet | EpCost | why it failed |
| --- | --- | --- | --- |
| `lagrangian_upper_bound: 5.0` | — | — | null: λ self-regulates and peaks near 1.8, never reaching 5 |
| `lambda_lr: 1e-5` (20x) | 7.2 | 26.8 | λ engages at epoch ~150 and crushes a half-learned policy |
| `lambda_lr: 2e-6` (4x) | 8.1 | 18.6 | same failure, milder |
| `warmup_epochs: 250` + 20x | 11.1 | 18.8 | delaying helps, but the fast ramp still overshoots |
| TD3Lag, `cost_limit 10` | 5.3 | 10.1 | satisfies the limit at half the reward of SACLag |

The upper bound added to `SACLag/TD3Lag/DDPGLag` (branch `offpolicy-stable`, default
`null`, so a no-op) is still worth keeping for the **PID** variants, where an integral term
accumulates with nothing to stop it — SACPID reached λ=629 on this task. Plain Lagrangian
does not need it.

## Caveat: these are not converged

Reward peaks near 27 at epoch ~210 (cost ~46, limit violated) and then declines
monotonically as λ rises — 12.0 at epoch 1000 and still falling. The saved numbers are a
point on a descending curve, not an equilibrium. Running longer will buy lower cost at
further reward. `figs/offpolicy_stable_curves.{png,pdf}` plots this; regenerate with
`plot_curves.py`.
