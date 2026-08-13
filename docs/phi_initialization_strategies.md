# Alternative `phi` initialization strategies for the SR critic

**Status: implemented.** All three strategies below (§3) are live as of this note, wired into
the existing `sr_cfgs.phi_source` switch. `phi_source` now accepts `'rff'` and `'ensemble'` in
addition to `'trunk'` / `'random'` / `'separate'`, and `'random'` / `'separate'` additionally
accept a new `phi_orthogonal_init` flag. §4 lists what's still hyperparameter-and-config
complete: every new knob is documented in §5 and threaded through every algorithm's YAML config.

This note evaluates the original three `phi` construction options in the `td_ridge` successor-
representation critic (`omnisafe/models/critic/successor_representation_critic.py`) and adds more
principled alternatives. Scope: **structured random features** — ways to build a frozen,
stationary `phi` that are still "random" (no task-specific training) but are constructed so their
approximation properties are actually interpretable. Other directions (adaptive-but-stable phi,
spectral/successor-theory representations, task-supervised phi) are noted briefly at the end for
future work — not implemented here.

## 1. Why this matters here specifically

`phi` in this codebase isn't a generic feature map — it has two hard requirements imposed by how
it's consumed:

- **Stationarity, if frozen.** `psi` is *defined* as the discounted sum of the `phi` stream
  (`psi_t ≈ phi_t + γ·psi_{t+1}`), and the ridge solve fits `w_r`/`w_c` against `phi` as the design
  matrix. If `phi` moves, both the TD target and the ridge basis are shifting under the estimator
  that's fitting them (`FrozenPhiFeatures` docstring,
  [successor_representation_critic.py:330-334](../omnisafe/models/critic/successor_representation_critic.py#L330-L334)).
  This is *why* `'random'`/`'separate'` freeze every parameter
  ([:361](../omnisafe/models/critic/successor_representation_critic.py#L361)) and exclude
  themselves from the SR optimizer — and it's the same reason the two new frozen families below
  (`'rff'`, and every member of `'ensemble'`) freeze every parameter too.
- **Two regression targets, not one.** `phi` has to be a good basis for ridge-regressing *both*
  reward and cost from the same design matrix, and empirically these don't fit equally well at
  the same shrinkage — commit `a1f13b5` split `ridge_kappa_cost` out from `ridge_kappa`
  specifically because cost's train/val `RidgeR2` gap (1.19) is far worse than reward's (0.05) on
  `SafetyPointGoal1-v0`
  ([ridge_update docstring, :272-277](../omnisafe/models/critic/successor_representation_critic.py#L272-L277)).
  So "is `phi` a good basis" is really "is `phi` a good basis for a dense reward *and* a sparse
  cost simultaneously" — a stronger requirement than generic representation quality.
- **Rank is capped, not just tunable.** An affine map (`'random'`, `hidden_sizes=[]`) cannot raise
  rank beyond `obs_dim + 1` no matter how large `sr_dim` is — measured 18 of 64 effective rank on
  a 17-dim observation
  ([module docstring, :57](../omnisafe/models/critic/successor_representation_critic.py#L57)).
  This is a structural ceiling, not something more random draws or a bigger `sr_dim` fixes — a
  spot-check of the shipped implementation reproduces it almost exactly (17-dim input, `sr_dim=64`:
  effective rank ≈16.5 plain-random, ≈17.2 orthogonal — both pinned near `obs_dim + 1 = 18`; see
  §3.1, ORF narrows the gap to that ceiling but does not break it).

## 2. Baseline critique

| `phi_source` | Stationary? | Rank | What kernel/function class is this? |
|---|---|---|---|
| `'trunk'` | No — drifts with every TD/value update even though nothing trains `phi_head` directly | Whatever the shared trunk happens to have | Undefined — a moving target, not a fixed function class |
| `'random'` | Yes | Capped at `obs_dim + 1` (affine map) | A random linear map — well understood, but provably rank-limited |
| `'separate'` | Yes | Full rank achievable (nonlinear MLP) | **Undefined.** An untrained random MLP's implicit kernel is not a standard, characterizable object — you can add width/depth but can't reason about *what* it approximates or tune its smoothness in a principled way |

`'random'` is honest about its limitation (it's explicitly documented as a rank-capped lower
bound). `'separate'` escapes the rank cap but trades it for architectural arbitrariness — there is
no articulable notion of what `'separate'` is a random draw *from*, which is the actual gap the
"structured random features" alternatives below close.

## 3. Structured random features

All three plug into the existing `build_frozen_phi` dispatcher
([successor_representation_critic.py:433-527](../omnisafe/models/critic/successor_representation_critic.py#L433-L527)),
so they extend `sr_cfgs.phi_source` without touching anything downstream — the ridge solve, TD
update, and diagnostics are all written against "some frozen `phi` module" and don't care which
one it is. `build_frozen_phi` is threaded through both `TDRidgeSuccessorRepresentationTrunk` (V
flavor) and `TDRidgeSuccessorRepresentationQTrunk` (Q flavor), and from there through
`ConstraintActorCritic`/`ConstraintActorQCritic`'s model-building code, which reads every new knob
off `model_cfgs.sr_cfgs` with the same `.get(key, default)` pattern the existing fields use.

### 3.1 Orthogonal Random Features (ORF) — cheapest, no new `phi_source`

**Construction.** Identical architecture to `'random'`/`'separate'` — same linear (or MLP)
layers — but every layer's weight matrix is drawn orthogonal (via `torch.nn.init.orthogonal_`,
QR of a Gaussian matrix) instead of i.i.d. Gaussian. Implemented by reusing the codebase's
existing `'orthogonal'` `weight_initialization_mode` (already a case in
`initialize_layer`, `omnisafe/utils/model.py:41-42`) rather than adding new init code: when
`phi_orthogonal_init: true`, `build_frozen_phi` simply builds the frozen net with
`weight_initialization_mode='orthogonal'` in place of whatever `model_cfgs.weight_initialization_mode`
would otherwise be
([:474-495](../omnisafe/models/critic/successor_representation_critic.py#L474-L495)).

**Why it helps.** Orthogonalizing the rows of a random projection matrix is a well-established
variance-reduction result relative to i.i.d. Gaussian rows, for the same number of rows. Because
`phi`'s output is always L2-normalized, only the projection's *directions* matter, not its raw
scale — so this is a pure win with no other side effect. In this codebase's terms it means better
*effective* rank utilization of a fixed `sr_dim` for the same `obs_dim`, narrowing (not
eliminating) the "18 of 64" rank gap `'random'` reports; see the measured numbers in §1.

**Hyperparameters.**

| Name | Type | Default | Applies to | Notes |
|---|---|---|---|---|
| `sr_cfgs.phi_orthogonal_init` | `bool` | `false` | `phi_source: 'random'` / `'separate'`, and any such entries inside `phi_source: 'ensemble'` | No other new knob — orthogonal init has no free parameter beyond the switch itself. |

### 3.2 Random Fourier Features (RFF)

**Construction.** `phi(x) = normalize(cos(Wx + b))`, with rows of `W` drawn `~ N(0, 1/bandwidth²)`
and `b ~ Uniform(0, 2π)`, both frozen at init. Implemented as a new class,
`FrozenRFFPhiFeatures`
([:369-406](../omnisafe/models/critic/successor_representation_critic.py#L369-L406)), and a new
`phi_source: 'rff'` branch in `build_frozen_phi`
([:496-497](../omnisafe/models/critic/successor_representation_critic.py#L496-L497)).

**Why it helps.** RFF is the classical Random Kitchen Sinks / Rahimi & Recht construction: it's a
random basis, but by design its inner products approximate a *named, well-understood* kernel
(Gaussian/RBF), so unlike `'separate'`'s untrained MLP you can reason about what function class it
approximates and tune its smoothness directly via the bandwidth. It also escapes `'random'`'s
affine rank cap (the `cos(·)` nonlinearity is what buys full rank) while staying architecturally
shallow — cheaper than `'separate'`'s multi-layer MLP for a comparable capacity gain.

**Hyperparameters.**

| Name | Type | Default | Applies to | Notes |
|---|---|---|---|---|
| `sr_cfgs.phi_rff_bandwidth` | `float > 0` | `1.0` | `phi_source: 'rff'`, and any such entries inside `phi_source: 'ensemble'` | RBF kernel bandwidth σ. Larger → smoother/lower-frequency features (rows of `W` scaled down by `1/σ`); smaller → higher-frequency, more locally-varying features. Fixed and hand-set for now, not data-adaptive — a natural follow-up would scale it off the observation norm of the first batch the way `ridge_kappa` is scaled off the Gram-matrix diagonal in `ridge_update` ([:277-283](../omnisafe/models/critic/successor_representation_critic.py#L277-L283)), but that's out of scope here. |

### 3.3 Ensemble of independent random phis

**Construction.** `k` independent frozen sub-bases (each one of `'random'` / `'separate'` /
`'rff'`) are built and their outputs concatenated, splitting `sr_dim` as evenly as possible across
members (`divmod(sr_dim, k)`, with the remainder distributed to the earliest members so widths
always sum to exactly `sr_dim`), then the concatenation is re-normalized as a whole. Implemented as
`EnsemblePhiFeatures`
([:407-431](../omnisafe/models/critic/successor_representation_critic.py#L407-L431)), built by a
`phi_source: 'ensemble'` branch in `build_frozen_phi` that **recurses into itself** once per
requested sub-basis
([:498-524](../omnisafe/models/critic/successor_representation_critic.py#L498-L524)) — so ORF and
the RFF bandwidth apply to any `'random'`/`'separate'`/`'rff'` entries nested inside an ensemble
too.

**Why it helps.** A single random draw is one sample from a distribution over bases; concatenating
several turns that into an average-case-robust basis instead of betting everything on one draw.
It's also a useful *diagnostic* device even when not used in production: the existing SR
diagnostics module (`omnisafe/utils/sr_diagnostics.py` — effective rank, stable rank, dead-dim
fraction) can be pointed at one sub-slice of `phi` at a time to see which family actually carries
the reward/cost signal — direct evidence for choosing between §3.1/§3.2 rather than guessing.

**Hyperparameters.**

| Name | Type | Default | Applies to | Notes |
|---|---|---|---|---|
| `sr_cfgs.phi_ensemble_sources` | `list[str] \| null` | `null` → `['random', 'separate', 'rff']` | `phi_source: 'ensemble'` only | Each entry one of `'random'` / `'separate'` / `'rff'`; `'trunk'` and nested `'ensemble'` are rejected with an assertion. `phi_orthogonal_init` and `phi_rff_bandwidth` above apply uniformly to every matching entry. |

## 4. Shared / pre-existing knobs these reuse

Not new, but worth naming since every strategy above depends on them:

| Name | Applies to | Notes |
|---|---|---|
| `sr_cfgs.phi_hidden_sizes` | `phi_source: 'separate'`, and `'separate'` entries of `'ensemble'` | Depth of the standalone MLP; every layer's width is forced to `sr_dim` (or, inside an ensemble, that member's split of `sr_dim`) at construction time, same convention as `sr_cfgs.hidden_sizes` for the trunk. |
| `sr_cfgs.activation` | All frozen phi families | Nonlinearity used inside `'separate'`'s MLP layers; unused by `'random'` (no hidden layers) and by `'rff'` (its nonlinearity is `cos`, not the configured activation). |

## 5. Where the new config fields live

`phi_orthogonal_init`, `phi_rff_bandwidth`, and `phi_ensemble_sources` were added to the
`sr_cfgs:` block of every algorithm config that already had a `phi_source` field — all 25
on-policy YAMLs under `omnisafe/configs/on-policy/` plus `omnisafe/configs/off-policy/SACPID.yaml`
— immediately after the existing `phi_hidden_sizes` field, with inline comments matching this
doc's descriptions. Model construction reads them via `sr_cfgs.get(key, default)` in
`omnisafe/models/actor_critic/constraint_actor_critic.py` and
`omnisafe/models/actor_critic/constraint_actor_q_critic.py`, so existing configs that predate
these fields still load unchanged (all three default to `'random'`/`'separate'`-preserving,
i.e. off, behavior).

## 6. Other directions (not implemented — for future work)

- **Stability-improved adaptive phi**: an EMA/Polyak target-network `phi` (slow-moving copy of the
  trunk-derived `phi`, analogous to a DQN target network), or a pretrain-then-freeze burn-in —
  middle ground between `'trunk'`'s adaptivity and full freezing's stationarity.
- **SR-theory-grounded representations**: Laplacian eigenfunctions / proto-value functions
  computed from the replay-buffer transition graph; Forward-Backward (FB) representations — note a
  sibling branch, `sr_fb_critic`, already has FB work started (`git log` shows a commit "adding fb
  representation" there) and should be reviewed before any reimplementation on this branch.
- **Task-supervised phi pretraining**: explicitly fit `phi` itself (not just `w`) to jointly
  predict reward and cost during a burn-in phase, reusing
  `RidgeSolvedReadoutWeights.regression_loss`
  ([:199-235](../omnisafe/models/critic/successor_representation_critic.py#L199-L235)) with
  gradients flowing into `phi` instead of detaching it, then freezing once burn-in ends.

## 7. Suggested next step

With all three implemented, the natural next step is an empirical sweep rather than more
implementation: run `phi_source` ∈ `{random, random+orthogonal, separate, separate+orthogonal,
rff, ensemble}` on the same task and compare `RidgeR2`/effective-rank/dead-dim diagnostics already
logged by `sr_diagnostics.py`, to check whether ORF's rank-utilization gain and RFF's/ensemble's
extra complexity actually move the reward/cost fit quality, rather than reasoning about it further
from first principles.
