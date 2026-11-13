#!/bin/bash
# Gradient-direction study: the Figure 1 CPO arm, re-run with the diagnostics of
# PolicyGradient._gradient_direction_study enabled.
#
# Configuration is the Figure 1 CPO cell of sweep hyfcu0dx exactly -- plain advantage on
# both streams, update_iters 2, critic lr 3e-4, SafetyPointGoal1-v0, cost limit 10,
# 512 epochs of 20k steps, 5 seeds -- so the gradient numbers can be read against the
# value-correlation numbers already reported for that cell, rather than against a
# differently-tuned run.
#
# What is new relative to that sweep:
#   * algo_cfgs.gradient_direction_study: on every eval epoch, build the cost and reward
#     policy gradients from TD(0), TD(lam) and TD(1) advantages over that epoch's own
#     ~2e4-sample batch and record the Euclidean and Fisher cosines between them, into
#     eval_data/epoch_NNNNN.pkl under the 'gradient_direction' key.  Costs no environment
#     steps.
#   * algo_cfgs.probe_gradient_study: on every eval epoch, draw probe_per_episode step
#     indices uniformly per episode out of THIS batch's own rollout, capture (s, a, s')
#     there, promote s' to a probe too, and estimate V at both ends by MC. Gives the cost
#     and reward policy gradients from the critic and from the MC estimates over the same
#     transitions -- a literal subsample of the update CPO just took (verified: the probe
#     TD(0) advantage matches the buffer's own adv_c exactly). Reported against two ceilings,
#     the 102-vs-2e4 subsample cosine and the split-half MC cosine.
#   * legacy_value_studies off: MCStudy/*, IntermediateMC/pos*/* and PooledMC/* scored states
#     from dedicated rollouts; the probe study scores batch states instead. Same eval cadence
#     as the original Figure 1 sweep (5 to epoch 50, then 50 -> 20 eval epochs).
#   * train_batch_diagnostics off: Value/Train/{Before,After}Update/* dropped.
#
# Timing: plain epoch 15.8 s (measured). probe_per_episode 25 gives 500 probes + 500 next
# states, i.e. 4.59e6 eval steps per eval epoch; at the measured 2966 eval steps/s that is
# ~26 min each, so 20 eval epochs ~= 8.6 h on top of ~2.2 h of training -> ~10.8 h per seed.
# The 24 h limit is headroom for node contention, measured to cost up to 3.5x.
#
# 25 rather than 5 per episode because the split-half ceiling at 5 (100 probes) came out at
# only 0.21 on the cost stream -- the MC cost gradient was close to pure noise, leaving no
# room to resolve a critic-vs-MC cosine against it.
#
# NOTE: 'plain' means the advantage is TD(0), so for this cell the 'c_td0' gradient IS the
# update the algorithm takes; 'c_tdlam' and 'c_td1' are counterfactuals on the same batch.
#
# NOTE: there is no `mc_value_study` or `intermediate_state_study` boolean -- both studies
# are gated on the single `eval_critic` flag (CPO.yaml) and sized by their *_probes /
# *_repeats keys.  Passing the non-existent booleans makes omnisafe's
# recursive_check_config raise KeyError before training starts.
#
# Submit with:  sbatch experiments/sbatch_grad_alignment.sh
#SBATCH --job-name=grad-align
#SBATCH --account=def-chgag196_cpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=1-5
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs

# The mujoco module supplies the mujoco python bindings (the wheelhouse carries no mujoco
# wheel at all) and must be loaded before the venv is activated, at run time as well as at
# build time.  python must be 3.10: safety-gymnasium's numpy-array dataclass defaults do
# not import under 3.11+.  Load python on its own line and mujoco after it -- a combined
# `module load` lets a dependent module's python requirement win.  Do NOT add the arrow
# module here: it forces python 3.11.  See setup_env_rorqual.sh's header for the details.
module load StdEnv/2023 gcc
module load python/3.10.13
module load mujoco/3.1.6
source /lustre09/project/6000049/yannpeq/envs/mice/bin/activate

# The repo is not pip-installed (no pyproject.toml, and setup.py declares no packages),
# so it has to be on sys.path explicitly; the venv's mice_repo.pth normally does this,
# and this makes it true even if the venv is rebuilt without it.
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Fail in seconds rather than after a queue wait if the env is not the one that was built.
python - <<'PYCHK'
import sys, mujoco, gymnasium, safety_gymnasium, omnisafe
assert sys.version_info[:2] == (3, 10), f'interpreter is {sys.version}'
assert gymnasium.__version__ == '0.28.1', f'gymnasium is {gymnasium.__version__}'
print(f'env OK | python {sys.version.split()[0]} | mujoco {mujoco.__version__} '
      f'| gymnasium {gymnasium.__version__} | omnisafe {omnisafe.__file__}', flush=True)
PYCHK

# Compute nodes have no outbound network on this cluster, so wandb must not try to reach
# its API: offline mode writes the run (artifacts included) to disk for a later
# `wandb sync` from a login node.  Without this the run blocks on the first wandb call.
# The per-epoch eval pickles are written to log_dir regardless of wandb, so the analysis
# path does not depend on the sync happening at all.
export WANDB_MODE=offline
export WANDB_DIR=/lustre10/scratch/yannpeq/wandb
mkdir -p "${WANDB_DIR}"

# torch's intra-op pool spins; with vector_env_nums rollout workers plus the eval study's
# own vectorized envs already competing for this allocation's cores, more than one thread
# per process is pure liability for a 2x64 MLP.  See the long perf note in
# launch_value_function_study.sh, which measured a 25-35x slowdown from this alone.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEED="${SLURM_ARRAY_TASK_ID}"
NAME="CPO_seed${SEED}"

python experiments/train_mice.py \
  --algo CPO \
  --env-id SafetyPointGoal1-v0 \
  --cost-limit 10 \
  --seed "${SEED}" \
  --device cpu \
  --torch-threads 1 \
  --train_cfgs:total_steps 10240000 \
  --train_cfgs:vector_env_nums 5 \
  --algo_cfgs:steps_per_epoch 20000 \
  --algo_cfgs:update_iters 2 \
  --algo_cfgs:adv_estimation_method plain \
  --algo_cfgs:cost_adv_estimation_method plain \
  --model_cfgs:critic:lr 0.0003 \
  --algo_cfgs:gradient_direction_study True \
  --algo_cfgs:probe_gradient_study True \
  --algo_cfgs:probe_per_episode 25 \
  --algo_cfgs:legacy_value_studies False \
  --algo_cfgs:train_batch_diagnostics False \
  --algo_cfgs:eval_critic True \
  --algo_cfgs:mc_value_study_probes 17 \
  --algo_cfgs:mc_value_study_repeats 10 \
  --algo_cfgs:mc_value_study_vector_envs 5 \
  --algo_cfgs:intermediate_state_study_probes 17 \
  --algo_cfgs:intermediate_state_study_repeats 10 \
  --logger_cfgs:log_dir "/lustre10/scratch/yannpeq/runs_grad_alignment/${NAME}"
