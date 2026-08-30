#!/bin/bash
# Launches the 2 algos (CPO, TRPOPID) x 5 seeds = 10 runs for the value-function
# estimation study (SafetyPointGoal1-v0, cost-limit=10, 10M steps, default config for
# everything else except value_eval_freq=25 for the post-epoch-100 cadence --
# early_eval_freq is already 5 by default, matching "every 5 epochs up to epoch 100").
# Training itself is vectorized (default vector-env-nums=5).
#
# v2: on top of the original s0-only study, this run also turns on:
#   - intermediate_state_study: the on-policy (not just reset-state) value-estimation study,
#     scoring V(s_t) at 5 within-episode positions using genuine on-policy states (see
#     omnisafe/utils/state_snapshot.py).
#   - per-eval-epoch data dumps: eval_data/epoch_NNNNN.pkl (raw per-probe arrays + aggregate
#     stats for every study that ran that epoch) and epoch_NNNNN_scatter.png (predicted vs.
#     MC-true scatter grid), plus a model checkpoint -- all on the same eval cadence (see
#     omnisafe/utils/eval_data_dump.py). Old runs_value_study/ (the first, s0-only pass) is
#     left untouched; this writes to runs_value_study_v2/.
#
# TWO SEQUENTIAL BATCHES OF 5, not all 10 at once: each run spawns 1 main + 5 training +
# 5 (s0 study) + 17 (intermediate-state study) = 28 subprocess workers, and each worker
# carries a real ~510MB baseline (torch+numpy+mujoco+safety_gymnasium loaded into every
# process, confirmed empirically -- ~373MB from the imports alone, before any env/model
# exists). 10 runs concurrently = 280 processes =~143GB -- overcommits this box's 123GB RAM
# and thrashes on swap (measured: 17GB swapped, epoch 0 still incomplete after 19 minutes vs.
# ~125s in isolated testing). 5 concurrently =~71GB, safely fits. Costs ~2x wall-clock (two
# sequential batches) to keep the full agreed probe budget (17 states per s0/each of the 5
# intermediate positions, 100 total) rather than reducing it.
#
# Perf notes (this is what took the *original* s0-only run of this script ~15-19h wall-clock
# before the fixes below):
#
# 1. torch_threads=1 (not the yaml default of 16, sized for one lone run, and not even the 2
#    tried first): with 5 concurrent runs x (5 vector_env_nums rollout workers + 1 main
#    process wanting torch_threads threads) =~35 threads competing for this box's 16 physical
#    (32 SMT) cores, even at torch_threads=2 "plain" epochs ran 25-35x slower than in
#    isolation (measured: ~30s vs ~1.1s), with 33.7% system time and 22k-58k context
#    switches/sec -- the signature of torch's *spinlock-based* intra-op thread pool under
#    contention (great with no contention, pathologically non-linear once other processes
#    genuinely need that CPU too). The actual network here is tiny (2x64 MLP); a second
#    thread buys nothing uncontended and is pure liability once contended. Plus
#    OMP/MKL_NUM_THREADS=1 so numpy/BLAS in the env subprocess workers don't repeat the same
#    mistake.
# 2. mc_value_study_vector_envs=5: the MC study used to run its probes x repeats episodes one
#    at a time on a single env. Now vectorized -- N probes run concurrently instead of
#    sequentially.
# 3. Total probe budget: 100 states total across ALL 6 categories (s0 + the 5 intermediate
#    positions), evenly split -- not 100 for each. mc_value_study_probes=17 and
#    intermediate_state_study_probes=17 (17 x 6 = 102, close enough to 100 for an even split).
#
# CAUTION -- torch-threads must be passed as `--torch-threads N`, NOT `--train_cfgs:torch_threads
# N`: experiments/train_mice.py's own argparse already registers `--torch-threads` (default 4,
# unrelated to the omnisafe-level train_cfgs config) and passes its whole `vars(args)` as
# `train_terminal_cfgs`, which AlgoWrapper._init_config() applies AFTER custom_cfgs -- so a
# `--train_cfgs:torch_threads` custom-cfgs override silently loses to that argparse default of 4.
# Found by noticing the resolved config actually printed "torch_threads": 4 despite this script
# passing `--train_cfgs:torch_threads 1`, right as batch 1 (CPO) finished -- meaning batch 1's
# entire ~7h run happened at torch_threads=4, not the intended 1, which is the real reason the
# interop-thread fix alone didn't close the isolated-vs-contended gap the earlier investigation
# expected it to. Batch 1's *data* is unaffected (torch_threads only changes wall-clock speed,
# not results), so it wasn't rerun; batch 2 (TRPOPID) picked up the corrected flag below.
set -e
cd /home/kaustubh/projects/calibration_rl/MICE
mkdir -p runs_value_study_v2/logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

launch_one() {
  local algo="$1" seed="$2"
  local name="${algo}_seed${seed}"
  local logfile="runs_value_study_v2/logs/${name}.log"
  nohup python experiments/train_mice.py \
    --algo "${algo}" \
    --env-id SafetyPointGoal1-v0 \
    --cost-limit 10 \
    --seed "${seed}" \
    --device cpu \
    --torch-threads 1 \
    --algo_cfgs:mc_value_study True \
    --algo_cfgs:mc_value_study_probes 17 \
    --algo_cfgs:value_eval_freq 25 \
    --algo_cfgs:mc_value_study_vector_envs 5 \
    --algo_cfgs:intermediate_state_study True \
    --algo_cfgs:intermediate_state_study_probes 17 \
    --logger_cfgs:log_dir "runs_value_study_v2/${name}" \
    > "${logfile}" 2>&1 &
  echo "launched ${name} (pid $!) -> ${logfile}"
  sleep 2
}

echo "=== batch 1: CPO seeds 1-5 ==="
for seed in 1 2 3 4 5; do
  launch_one CPO "${seed}"
done
wait
echo "=== batch 1 finished, starting batch 2: TRPOPID seeds 1-5 ==="

for seed in 1 2 3 4 5; do
  launch_one TRPOPID "${seed}"
done
wait
echo "=== batch 2 finished, all 10 runs complete ==="
