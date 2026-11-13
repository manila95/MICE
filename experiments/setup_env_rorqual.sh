#!/bin/bash
# Build the venv for the MICE/omnisafe gradient-alignment runs on rorqual.
#
# Four Alliance- or version-specific obstacles, all handled below.  Each one is a
# dependency that cannot be satisfied the way requirements.txt asks for it, so each is
# replaced by a module or by a --no-deps install plus a hand-rebuilt dependency set.
#
#  0. PYTHON MUST BE 3.10, NOT 3.11.  safety-gymnasium declares dataclass fields with
#     numpy-array defaults (e.g. `color: np.array = COLOR['apple']` in
#     assets/geoms/apples.py).  Python 3.11 changed dataclasses to reject ANY unhashable
#     default, and np.ndarray.__hash__ is None, so `import safety_gymnasium` raises
#     "mutable default <class 'numpy.ndarray'> ... use default_factory" before any env is
#     built.  3.10 only rejects list/dict/set defaults, so it imports fine.  Note that
#     the Gentoo stack's own /usr/bin/python (3.11.4) SHADOWS a loaded python module on
#     PATH in non-interactive shells, so the interpreter is taken from \$EBROOTPYTHON
#     explicitly rather than from `python`.
#
#  1. pyarrow is dropped entirely.  The wheelhouse's pyarrow is a deliberately-failing
#     dummy wheel ("pip install pyarrow" errors out by design) and the real package ships
#     via the `arrow` module -- but every `arrow` build here depends on python 3.11 and
#     silently RELOADS python/3.10.13 => python/3.11.5 when loaded, which obstacle 0
#     forbids.  requirements.txt lists pyarrow only as pandas' optional string/IO
#     accelerator; no module under omnisafe/ or experiments/ imports it, so it is simply
#     left out and pandas falls back to its numpy object paths.
#
#  2. mujoco: every safety-gymnasium release pins mujoco==2.3.x, which cannot be
#     installed here.  The Alliance wheelhouse carries no `mujoco` wheel at all (only
#     `mujoco_mjx`), and PyPI's manylinux wheels are rejected by this Gentoo stack's
#     platform tags -- `pip download --only-binary :all: mujoco==2.3.3` reports the only
#     candidates as 4.0.0/4.0.1 -- while the 2.3.3 sdist does not build.  The working
#     python bindings here come from
#     `module load mujoco/3.1.6`, which is importable inside the venv via PYTHONPATH.  The
#     mujoco pin is therefore dropped from every dependency set below.  The module
#     resolves its own python3.X/site-packages, so it follows the python module.
#     CAVEAT: 3.1.6 is NOT the 2.3.x safety-gymnasium was written against.  Physics and
#     RNG are not guaranteed to match runs made elsewhere, so numbers from this env are
#     comparable within themselves, not bit-comparable with the original sweeps.
#
#  3. safety-gymnasium hard-pins pygame==2.1.0, which the wheelhouse does not carry (its
#     only pygame is 2.5.2) and which does not build from source here (configure reports
#     no PNG/JPEG headers).  pygame is only used for rendering and these runs are
#     headless, so 2.5.2 goes in instead.
#
#  4. gymnasium must stay at 0.28.1: safety-gymnasium does
#     `from gymnasium.wrappers.compatibility import EnvCompatibility`, which later
#     gymnasium removed.  pettingzoo and gymnasium-robotics both drag gymnasium forward,
#     so they go in with --no-deps and gymnasium is pinned back afterwards.
set -e
BASE=/lustre09/project/6000049/yannpeq
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Load python on its own line and mujoco after it: a combined `module load` lets a
# dependent module's python requirement win, which is how arrow used to drag in 3.11.
module load StdEnv/2023 gcc
module load python/3.10.13
module load mujoco/3.1.6
PY310="${EBROOTPYTHON}/bin/python"
"$PY310" --version | grep -q '^Python 3\.10\.' || { echo "FATAL: $PY310 is not 3.10"; exit 1; }

rm -rf "$BASE/envs/mice"
"$PY310" -m virtualenv --no-download "$BASE/envs/mice"
source "$BASE/envs/mice/bin/activate"
python --version
pip install --no-index --upgrade pip
python -c "import mujoco; print('mujoco from module:', mujoco.__version__)"

# pytorch_lightning is an UNDECLARED dependency: it is absent from requirements.txt, but
# omnisafe/algorithms/__init__.py imports the off_policy package eagerly and
# off_policy/crabs.py does `import pytorch_lightning`, so `import omnisafe` fails without
# it even though these runs only use on-policy CPO.
pip install --no-index torch numpy pandas matplotlib seaborn pyyaml tensorboard \
    wandb typing-extensions scipy pygame pytorch_lightning
echo "=== wheelhouse set installed ==="

# safety-gymnasium: --no-deps, then its own requirements minus the ones we override.
# gymnasium-robotics is dropped here and reinstalled with --no-deps below: it requires
# mujoco~=2.3.0, which would drag the unbuildable mujoco sdist back in through the side
# door even though safety-gymnasium's own mujoco pin has been filtered out.
pip install --no-deps safety-gymnasium
python - <<PY
import importlib.metadata as m, re
drop = ('pygame', 'mujoco', 'gymnasium', 'gymnasium-robotics')
reqs = []
for r in (m.requires('safety-gymnasium') or []):
    if 'extra ==' in r:
        continue
    name = re.split(r'[\[\s<>=!~;]', r, 1)[0].lower()
    if name in drop:
        continue
    reqs.append(r)
print('safety-gymnasium deps (pygame/mujoco/gymnasium/gymnasium-robotics dropped):', reqs)
open('$TMP/sg_deps.txt', 'w').write('\n'.join(reqs) + '\n')
PY
pip install -r "$TMP/sg_deps.txt"
pip install --no-deps 'gymnasium==0.28.1'
pip install jax-jumpy farama-notifications cloudpickle
echo "=== safety-gymnasium installed ==="

# gymnasium-robotics and pettingzoo: --no-deps so they cannot move gymnasium or mujoco.
pip install --no-deps 'gymnasium-robotics==1.2.2' pettingzoo
pip install xmltodict imageio
echo "=== robotics extras installed ==="

grep -vE '^[[:space:]]*#|safety-gymnasium|pyarrow|mujoco|gymnasium' "$BASE/MICE/requirements.txt" \
    | grep -v '^[[:space:]]*$' > "$TMP/mice_rest.txt"
pip install -r "$TMP/mice_rest.txt"
cd "$BASE/MICE"

# NOT `pip install -e .`: this fork has no pyproject.toml and its setup.py calls
# setup(name, version) with no `packages=`, so setuptools falls back to flat-layout
# auto-discovery and refuses the build ("Multiple top-level packages discovered in a
# flat-layout: ['omnisafe', 'experiments', 'analysis_data']").  The repo is used from its
# root on sys.path instead, which is what the job script arranges via PYTHONPATH.  A
# .pth file in the venv makes that automatic for anything using this venv.
echo "$BASE/MICE" > "$BASE/envs/mice/lib/python3.10/site-packages/mice_repo.pth"

# gymnasium is the pin most easily clobbered by anything above; assert it, then step the
# env for real.  A reset alone would not exercise the mujoco 3.1.6 / safety-gymnasium
# mismatch, which is the whole question this smoke test exists to answer.
python - <<'PY'
import sys, gymnasium, mujoco, torch, pygame, safety_gymnasium, omnisafe
assert sys.version_info[:2] == (3, 10), f'interpreter is {sys.version}'
assert gymnasium.__version__ == '0.28.1', f'gymnasium moved to {gymnasium.__version__}'
print('IMPORT OK | python', sys.version.split()[0], '| torch', torch.__version__,
      '| gymnasium', gymnasium.__version__, '| mujoco', mujoco.__version__,
      '| pygame', pygame.version.ver)
env = safety_gymnasium.make('SafetyPointGoal1-v0')
obs, _ = env.reset(seed=0)
tot_r = tot_c = 0.0
for _ in range(100):
    obs, r, c, term, trunc, _ = env.step(env.action_space.sample())
    tot_r += r
    tot_c += c
    if term or trunc:
        obs, _ = env.reset()
print(f'ENV OK | obs {obs.shape} | act {env.action_space.shape} | '
      f'100 steps: reward {tot_r:.3f} cost {tot_c:.1f}')
print('omnisafe from', omnisafe.__file__)
PY
echo "SETUP COMPLETE"
