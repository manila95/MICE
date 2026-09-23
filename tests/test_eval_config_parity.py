"""Every on-policy config must carry CPO's evaluation block.

Run either way -- ``pytest tests/test_eval_config_parity.py`` or
``python tests/test_eval_config_parity.py``.

``PolicyGradient._run_eval_studies`` (inherited by every on-policy algorithm, and called
explicitly by MICE's own ``learn()``) reads every one of its gates through
``getattr(self._cfgs.algo_cfgs, <key>, <default>)``, and ``_init_log`` registers the
corresponding logger keys behind the *same* gates. A config that simply omits a key therefore
silently skips that study rather than failing -- which is how OnCRPO (and 23 others) ended up
running only the superseded ``test_estimate`` path while CPO/TRPOPID ran the MC studies.

Worse, the omission cannot be worked around at the call site: ``custom_cfgs`` validates against
the keys the yaml already declares, so ``mc_value_study=True`` on an algorithm whose yaml lacks
the key raises ``KeyError: 'Invalid key: mc_value_study'``. Declaring the block in the yaml is
the only way to make the study reachable at all.

This test pins that parity so a newly-added algorithm config (or a hand-edit) can't drift back.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml


CONFIG_DIR = Path(__file__).parent.parent / 'omnisafe' / 'configs' / 'on-policy'
REFERENCE = 'CPO'

#: Keys that make up the shared evaluation block. Value parity with CPO is required for all of
#: them except the two study switches, which are allowed to be off for the algorithms below.
EVAL_KEYS = (
    'test_estimate',
    'value_eval_episodes',
    'early_eval_freq',
    'early_eval_epochs',
    'eval_rng_seed_base',
    'async_eval',
    'async_eval_spawn_worker',
    'value_eval_freq',
    'mc_value_study',
    'mc_value_study_probes',
    'mc_value_study_repeats',
    'mc_value_study_seed_offset',
    'mc_value_study_vector_envs',
    'mc_eval_bootstrap_threshold',
    'mc_eval_tail',
    'intermediate_state_study',
    'intermediate_state_study_positions',
    'intermediate_state_study_probes',
    'intermediate_state_study_repeats',
)

STUDY_SWITCHES = ('mc_value_study', 'intermediate_state_study')

#: Algorithms whose adapter augments the observation space (SauteAdapter / SimmerAdapter append a
#: safety state), while ``_get_mc_value_study_env`` / ``_get_intermediate_state_env`` rebuild the
#: env with the base ``OnlineAdapter`` wrapper recipe only. Probe observations come back one
#: dimension short of what the critic expects, so the studies are wired in but left off. Forcing
#: them on raises ``RuntimeError: mat1 and mat2 shapes cannot be multiplied (Nx60 and 61x64)``.
#: Drop an entry here once those builders reproduce the algorithm-specific augmentation.
OBS_AUGMENTING = frozenset({'PPOSaute', 'TRPOSaute', 'PPOSimmerPID', 'TRPOSimmerPID'})


def _algo_cfgs(name: str) -> dict:
    with open(CONFIG_DIR / f'{name}.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)['defaults']['algo_cfgs']


def _all_algos() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob('*.yaml'))


def test_reference_config_declares_every_eval_key() -> None:
    """The reference itself must be complete, or the parity check below is vacuous."""
    cfgs = _algo_cfgs(REFERENCE)
    missing = [k for k in EVAL_KEYS if k not in cfgs]
    assert not missing, f'{REFERENCE}.yaml is missing {missing}'
    for switch in STUDY_SWITCHES:
        assert cfgs[switch] is True, f'{REFERENCE}.yaml has {switch} off; it is the reference'


@pytest.mark.parametrize('algo', _all_algos())
def test_eval_block_matches_cpo(algo: str) -> None:
    """Every on-policy config declares the block, with CPO's values."""
    reference, cfgs = _algo_cfgs(REFERENCE), _algo_cfgs(algo)

    missing = [k for k in EVAL_KEYS if k not in cfgs]
    assert not missing, (
        f'{algo}.yaml is missing {missing}. _run_eval_studies gates on getattr(..., default), so '
        f'these studies would silently not run -- and custom_cfgs cannot add them at the call '
        f'site (KeyError: Invalid key). Copy the block from {REFERENCE}.yaml.'
    )

    for key in EVAL_KEYS:
        if key in STUDY_SWITCHES and algo in OBS_AUGMENTING:
            assert cfgs[key] is False, (
                f'{algo}.yaml sets {key}={cfgs[key]!r}, but its adapter augments the observation '
                f'space and the eval-env builders do not reproduce that yet. Either keep it False '
                f'or fix the builders and drop {algo} from OBS_AUGMENTING.'
            )
            continue
        assert cfgs[key] == reference[key], (
            f'{algo}.yaml has {key}={cfgs[key]!r}, {REFERENCE}.yaml has {reference[key]!r}. '
            f'The evaluation block is meant to be identical across on-policy algorithms so their '
            f'critic-calibration numbers are directly comparable.'
        )


if __name__ == '__main__':
    raise SystemExit(pytest.main([os.path.abspath(__file__), '-q']))
