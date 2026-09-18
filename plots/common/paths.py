"""Where the figures are written: plots/figures/, beside the scripts that make them.

Anchored to this file rather than to the working directory, because the scripts
have to be run from the RL_CAL root -- that is where the eval_data_* trees they
read live.
"""

from __future__ import annotations

from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def fig_path(name: str) -> str:
    """Output stem for a figure, e.g. fig_path("fig1") -> <repo>/plots/figures/fig1."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return str(FIG_DIR / name)
