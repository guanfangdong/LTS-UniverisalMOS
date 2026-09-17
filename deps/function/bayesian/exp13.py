"""Compatibility entry point for deps.bayesian.exp13."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deps.bayesian.exp13 import *  # noqa: F401,F403
from deps.bayesian.exp13 import main as _main

if __name__ == "__main__":
    _main()
