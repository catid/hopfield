"""Execution-only path correction for the frozen H4r search implementation.

The copied runner retained H4q's artifact directory. Initialization stopped on
its existing queue before any modification. Set the H4r directory explicitly;
all fitted-model hashes, protocol, search code and settings remain frozen.
"""
from pathlib import Path
import noisy_pl_search as runner

runner.OUT = runner.ROOT / 'artifacts/research/20261114/foureyes/noisy_pl'
if __name__ == '__main__':
    runner.main()
