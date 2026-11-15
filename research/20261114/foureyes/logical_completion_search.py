"""H4t adapter; the frozen H4s search loop is reused unchanged."""
from pathlib import Path
import noisy_pl_long_search as search

HERE = Path(__file__).resolve().parent
search.OUT = search.ROOT/'artifacts/research/20261114/foureyes/logical_completion'
search.PROTOCOL = HERE/'logical_completion.json'

if __name__ == '__main__':
    search.main()
