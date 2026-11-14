"""H3d: learned-certificate-first output selection on immutable saved searches."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'research/20261114/loki/h3d'
OUT = ROOT / 'artifacts/research/20261114/loki/h3d'


def sha(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def choose(energy, stopped):
    """Learned flags only; no puzzle rules, labels or saved correctness enters selection."""
    import numpy as np
    original = energy.argmin(axis=1)
    certified = np.where(stopped, energy, np.inf).argmin(axis=1)
    return np.where(stopped.any(axis=1), certified, original)


def prepare():
    assert not (SRC / 'protocol.json').exists(), 'Frozen protocol already exists'
    cases = []
    for arm, updates in [('adamw-decay', 4000)] + [(a, u) for a in ['adam-nodecay', 'sgd-pre'] for u in [250, 500, 1000, 2000, 4000]]:
        for seed in [0, 1, 2]:
            stem = f'artifacts/research/20261114/loki/h3b/primary/{arm}/s{seed}/final' if updates == 4000 else f'artifacts/research/20261114/loki/h3c/search/{arm}/u{updates}/s{seed}/'
            record = ROOT / (stem + '.json' if updates == 4000 else stem + 'result.json')
            arrays = ROOT / (stem + '.npz' if updates == 4000 else stem + 'records.npz')
            rec = json.loads(record.read_text()); assert sha(arrays) == rec['records_sha256']
            cases.append(dict(arm=arm, updates=updates, seed=seed, record=str(record.relative_to(ROOT)),
                arrays=str(arrays.relative_to(ROOT)), record_sha256=sha(record), arrays_sha256=sha(arrays),
                model=rec['model_path'], model_sha256=rec['model_sha256']))
    protocol = dict(id='H3d-certificate-first', frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        hypothesis='Prioritizing the lowest-energy stopped chain when one exists improves decay-AdamW N20 accuracy by at least 0.2 percentage points, using the identical saved search trajectories.',
        antecedent='H3b/H3c audits found zero wrong stopped chains but some N20 errors under lowest-energy selection over stopped and capped chains. Existence of a recoverable stopped chain among those errors was not inspected before freezing this protocol.',
        primary=dict(arm='adamw-decay', updates=4000, minimum_gain_pp=.2, unique_inputs=6513, search_seeds=[0, 1, 2]),
        selector='For each input, select argmin energy among stopped chains if any; otherwise use original argmin over all chains. NumPy argmin first-index tie break retained. No learned weights, temperatures, stopping conditions, chain count or trajectory is changed.',
        decision='Support if one-sided 95% paired-input bootstrap lower accuracy gain >=0.2pp; falsified if upper gain <0.2pp; otherwise inconclusive. Resample unique inputs with all three search seeds retained, 2000 draws, RNG20261114.',
        secondary='Every prior optimizer/checkpoint is reported. Coverage is unchanged because all returned chains are identical; N1 and search work also unchanged. Selection may change N20 correctness; no feedback into search.',
        certificate_audit='Recompute float64 full Ising single-cell margins on every saved returned board, masking givens. For each stopped flag, compare min free-cell margin to model.delta-0.001. Report strict failures and maximum signed violation. Numerical allowance 0.001 absolute energy units; reject interpretation as reliably certified if a stopped board violates by more than this. Capped returned boards are best-energy visited boards, not necessarily last states, so do not assert their flags equal recomputed final-board certificates.',
        selector_validation='Independently choose using a literal row loop and stopped-index subsets; compare identity to vectorized selector. Confirm the original lowest-energy index reproduces saved N20. Prove no-certificate fallback identity; report selected-validity and stopped/capped strata.',
        timing=dict(scope='CPU selection only on the primary seed0 energy/status arrays, excluding I/O, certificate recomputation and rule checking; original argmin and new selector measured separately.',
                    warmups_each=10, repeats=101, order='Alternate baseline/new and new/baseline each repeat; retain every timing and report medians, overhead and ratio. This is not a search speedup and no extra search does not mean zero selector cost.'),
        source_sha256={str(Path(__file__).relative_to(ROOT)): sha(__file__)}, cases=cases,
        limits=['This is an adaptive saved-output reanalysis on a previously used official test set, not independent test confirmation.',
                'The selector differs from the paper lowest-energy-over-all-chains convention and is a derivative policy.',
                'Only deterministic fitted models and repeated search seeds already studied are covered; no cross-task reliability claim.',
                'A learned stopping condition is not a general proof of puzzle correctness; independent rules are used only after selection for evaluation.'])
    SRC.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    (SRC / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(dict(status='frozen', protocol_sha256=sha(SRC / 'protocol.json'), cases=len(cases))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['prepare']); args = ap.parse_args(); prepare()
