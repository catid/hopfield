"""Freeze a fresh search comparison before accessing its reserved target slice."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from verify_sudoku_soundness import verify

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/research/20261114/foureyes/logical_completion'
PUBLIC = ROOT/'results/research/foureyes/logical_soundness'
PROTOCOL = HERE/'logical_completion.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    assert not PROTOCOL.exists() and not OUT.exists()
    verification = verify()
    assert verification['status'] == 'passed'
    records = json.loads((PUBLIC/'certificates.json').read_text())
    z = np.load(PUBLIC/'supports.npz', allow_pickle=False)
    lookup = dict(zip(map(str, z['job_ids']), z['supports']))
    ii, jj = np.triu_indices(81, 1)
    row, col = np.arange(81)//9, np.arange(81) % 9
    truth = (row[ii] == row[jj]) | (col[ii] == col[jj]) | ((row[ii]//3 == row[jj]//3) & (col[ii]//3 == col[jj]//3))
    assert truth.sum() == 810
    selected = [r for r in records['rows'] if r['sound'] and (~lookup[r['job']][truth]).any()]
    assert {a: sum(r['arm'] == a for r in selected) for a in ['adamw', 'paper_pl', 'long_pl']} == dict(adamw=13, paper_pl=8, long_pl=11)
    old = json.loads((HERE/'noisy_pl_long.json').read_text())
    source_files = [HERE/name for name in ['prepare_logical_completion.py', 'logical_completion_search.py', 'audit_logical_completion.py', 'noisy_pl_long_search.py', 'verify_sudoku_soundness.py', 'audit_noisy_pl_long.py']]
    source_hashes = dict(old['source_sha256'])
    source_hashes.update({str(p.relative_to(ROOT)): sha(p) for p in source_files})
    p = dict(id='H4t-logically-implied-edges-search', frozen_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='On the13selected sound BPAdamW supports missing trueedges, adding all implied Sudoku peeredges improves fresh short-cap accuracy by at least1percentagepoint with zero wrongstops.',
        selection=[dict(bank=r['bank'], source_arm=r['arm'], source_job=r['job'], support_sha256=r['support_sha256']) for r in selected],
        selection_scope='Adaptive follow-up after H4s and its exact soundness diagnostic. Fixed historical models selected by soundness and missing edges; no fresh outcomes used.',
        primary=dict(source_arm='adamw', target_gain_pp=1.0, interval='two-sided95percent paired input-cluster bootstrap; average13 fixedmodels within each input;5000draws; seed20261115555', stop_gate='zero wrong certified stops in both primary arms'),
        secondary='SGD32 andSGD256 source subsets descriptive only; no optimizer-superiority or independent new training-bank claim.',
        scientific_change='Original unit graph versus original union810officialpeeredges. Geometry explicitly supplied. Proof shows identical zero sets, not identical nonzero energies or trajectories.',
        source_sha256=source_hashes,
        frozen_inputs={str((PUBLIC/name).relative_to(ROOT)): sha(PUBLIC/name) for name in ['certificates.json', 'supports.npz']},
        split='full-all', rows_half_open=[320000, 330000], search_seed=0, chains_by_arm={'original':4, 'completed':4},
        checkpoints=[0, 4*262144], settings=dict(unit=1, bias=0, tau=0, ebar=0, temperature=[.4,.31], annealing_moves=524288, focus_threshold=.45, check=4096),
        budget=dict(max_queue_wall_seconds=300, max_gpu_minutes=20, gpus=[0,1,2,3]),
        input_scope='New10k contiguous slice from reused officialtestcorpus; seed0fixed. No freshfit or acquisition.',
        numerical_gate='Inherited native/capture/sparse parity on first16inputs,8checks peruniquegraph; exact integer fields/energies atcap, independent savedboard audit.',
        verifier_receipt=verification)
    save(PROTOCOL, p)
    jobs = []; mapping = []; seen = {}
    for model in selected:
        original = lookup[model['job']]
        for arm, support in [('original', original), ('completed', original | truth)]:
            digest = hashlib.sha256(support.tobytes()).hexdigest()
            if digest not in seen:
                job = f'm{len(seen):03d}';seen[digest] = job
                path = OUT/'models'/f'{job}.npz';path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, support=support)
                jobs.append(dict(job_id=job, bank=model['bank'], source_arm=model['arm'], arm=arm, n=600, seed=0,
                    threshold='logical-completion', model_path=str(path.relative_to(ROOT)), model_sha256=sha(path), support_sha256=digest, status='pending'))
            mapping.append(dict(bank=model['bank'], source_arm=model['arm'], arm=arm, actual=seen[digest], support_sha256=digest))
    save(OUT/'prepared.json', dict(protocol_sha256=sha(PROTOCOL), jobs=jobs, mappings=mapping,
        reuse='Byte-identical unit supports with allsearchsettings,inputcohort,andseedmatched. Preserve logicalcellweights.', logical_cells=len(mapping), unique_searches=len(jobs)))
    print(json.dumps(dict(protocol_sha256=sha(PROTOCOL), logical_cells=len(mapping), unique_searches=len(jobs))))


if __name__ == '__main__':
    main()
