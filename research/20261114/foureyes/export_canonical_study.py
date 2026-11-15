"""Curated H4w model export and post-hoc exact zero-stop diagnostics."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
from certify_sudoku_soundness import close

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT/'artifacts/research/20261114/foureyes/canonical_study'
PUBLIC = ROOT/'results/research/foureyes/canonical_study'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary = json.loads((PUBLIC/'summary.json').read_text())
    arrays = {}
    for bank in range(64):
        path = RAW/'fits'/f'b{bank:02d}'/'models.npz'
        assert sha(path) == summary['input_sha256'][str(path.relative_to(ROOT))]
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                arrays.setdefault(key, []).append(data[key])
    np.savez_compressed(PUBLIC/'models.npz', **{k: np.stack(v) for k,v in arrays.items()})
    proof_rows, witnesses = [], {}
    for mapping, outcome in zip(summary['model_mappings'], summary['outcomes'], strict=True):
        bank, arm, job = mapping['bank'], mapping['arm'], mapping['actual']
        support = arrays[arm+'_support'][bank]
        trace, known = close(support)
        row = dict(bank=bank, arm=arm, job=job, trace=trace, certified_units=known,
                   sound=len(known)==27, observed_wrong_stops=outcome['wrong_stops'])
        if row['sound']:
            assert outcome['wrong_stops'] == 0
        elif outcome['wrong_stops']:
            if job not in witnesses:
                path = RAW/job/'moves-262144/boards.npz'
                assert sha(path) == summary['input_sha256'][str(path.relative_to(ROOT))]
                with np.load(path, allow_pickle=False) as data:
                    k = int(np.flatnonzero(data['stopped'] & ~data['valid'])[0])
                    witnesses[job] = dict(job=job, local_index=k, puzzle_index=int(data['puzzle'][k]),
                        board=data['boards'][k].astype(int).tolist(), solution=data['solutions'][k].astype(int).tolist(),
                        free=data['free'][k].astype(bool).tolist(), source_archive=str(path.relative_to(ROOT)),
                        source_sha256=sha(path))
            row['witness_job'] = job
        proof_rows.append(row)
    grouped = []
    for arm in ['adamw','canonical_half','canonical_weak','restricted']:
        rows = [r for r in proof_rows if r['arm']==arm]
        grouped.append(dict(arm=arm, sound=sum(r['sound'] for r in rows),
            unsound_witness=sum('witness_job' in r for r in rows),
            unresolved=sum(not r['sound'] and 'witness_job' not in r for r in rows)))
    gates = {}
    for name in ['canonical_pl_gate','canonical_half_gate']:
        source = ROOT/'artifacts/research/20261114/foureyes'/name
        dest = PUBLIC/name
        dest.mkdir(exist_ok=True)
        for path in sorted(source.iterdir()):
            if path.suffix in ['.json','.npz']:
                shutil.copyfile(path,dest/path.name)
                gates[str((dest/path.name).relative_to(ROOT))] = sha(dest/path.name)
    result = dict(scope='Post-hoc geometry-supplied exact zero-stop diagnostic; no additional training or search.',
        summary_sha256=sha(PUBLIC/'summary.json'), models_sha256=sha(PUBLIC/'models.npz'),
        producer_sha256=sha(__file__), rows=proof_rows, witnesses=list(witnesses.values()),
        grouped=grouped, source_gate_exports=gates)
    (PUBLIC/'certificates.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(grouped=grouped, unique_witnesses=len(witnesses))))


if __name__=='__main__':
    main()
