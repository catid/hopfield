"""Post-seal graph diagnosis and exact H3w outcome-identity mapping."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'research/20261114/loki/h3x'
OUT = ROOT / 'artifacts/research/20261114/loki/h3x'
PUBLIC = ROOT / 'results/research/loki/h3x'
BASE = ROOT / 'artifacts/research/20261114/loki/h3w'
SOURCE_SEAL = '1ec9ed198c923c8f5570c51829e42d12b58e9946e02a2c66edbf82ad22eaf39b'
HASHES = {}
def sha(p):
    p = Path(p)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    HASHES[str(p.relative_to(ROOT))] = digest
    return digest
def read(p):
    sha(p)
    return json.loads(Path(p).read_text())
def save(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False)+'\n')
def csvsave(p, rows):
    with Path(p).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    assert socket.gethostname() == 'loki'
    start = time.perf_counter()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    p = read(SRC/'protocol.json')
    source = read(OUT/'source-result.json')
    assert sha(OUT/'source-result.json') == SOURCE_SEAL
    assert source['protocol_sha256'] == sha(SRC/'protocol.json')
    assert source['status'] == 'source_selected_and_independently_certified'
    assert source['certificate_npz_sha256'] == sha(OUT/'source-certificate.npz')
    assert source['verification_sha256'] == sha(OUT/'source-verification.json')
    assert source['selection_sha256'] == sha(OUT/'source-selection.json')
    assert read(OUT/'source-verification.json')['status'] == 'passed'
    for group in ['source_sha256', 'source_data_sha256']:
        for path, digest in p[group].items():
            assert sha(ROOT/path) == digest
    seal_path = OUT/'source-seal.json'
    if not seal_path.exists():
        save(seal_path, dict(status='sealed_before_target_diagnostics', utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                            source_result_sha256=SOURCE_SEAL, protocol_sha256=sha(SRC/'protocol.json'),
                            externally_reported_before_target_diagnostics=True))
    assert read(seal_path)['source_result_sha256'] == SOURCE_SEAL
    # No target file is opened above this boundary.
    identity = p['identity_mapping']
    old_protocol = read(ROOT/'research/20261114/loki/h3w/protocol.json')
    assert sha(ROOT/'research/20261114/loki/h3w/protocol.json') == identity['reference_protocol_sha256']
    assert sha(ROOT/'research/20261114/loki/h3w/backend.py') == identity['backend_sha256']
    assert sha(BASE/'data.npz') == identity['data_sha256']
    assert old_protocol['search'] == {key: identity['settings'][key] for key in old_protocol['search']}
    assert old_protocol['search']['seeds'] == [0, 1, 2]
    assert sha(BASE/'models/oracle-permutation/model.npz') == identity['reference_model_sha256']
    oracle_model = dict(np.load(BASE/'models/oracle-permutation/model.npz'))
    old_summary = read(ROOT/'results/research/loki/h3w/summary.json')
    assert old_summary['status'] == 'validated'
    for path, digest in old_summary['input_sha256'].items():
        assert sha(ROOT/path) == digest, path
    data = dict(np.load(BASE/'data.npz'))
    n = 13
    i, j = np.triu_indices(n*n, 1)
    dr, dc = j//n-i//n, j%n-i%n
    true_edges = (dr == 0) | (dc == 0) | (dr == np.abs(dc)) | ((dr == 1) & (np.abs(dc) == 3)) | ((dr == 3) & (np.abs(dc) == 1))
    assert np.array_equal(oracle_model['F'][i, j].astype(bool), true_edges)
    spec = importlib.util.spec_from_file_location('h3w_audit', ROOT/'research/20261114/loki/h3w/analyze.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    target_keys = {r.tobytes() for r in data['all_solutions']}
    minima = []
    mapped_rows = []
    per_seed = []
    fresh_audits = []
    for mask in source['all_minimum_masks']:
        finite = {tuple(offset) for orbit in source['orbits'] if mask & (1 << orbit['index']) for offset in orbit['offsets']}
        inferred = np.array([any(a*int(r)+b*int(c) == 0 for a, b in source['inferred_lines']) or (int(r), int(c)) in finite for r, c in zip(dr, dc)])
        F = np.zeros((169, 169), np.int32)
        F[i, j] = inferred
        F[j, i] = inferred
        model = dict(J=F.astype(np.float32), h=np.zeros(169, np.float32), F=F, unit=np.array(1.), q=np.array(13))
        folder = OUT/f'models/minimum-{mask}'
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/'model.npz', **model)
        equal = {}
        for key, arr in model.items():
            ref = oracle_model[key]
            equal[key] = dict(dtype=str(arr.dtype), shape=list(arr.shape), c_order_sha256=hashlib.sha256(arr.tobytes(order='C')).hexdigest(),
                              reference_c_order_sha256=hashlib.sha256(ref.tobytes(order='C')).hexdigest(),
                              dtype_shape_bytes_equal=arr.dtype == ref.dtype and arr.shape == ref.shape and arr.tobytes(order='C') == ref.tobytes(order='C'))
        match = all(x['dtype_shape_bytes_equal'] for x in equal.values())
        capacity, capinfo = audit.capacity(model, data)
        rec = dict(mask=mask, generators=[o['generator'] for o in source['orbits'] if mask & (1 << o['index'])],
                   forbidden_target_pairs=int(inferred.sum()), true_target_pairs=int(true_edges.sum()),
                   false_target_pairs=int((inferred & ~true_edges).sum()), missed_target_pairs=int((~inferred & true_edges).sum()),
                   all_parameter_arrays_identical_to_h3w_oracle=match, parameter_arrays=equal,
                   model_sha256=sha(folder/'model.npz'), npz_container_byte_identical=sha(folder/'model.npz') == identity['reference_model_sha256'],
                   exact_zero_set_capacity=capinfo, identity_mapping=None)
        if match:
            assert float(model['unit']) == identity['settings']['unit'] == 1
            assert int(model['q']) == identity['settings']['q'] == 13
            for seed in identity['settings']['seeds']:
                records, row, diagnostic = audit.audit('oracle-permutation', seed, model, data, target_keys, capacity)
                row.update(arm=f'minimum-{mask}-permutation', evidence='identity-derived H3w oracle outcome; no new search', reference_arm='oracle-permutation')
                per_seed.append(row)
                diagnostic.update(mapped_minimum_mask=mask, evidence='new saved-state audit of inherited H3w records')
                fresh_audits.append(diagnostic)
            rec['identity_mapping'] = dict(reference_arm='oracle-permutation', settings=identity['settings'],
                                            backend_sha256=identity['backend_sha256'], data_sha256=identity['data_sha256'],
                                            inherited_search_seeds=[0, 1, 2], new_searches=0, reaudited_saved_states=3*4096*20)
            old = next(row for row in old_summary['rows'] if row['arm'] == 'oracle-permutation')
            mapped_rows.append(dict(old, arm=f'minimum-{mask}-permutation', evidence='identity-derived H3w oracle outcomes; no new search', source_minimum_mask=mask))
        minima.append(rec)
        shutil.copy2(folder/'model.npz', PUBLIC/f'minimum-{mask}-model.npz')
    supported = len(minima) == 1 and minima[0]['false_target_pairs'] == 0 and minima[0]['missed_target_pairs'] == 0
    decision = 'supported' if supported else 'falsified'
    baseline_rows = [dict(row, evidence='H3w published reference, same cohort; no H3x rerun') for row in old_summary['rows'] if row['arm'] in ['bp-adamw-permutation', 'uniform-filtered-permutation']]
    summary = dict(status='validated', utc=dt.datetime.now(dt.timezone.utc).isoformat(), hypothesis='H3x', decision=decision,
                   primary='Unique source minimum and exact N13 rule graph, conditional on source-completeness and D4/minimum-cardinality priors',
                   source_result_sha256=SOURCE_SEAL, source_result=source, target_minima=minima,
                   target_error_ranges=dict(false_pairs_min=min((x['false_target_pairs'] for x in minima), default=None),
                                            false_pairs_max=max((x['false_target_pairs'] for x in minima), default=None),
                                            missed_pairs_min=min((x['missed_target_pairs'] for x in minima), default=None),
                                            missed_pairs_max=max((x['missed_target_pairs'] for x in minima), default=None)),
                   rows=baseline_rows+mapped_rows, dataset=old_summary['dataset'], inherited_outcome_audits=fresh_audits,
                   identity_mappings=len(mapped_rows), new_optimizer_fits=0, new_gpu_search_jobs=0, independent_learning_replicates=1,
                   inherited_search_seed_count=3 if mapped_rows else 0, new_saved_state_audits=len(fresh_audits)*4096*20,
                   source_wall_seconds=source['total_source_wall_seconds'], source_cpu_seconds=source['total_source_cpu_seconds'],
                   target_diagnostic_and_mapping_seconds=time.perf_counter()-start, limits=p['limits'], input_sha256=HASHES)
    sha(__file__)
    for name in ['source-result.json', 'source-selection.json', 'source-verification.json', 'source-certificate.npz', 'source-seal.json']:
        shutil.copy2(OUT/name, PUBLIC/name)
    if per_seed:
        csvsave(PUBLIC/'mapped_per_seed.csv', per_seed)
    csvsave(PUBLIC/'all_subsets.csv', [dict(mask=s['mask'], cardinality=s['cardinality'], rejected_negatives=s['rejected_negatives'],
                                          covers_all_negatives=s['covers_all_negatives'],
                                          first_uncovered_universe_index=s['first_uncovered_universe_index'],
                                          first_uncovered_columns=json.dumps(s['first_uncovered_columns'])) for s in source['subsets']])
    save(PUBLIC/'summary.json', summary)
    print(json.dumps(dict(decision=decision, minimum_cardinality=source['minimum_cardinality'], selected_generators=source['selected_generators'],
                         target_diagnostics=[{k: x[k] for k in ['mask', 'forbidden_target_pairs', 'false_target_pairs', 'missed_target_pairs', 'all_parameter_arrays_identical_to_h3w_oracle', 'npz_container_byte_identical']} for x in minima],
                         rows=summary['rows'], new_gpu_search_jobs=0, reaudited_saved_states=summary['new_saved_state_audits'])))

if __name__ == '__main__':
    main()
