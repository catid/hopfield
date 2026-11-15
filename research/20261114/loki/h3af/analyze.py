"""Audit H3af's stopped CPU gate; never resume compilation or target stages."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import csv
import json
import shutil
import time
from pathlib import Path
import numpy as np
from common import ROOT, SRC, OUT, read, save, sha, now, frozen
PUBLIC = ROOT/'results/research/loki/h3af'

def main():
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    p = frozen(target=True)
    gate, campaign = read(OUT/'compilation-gate.json'), read(OUT/'campaign.json')
    assert gate['status'] == 'failed_or_incomplete'
    assert campaign['status'] == 'stopped_with_failure'
    assert gate['protocol_sha256'] == sha(SRC/'protocol.json')
    assert not gate['partial_cache_usable'] and gate['aggregate_cpu_seconds'] <= 120
    assert 'TimeoutError' in gate['error'] and len(gate['completed_families']) == 1
    forbidden = ['external-graph-gate.json', 'graph-gate.json', 'datasets.json', 'global-budget.json', 'query-worker.json', 'query.log']
    assert all(not (OUT/name).exists() for name in forbidden)
    forbidden_globs = ['b*/data.npz', 'b*/query-inputs.npy', 'b*/dataset.json', 'b*/preflight.json', 'b*/exact/*', 'b*/target-worker-*.json', 'target-*.log']
    assert all(not list(OUT.glob(pattern)) for pattern in forbidden_globs)
    public_files = ['campaign.json', 'campaign.log', 'compilation-gate.json', 'compilation-progress.json', 'query-validation.json']
    rows = []
    for bank, cfg in enumerate(p['models']):
        folder = OUT/f'b{bank}'
        m, forward = dict(np.load(folder/'model.npz')), np.load(folder/'unverified-forward-columns.npy')
        assert int(m['q']) == 15 and forward.ndim == 2 and forward.shape[1] == 15
        assert forward.dtype == np.uint8 and int(forward.max()) < 15
        transfer = read(folder/'transfer.json')
        assert transfer['inherited_N13_model_identically_reconstructed']
        public_files += [f'b{bank}/{name}' for name in ['model.npz', 'transfer.json', 'unverified-forward-columns.npy']]
        if bank == 0:
            comp = read(folder/'compilation.json')
            assert comp == gate['completed_families'][0]
            assert comp['source_model_sha256'] == sha(folder/'model.npz')
            assert comp['cache_sha256'] == sha(folder/'cache.npz')
            assert comp['source_sha256'] == sha(SRC/'compiler.py')
            cache = dict(np.load(folder/'cache.npz'))
            assert np.array_equal(cache['columns'], forward) and len(forward) == comp['zero_F_solutions']
            cells = 15*np.arange(15)[None,:]+forward
            boards = np.zeros((len(cells),225),bool)
            boards[np.arange(len(cells))[:,None],cells] = True
            assert np.array_equal(boards,cache['boards'])
            assert np.array_equal(np.packbits(boards.T,axis=1,bitorder='little'),cache['index'])
            assert len(np.unique(forward,axis=0)) == len(forward)
            # Post-run certificate identity audit only; no enumeration or external geometry.
            ii,jj = np.triu_indices(15,1)
            assert not m['F'][cells[:,ii],cells[:,jj]].any()
            assert np.array_equal(m['F'],m['F'].T) and np.isin(m['F'],[0,1]).all()
            assert not m['h'].any() and (m['J'][m['F'].astype(bool)]>0).all() and not m['J'][~m['F'].astype(bool)].any()
            public_files += [f'b{bank}/{name}' for name in ['compilation.json','cache.npz']]
            row = dict(bank=bank,family=cfg['family'],method='BP AdamW model-only exact compilation',status='complete_verified_model_only',verified_states=len(forward),provisional_forward_states=None,primary_seconds=comp['primary_seconds'],independent_seconds=comp['independent_seconds'],complete_compile_cpu_seconds=comp['compile_cpu_seconds'],complete_compile_wall_seconds=comp['compile_wall_seconds'],externally_validated=False)
        else:
            assert not (folder/'compilation.json').exists() and not (folder/'cache.npz').exists()
            progress = gate['progress']
            assert progress['bank'] == bank and progress['stage'] == 'independent_reverse' and progress['unverified_forward_solutions'] == len(forward)
            row = dict(bank=bank,family=cfg['family'],method='BP AdamW model-only exact compilation',status='independent_enumeration_incomplete',verified_states=None,provisional_forward_states=len(forward),primary_seconds=progress['primary_seconds'],independent_seconds=None,complete_compile_cpu_seconds=None,complete_compile_wall_seconds=None,externally_validated=False)
        rows.append(row)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    for rel in public_files:
        target = PUBLIC/rel; target.parent.mkdir(exist_ok=True); shutil.copyfile(OUT/rel,target)
    summary = dict(status='validated_failure_audit',utc=now(),hypothesis='H3af',decision='inconclusive_not_run',reason='The shared 120 CPU-second complete-enumeration gate expired during the second independent reverse enumeration.',protocol_sha256=sha(SRC/'protocol.json'),compilation_gate=gate,campaign=campaign,family_results=rows,partial_cache_usable=False,external_graph_gate_run=False,target_cohorts_created=0,GPU_jobs_run=0,GPU_worker_seconds=0,query_calls_run=0,speed_ratios=None,amortization_estimates=None,uncompleted_family_observed_cpu_remainder=gate['aggregate_cpu_seconds']-rows[0]['complete_compile_cpu_seconds'],post_run_audit=dict(complete_cache_identity_index_and_literal_occupied_pairs_verified=True,provisional_forward_count_only=True,no_enumeration_rerun=True,no_external_geometry_access=True,absent_stage_paths_checked=forbidden+forbidden_globs,cpu_seconds=time.process_time()-start_cpu,wall_seconds=time.perf_counter()-start_wall),limits=p['limits'],input_sha256={str((OUT/rel).relative_to(ROOT)):sha(OUT/rel) for rel in public_files},public_sha256={rel:sha(PUBLIC/rel) for rel in public_files},audit_source_sha256=sha(__file__))
    save(PUBLIC/'summary.json',summary)
    with (PUBLIC/'compilation.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:summary[k] for k in ['decision','target_cohorts_created','GPU_jobs_run','post_run_audit']},indent=2))
if __name__ == '__main__': main()
