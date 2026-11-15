"""Independent verification of every BFS board and workload-labeled comparison."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import time
import numpy as np
import audit as A
import study as S


def one_shard(args):
    tokens,boards,distances=args;checked=[]
    for token,board,distance in zip(tokens,boards,distances):
        ok,expected=A.independent(token,board.astype(bool));assert ok and expected==int(distance) and int(board.sum())==int(distance)+1;checked.append(expected)
    return checked


def main():
    started=time.perf_counter();dest=S.OUT/'bfs_scope_control';source=S.OUT/'window_fallback';result=json.loads((dest/'result.json').read_text());assert S.sha(dest/'outputs.npz')==result['outputs_sha256'];manifest=json.loads((source/'datasets.json').read_text());pieces=[];distances=[]
    for ds in manifest['datasets']:
        assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as x:pieces.append(x['tokens']);distances.append(x['shortest_edges'])
    with np.load(dest/'outputs.npz') as x:boards=x['predictions'];got=x['distances'];status=x['status']
    tokens=np.concatenate(pieces);expected=np.concatenate(distances);assert hashlib.sha256(tokens.tobytes()).hexdigest()==result['input_tokens_sha256'];assert boards.shape==tokens.shape==(16384,120,120) and np.all(status==1) and np.array_equal(got,expected)
    with ProcessPoolExecutor(max_workers=8) as pool:checked=list(pool.map(one_shard,[(pieces[s],boards[s*256:(s+1)*256],got[s*256:(s+1)*256]) for s in range(64)]))
    assert np.array_equal(np.concatenate(checked),expected);previous=json.loads((source/'audit.json').read_text());comparisons=[];bfs=float(result['median_solve_and_cpu_output_seconds'])
    for policy in ['bounded1024','cohort_end']:
        rows=[r for r in previous['timing'] if r['policy']==policy];common=float(np.median([r['native_common_seconds'] for r in rows]));total=float(np.median([r['native_common_seconds']+r['incremental_seconds'] for r in rows]));native_answer=float(np.median([r['cost_seconds']['native_answer'] for r in rows]))
        comparisons.append({'policy':policy,'measured_cram_model_input_outputs':49152,'native_common_summed_gpu_seconds':common,'decoder_common_plus_incremental_summed_gpu_seconds':total,'native_answer_summed_gpu_seconds':native_answer,'native_common_seconds_per_model_input':common/49152,'decoder_seconds_per_model_input':total/49152,'bfs_cpu_seconds_per_graph':bfs/16384,'decoder_per_model_input_over_bfs_per_graph_ratio':(total/49152)/(bfs/16384),'comparison_caution':'Average per-model per-input compute, different hardware/algorithm; summedGPUcomponents are not parallelwall latency. BFSreturns one exactpath pergraph; CRAMstudy evaluated3fixed models pergraph.'})
    audit={'status':'validated','utc':S.utc(),'reused_unique_inputs':16384,'shortest_valid_boards':16384,'independent_distance_and_graph_checks':16384,'independent_checker_source_sha256':S.sha(A.__file__),'processes':8,'solver_result_sha256':S.sha(dest/'result.json'),'outputs_sha256':S.sha(dest/'outputs.npz'),'script_sha256':S.sha(__file__),'solver':result,'comparison':comparisons,'audit_seconds':time.perf_counter()-started,'scope':'Hand-engineered suppliedgraph BFS comparator. No training, no CRAMselection intervention, no new independent input cohort. Retains directsolver advantage as measured.'}
    S.save(dest/'audit.json',audit);S.emit('bfs_scope_audit_passed',inputs=16384,median_solver_seconds=bfs,seconds=audit['audit_seconds'])


if __name__=='__main__':main()
