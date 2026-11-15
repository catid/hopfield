"""Independent greedy/MILP cover audit, exact lower certificate and inference reuse."""
import csv
import hashlib
import json
import math
import numpy as np
import torch
from run import ROOT,HERE,save,sha,now

OUT = ROOT/'artifacts/research/20261114/ripper/compression'
PUBLIC = ROOT/'results/research/ripper'
PROTOCOL = HERE/'protocol-compression.json'


def main():
    p = json.loads(PROTOCOL.read_text()); manifest = json.loads((ROOT/p['inputs']).read_text())
    assert sha(ROOT/p['inputs'])==p['input_manifest_sha256']
    pair = np.triu_indices(81,1); i,j = pair
    peer = (i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3))
    different = (i//27!=j//27)&(i%9//3!=j%9//3)
    assert peer.sum()==810 and different.sum()==1458
    rows = []; selections = []; hashes = {}; reuse = []; certificates = []
    reference = json.loads((PUBLIC/'clues.json').read_text()); assert reference['status']=='validated'
    for seed in p['seeds']:
        d = OUT/f's{seed}'; r = json.loads((d/'receipt.json').read_text())
        assert r['status']=='complete' and r['protocol_sha256']==sha(PROTOCOL)
        assert r['wrapper_sha256']==sha(HERE/'compress_pool.py') and r['highs_version']=='1.12.0'
        assert r['solver_options']['threads']==1 and r['solver_options']['time_limit']==55.
        assert r['solver_process_cpu_s']<=60 and r['pool_labels_read']==1000
        for name,h in r['output_sha256'].items(): assert sha(d/name)==h
        e = next(e for e in manifest['inputs'] if e['seed']==seed); src = ROOT/e['path']
        assert sha(src)==e['sha256']==r['training_sha256']; z = np.load(src)
        sol,source = z['solutions'],z['source_row']; assert len(sol)==1000 and len(np.unique(source))==1000
        # Independent construction by per-digit cell groups instead of broadcasting all cell pairs.
        matrix = np.zeros((1000,3240),bool)
        for row,board in enumerate(sol):
            for digit in range(9):
                cells = np.flatnonzero(board==digit); assert len(cells)==9
                for k,left in enumerate(cells[:-1]):
                    right = cells[k+1:]; indices = left*(161-left)//2 + right-left-1
                    matrix[row,indices]=True
        universe = matrix.any(0); support = ~universe
        assert np.all(matrix.sum(1)==324) and universe.sum()==r['universe_size']==2430
        assert np.array_equal(support,peer) and r['counting_lower_bound']==math.ceil(universe.sum()/324)==8
        assert hashlib.sha256(support.tobytes()).hexdigest()==r['support_bytes_sha256']
        # Every greedy decision must attain the maximum gain with the frozen source-row tie break.
        covered = np.zeros(3240,bool)
        for t in r['greedy_trace']:
            gain = (matrix&~covered).sum(1); candidates = np.flatnonzero(gain==gain.max())
            expected = candidates[np.argmin(source[candidates])]
            assert t['pool_index']==expected and t['source_row']==source[expected] and t['new_pairs']==gain.max()
            covered |= matrix[expected]; assert t['remaining_pairs']==int((universe&~covered).sum())
        assert np.array_equal(covered,universe) and len(r['greedy_trace'])==r['greedy_size']
        for method,expected_size in [('greedy',r['greedy_size']),('milp',r['incumbent_size'])]:
            selected = np.load(d/f'{method}_selection.npz'); indices = selected['pool_indices']
            assert len(indices)==len(np.unique(indices))==expected_size
            assert np.array_equal(selected['source_rows'],source[indices]) and np.array_equal(selected['solutions'],sol[indices])
            assert np.array_equal(selected['support'],support) and np.array_equal(matrix[indices].any(0),universe)
            for ix in indices: selections.append({'seed':seed,'method':method,'pool_index':int(ix),'source_row':int(source[ix])})
        # Post-hoc exact lower certificate: nine 162-pair sets would need to be disjoint to cover1458.
        b = matrix[:,different]; assert np.all(b.sum(1)==162) and b.any(0).all()
        packed = np.packbits(b,axis=1); histogram = np.zeros(163,np.int64); minimum = 163
        for row in range(999):
            overlap = np.bitwise_count(packed[row+1:]&packed[row]).sum(1).astype(np.int32)
            minimum = min(minimum,int(overlap.min())); histogram += np.bincount(overlap,minlength=163)
        assert histogram.sum()==499500 and minimum>0
        assert r['solver_dual_bound']==10. and not r['optimum_claim']
        assert r['solver_status']=='Time limit reached'
        assert abs(r['solver_relative_gap']-(r['incumbent_size']-10)/r['incumbent_size'])<1e-12
        certificates.append({'seed':seed,'orbit_pairs':1458,'pairs_per_board':162,
            'candidate_pairs_checked':499500,'minimum_candidate_overlap':minimum,
            'exact_finite_pool_lower_bound':10,'intersection_histogram':histogram.tolist(),
            'scope':'Post-selection geometry certificate; not used by the greedy selector or MILP.'})
        # Model equality makes inference identical for every puzzle, not merely a tested sample.
        expectedW = torch.from_numpy(support.astype(np.float32))[:,None,None]*torch.eye(9)
        refdir = ROOT/f'artifacts/research/20261114/ripper/clues/jobs/full-n300-s{seed}'
        receipt = json.loads((refdir/'receipt.json').read_text())
        model = torch.load(refdir/'searched_model.pt',weights_only=True,map_location='cpu')
        assert sha(refdir/'searched_model.pt')==receipt['output_sha256']['searched_model.pt']
        assert torch.equal(expectedW,model['W']) and not model['B'].any()
        assert receipt['used_unit']==1 and receipt['tau']==receipt['ebar']==0
        rng = np.random.default_rng(20261124+seed); boards = rng.integers(0,9,(4096,81),dtype=np.int8)
        es = (boards[:,i[support]]==boards[:,j[support]]).sum(1)
        for method in ['greedy','milp']:
            ss = np.load(d/f'{method}_selection.npz')['support']
            assert np.array_equal(es,(boards[:,i[ss]]==boards[:,j[ss]]).sum(1))
        reuse.append({'seed':seed,'canonical_job':receipt['job_id'],'test_rows_half_open':[100000,110000],
            'existing_solved':receipt['solved'],'n_test':10000,'new_gpu_searches':0,
            'support_B_unit_tau_ebar_identical':True,'move_cap':262144,'chains':4,
            'reference_receipt_sha256':sha(refdir/'receipt.json')})
        row = {k:r[k] for k in ['seed','pool_labels_read','universe_size','greedy_size','incumbent_size',
            'solver_dual_bound','solver_relative_gap','solver_status','solver_process_cpu_s','solver_wall_s','mip_nodes','greedy_wall_s']}
        row.update(exact_counting_lower_bound=8,exact_orbit_lower_bound=9,exact_finite_pool_lower_bound=10,
            minimum_candidate_B_overlap=minimum,greedy_primary_pass=r['greedy_size']<=16)
        rows.append(row); hashes[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
    decision = 'supported_on_all_six_pools' if all(r['greedy_primary_pass'] for r in rows) else 'falsified_for_fixed_greedy_algorithm'
    report = {'status':'validated','utc':now(),'primary_decision':decision,'primary_greedy_limit':16,
        'selection_cost':'All1000 complete labels per pool are read before selection. Compression, not few-label learning.',
        'primary_scope':'The <=16 claim is falsified for the fixed greedy algorithm. Bounded MILP timeouts do not falsify existence of smaller covers.',
        'rows':rows,'exact_lower_certificates':certificates,'selected_boards':selections,
        'inference_reuse':reuse,'new_gpu_searches':0,'random_board_energy_checks':6*2*4096,
        'candidate_pair_overlap_checks':6*499500,'protocol_sha256':sha(PROTOCOL),'audit_sha256':sha(__file__),
        'lower_bound_proof_sha256':sha(HERE/'COMPRESSION_LOWER_BOUND.md'),'receipt_sha256':hashes,
        'existing_public_inference_sha256':sha(PUBLIC/'clues.json'),
        'limitations':p['limitations']+['All six MILPs timed out with large remaining gaps. Their warm-start covers were not improved.',
            'The exact ten-board lower bound is a post-hoc finite-pool certificate, separate from the frozen greedy hypothesis.',
            'Matching full-count inference is reused by exact energy equality; it adds no independent accuracy evidence.']}
    save(OUT/'audit.json',report); save(PUBLIC/'compression.json',report)
    for name,rs in [('compression_per_seed.csv',rows),('compression_selected.csv',selections)]:
        with (PUBLIC/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    summary = json.loads((PUBLIC/'summary.json').read_text());summary['training_pool_compression']=report;save(PUBLIC/'summary.json',summary)
    lines = ['# H1r: compressing a fully labeled training pool','',
        '**Selection reads all 1,000 complete labels per bank. This is training-set compression, not learning from only the retained labels.**','',
        'The frozen hypothesis that the fixed greedy algorithm selects at most 16 boards in every pool is **falsified**. This rule retains 26–27 boards in the six pools. Every retained subset produces the same count energy as its full pool. The bounded MILP timeouts do not falsify the existence of smaller covers.','',
        '| Seed | Labels read | Greedy cover | MILP incumbent | Exact lower bound | Solver status | Gap |',
        '|---:|---:|---:|---:|---:|---|---:|']
    for r in rows:
        lines.append(f'| {r["seed"]} | 1,000 | {r["greedy_size"]} | {r["incumbent_size"]} | 10 | Time limit | {100*r["solver_relative_gap"]:.2f}% |')
    lines += ['', 'Each MILP received its greedy cover as a feasible warm start. All six runs hit the 55-second wall cap, used less than the 60-CPU-second budget and retained their initial incumbents. HiGHS reports a lower bound of ten for every pool; the remaining 61.5–63.0% gaps prevent an optimality claim.','',
        'The selector sees only equal-digit pair coverage and source-row identifiers. It greedily maximizes newly covered pairs, breaking ties by the original source row. A reversed pool order and independent per-board digit relabelings preserve the selected coverage. No peer geometry is used for construction.','',
        'An independent post-selection geometry certificate establishes the ten-board lower bound exactly, separately from the HiGHS floating-point dual estimate of ten. The elementary counting bound is ceil(2430/324)=8. Every board covers 162 of the 1,458 different-band-and-stack pairs, giving the exact orbit bound nine. Nine boards could cover that orbit only without overlap, but every pair of candidate boards in each pool overlaps on at least one such pair, giving the exact finite-pool bound ten. All 2,997,000 candidate-pair overlaps were checked with integer arithmetic. This does not determine the minimum over boards outside the fixed pools.','',
        'Greedy and MILP subsets were independently verified against the full equality union. Their unit1 equal-digit energies, zero biases and zero stopping/focus calibration are identical to the existing full-count control. No duplicate GPU inference was run. The mapped six searches on [100000,110000) remain the earlier 80.9517% short-budget result, not new evidence.','',
        'BP AdamW is outside this count-energy compression study and was not run.','',
        '[Protocol](protocol-compression.json) · [Selector and solver](compress_pool.py) · [Independent audit](audit_compression.py) · [Exact lower-bound proof](COMPRESSION_LOWER_BOUND.md) · [Public results](../../../results/research/ripper/compression.json)','']
    (HERE/'REPORT-compression.md').write_text('\n'.join(lines))
    print(json.dumps({'primary_decision':decision,'rows':rows,'new_gpu_searches':0}))


if __name__ == '__main__': main()
