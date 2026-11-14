"""Frozen randomization test, conditional expectation and search audit for H1q."""
import csv
import hashlib
import json
import numpy as np
import torch
from run import ROOT,HERE,save,sha,now
from audit_clues import validity

OUT = ROOT/'artifacts/research/20261114/ripper/masks'
PUBLIC = ROOT/'results/research/ripper'
PROTOCOL = HERE/'protocol-masks.json'


def main():
    p = json.loads(PROTOCOL.read_text()); manifest = json.loads((ROOT/p['input_manifest']).read_text())
    assert sha(ROOT/p['input_manifest']) == p['input_manifest_sha256']
    assert sha(ROOT/p['official_models']) == p['official_models_sha256']
    old = json.loads((ROOT/p['official_models']).read_text())
    pair = np.triu_indices(81,1); i,j = pair
    peer = (i//9 == j//9)|(i%9 == j%9)|((i//27 == j//27)&(i%9//3 == j%9//3))
    assert peer.sum()==810
    expected_rows = []; draws_rows = []; first_rows = []; audits = []; counts = []; official = []; supports = {}
    completion = {}; hashes = {}; timings = []
    for seed in p['seeds']:
        d = OUT/f's{seed}'; r = json.loads((d/'receipt.json').read_text())
        a = json.loads((d/'audit_counts.json').read_text())
        assert a['status']=='validated' and a['protocol_sha256']==sha(PROTOCOL)
        assert a['wrapper_sha256']==sha(HERE/'audit_mask_counts.py') and a['count_receipt_sha256']==sha(d/'receipt.json')
        assert r['wrapper_sha256']==sha(HERE/'mask_count.py') and r['protocol_sha256']==sha(PROTOCOL)
        assert a['matrix_prefix_checks']==3072 and a['independent_cpu_dense_prefix_checks']==6
        assert a['full_rng_mask_replay_verified'] and a['mask_count_preservation_verified']
        assert a['hidden_value_invariance_checks']==4
        for name,h in r['output_sha256'].items(): assert sha(d/name)==h
        audits.append(a); timings.append({'seed':seed,'count_elapsed_s':r['elapsed_s'],'matrix_audit_elapsed_s':a['elapsed_s'],**r['phase_s']})
        e = next(e for e in manifest['inputs'] if e['seed']==seed)
        assert sha(ROOT/e['completion_path'])==e['completion_sha256'] and sha(ROOT/e['clues_path'])==e['clues_sha256']
        sol = np.load(ROOT/e['completion_path'])['solutions']; completion[seed] = sol
        original = np.load(ROOT/e['clues_path']); k = original['given'].sum(1)
        first = np.load(d/'first_clues.npz'); firstmodel = np.load(d/'first_model.npz')
        assert np.array_equal(first['given'].sum(1),k) and np.array_equal(first['source_row'],original['source_row'])
        assert np.array_equal(first['values'][first['given']],sol[first['given']])
        oldrow = next(e for e in old['rows'] if e['seed']==seed and e['n']==300)
        assert sha(ROOT/oldrow['path'])==oldrow['sha256']
        oldsupport = np.load(ROOT/oldrow['path'])['support']
        supports['official',seed] = oldsupport; supports['uniform',seed] = firstmodel['support']
        official.append(int((oldsupport&~peer).sum()))
        assert not np.any(peer&~oldsupport)
        dr = np.load(d/'draws.npz'); packed = np.unpackbits(dr['packed_supports'],axis=2).astype(bool)
        assert packed.shape==(1024,3,3240) and packed[:,:,peer].all()
        extras = (packed&~peer).sum(2)
        assert np.array_equal(extras,dr['retained_edges']-810)
        assert np.array_equal(firstmodel['support'],packed[0,0])
        counts.append(extras)
        # Geometry enters only this post-fit audit. Two algebraically distinct evaluations.
        equal = sol[:,i[~peer]] == sol[:,j[~peer]]
        q = k*(k-1)/(81*80)
        for z,n in enumerate(p['sizes']):
            factors = np.where(equal[:n],1-q[:n,None],1.)
            direct = np.prod(factors,axis=0)
            logform = np.exp((equal[:n]*np.log1p(-q[:n,None])).sum(0))
            assert np.allclose(direct,logform,rtol=1e-10,atol=1e-20)
            mu = float(direct.sum()); values = extras[:,z]
            se = float(values.std(ddof=1)/np.sqrt(1024))
            expected_rows.append({'seed':seed,'n':n,'exact_conditional_expected_extras':mu,
                'mc_mean_extras':float(values.mean()),'mc_standard_error':se,
                'mc_mean_minus_expectation':float(values.mean()-mu),
                'mc_z_descriptive':float((values.mean()-mu)/se) if se else None,
                'exact_recoveries':int((values==0).sum()),'draws':1024,
                'union_failure_bound':min(1.,mu)})
            for draw,value in enumerate(values): draws_rows.append({'seed':seed,'draw':draw,'n':n,'extra_edges':int(value)})
        first_rows.append({'seed':seed,'draw':0,'n':300,'official_extra_edges':official[-1],
            'uniform_extra_edges':int(extras[0,0]),'official_support_sha256':hashlib.sha256(oldsupport.tobytes()).hexdigest(),
            'uniform_support_sha256':hashlib.sha256(firstmodel['support'].tobytes()).hexdigest()})
        hashes[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
        hashes[str((d/'audit_counts.json').relative_to(ROOT))]=sha(d/'audit_counts.json')
    counts = np.array(counts); simulated = counts[:,:,0].sum(0); observed = sum(official)
    lower_or_equal = int((simulated<=observed).sum()); pv = (1+lower_or_equal)/1025
    primary = {'hypothesis':'Official masks yield unusually fewer N300 extra edges than uniform count-matched masks.',
        'official_extra_edges_per_seed':official,'official_total':observed,'simulated_joint_draws':1024,
        'simulated_mean_total':float(simulated.mean()),'simulated_min_total':int(simulated.min()),
        'simulated_max_total':int(simulated.max()),'simulated_quantiles_025_50_975':list(np.quantile(simulated,[.025,.5,.975])),
        'simulated_le_official':lower_or_equal,'one_sided_finite_simulation_p':pv,'threshold':.01,
        'decision':'supported_unusually_fewer_extras' if pv<=.01 else 'not_supported_in_prespecified_direction'}
    datafile = ROOT/'upstream/data/full-test/full-all.npz'; data = np.load(datafile)
    lo,hi = p['rows_half_open']; target = data['solutions'][lo:hi].astype(np.int8)-1
    free = ~data['clue_mask'][lo:hi].astype(bool); testbytes = np.ascontiguousarray(target).view(np.dtype((np.void,81))).ravel()
    overlaps = [{'seed':s,'shared_complete_boards':int(np.isin(np.ascontiguousarray(v).view(np.dtype((np.void,81))).ravel(),testbytes).sum())} for s,v in completion.items()]
    queue = json.loads((OUT/'queue.json').read_text()); mapping = json.loads((OUT/'search_mapping.json').read_text())
    assert len(queue)==mapping['unique_searches']==12 and len(mapping['rows'])==12
    assert all(j['status']=='complete' for j in queue) and mapping['test_data_sha256']==sha(datafile)
    rows = []; arrays = {}; oracle = []
    for job in queue:
        d = OUT/'jobs'/job['job_id']; r = json.loads((d/'receipt.json').read_text())
        assert r['protocol_sha256']==sha(PROTOCOL) and r['wrapper_sha256']==sha(HERE/'mask_search.py')
        assert r['backend_sha256']==p['backend_sha256'] and r['capture_sha256']==p['capture_sha256']
        assert r['rows_half_open']==[120000,130000] and r['test_rows']==10000
        assert r['used_unit']==1 and r['tau']==r['ebar']==0 and r['move_cap']==262144
        assert r['preflight_state_comparisons']==125 and r['hidden_target_state_comparisons']==110
        assert r['search_kwargs']=={'temp':.4,'theta':.45,'focused':True,'C':4,'tau':0.,'tol':.001,'ebar':0.,'cap':4e11,'check':4096,'seed':r['seed'],'anneal':[.4,.31,524288]}
        for name,h in r['output_sha256'].items(): assert sha(d/name)==h
        support = supports[r['arm'],r['seed']]
        assert r['support_bytes_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
        model = torch.load(d/'searched_model.pt',weights_only=True,map_location='cpu')
        assert torch.equal(model['W'],torch.from_numpy(support.astype(np.float32))[:,None,None]*torch.eye(9)) and not model['B'].any()
        z = np.load(d/'boards.npz'); b = z['boards']; st = z['stopped']; moves = z['moves']
        assert np.array_equal(z['solutions'],target) and np.array_equal(z['free'],free) and np.array_equal(z['puzzle'],np.arange(lo,hi))
        exact = (b==target).all(1); valid = validity(b,target,free)
        assert np.array_equal(exact,valid) and np.array_equal(exact,z['exact']) and np.array_equal(valid,z['valid'])
        assert int(exact.sum())==r['solved'] and int((~st).sum())==r['budget_failures']
        assert int((st&~exact).sum())==r['wrong_stops']==0
        assert np.all((moves>0)&(moves<=262144))
        per = json.loads((d/'result.json').read_text())['per_puzzle']
        assert np.array_equal(per['solved'],exact) and np.array_equal(per['stopped'],st) and np.array_equal(per['flops'],moves*r['fl_move']*4)
        energy = (target[:,i[support]]==target[:,j[support]]).sum(1)
        assert np.all(~st|(energy==0))
        oracle.append({'seed':r['seed'],'arm':r['arm'],'zero_energy_targets':int((energy==0).sum()),
            'mean_target_energy':float(energy.mean()),'correct_certified_stops':int((st&exact).sum()),
            'correct_uncertified_returns':int((~st&exact).sum())})
        arrays[r['arm'],r['seed']]=exact
        rows.append({k:r[k] for k in ['job_id','n','seed','arm','solved','wrong_stops','budget_failures','mean_moves','wall_s','edges','D']})
        hashes[str((d/'receipt.json').relative_to(ROOT))]=sha(d/'receipt.json')
    grouped = []
    for arm in p['arms']:
        solved = [int(arrays[arm,s].sum()) for s in p['seeds']]
        grouped.append({'arm':arm,'solved_per_seed':solved,'mean_accuracy_pct':float(np.mean(solved)/100),'wrong_stops':0,'seeds':6,'n_per_seed':10000})
    diff = np.array([arrays['uniform',s].astype(float)-arrays['official',s] for s in p['seeds']])
    rng = np.random.default_rng(20261121)
    boot = [100*diff[rng.integers(0,6,6)][:,rng.integers(0,10000,10000)].mean() for _ in range(5000)]
    contrast = {'contrast':'uniform minus official','effect_pp':float(100*diff.mean()),
        'descriptive_ci95_pp':[float(x) for x in np.quantile(boot,[.025,.975])],'scope':'Secondary prespecified draw0, not selected from the 1024 Monte Carlo draws.'}
    report = {'status':'validated','utc':now(),'primary':primary,'conditional_expectations':expected_rows,
        'first_stream_structure':first_rows,'grouped':grouped,'secondary_search_contrast':contrast,
        'rows':rows,'oracle_diagnostic':oracle,'saved_boards_verified':120000,
        'randomization_draws_per_seed':1024,'independently_audited_supports':18432,
        'count_audits':audits,'timings':timings,'search_wall_sum_s':sum(r['wall_s'] for r in rows),
        'training_test_completeboard_overlap':overlaps,'artifact_sha256':hashes,
        'protocol_sha256':sha(PROTOCOL),'audit_sha256':sha(__file__),
        'independent_board_audit_helper_sha256':sha(HERE/'audit_clues.py'),
        'theorem_sha256':sha(HERE/'RANDOM_MASK_COVERAGE_THEOREM.md'),
        'test_data_sha256':sha(datafile),'limitations':p['limitations']+[
            'The 1024 streams are deterministic pseudorandom realizations. The theorem describes ideal independent uniform masks.',
            'Failure to reject the one-sided hypothesis does not establish mask equivalence or validate a post-hoc opposite-direction test.',
            'A correct unstopped best board may have positive model energy. Oracle E0 fractions bound certified correct stops only.',
            'Generation and audit timings are operational observations of different algorithms, not paired speed benchmarks.']}
    save(OUT/'audit.json',report); save(PUBLIC/'masks.json',report)
    for name,rs in [('masks_draws.csv',draws_rows),('masks_expectations.csv',expected_rows),('masks_first_stream.csv',first_rows),('masks_per_seed.csv',rows),('masks_oracle.csv',oracle)]:
        with (PUBLIC/name).open('w',newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    summary = json.loads((PUBLIC/'summary.json').read_text()); summary['uniform_mask_randomization']=report; save(PUBLIC/'summary.json',summary)
    lines = ['# H1q: conditional randomization of training clue masks','',
        f'The registered hypothesis that official masks leave unusually fewer false extra edges is **{primary["decision"]}**. The six official N300 banks retain {observed} extra edges in total; the 1,024 count-matched uniform-mask draws average {simulated.mean():.3f}. The one-sided finite-simulation p-value is **{pv:.6f}** at the frozen 0.01 threshold.','',
        'Each complete training board and its original number of given cells is held fixed. Only the chosen cells are randomized, independently across rows. This is conditional mask randomization, not uniform sampling of Sudoku solutions. All 18,432 prefix supports were independently checked using exact integer-valued matrix counts; the full mask RNG sequence was replayed and source clue counts preserved.','',
        '| Mask arm, N300 | Mean solved (%) | Wrong stops |','|---|---:|---:|',
        '| BP AdamW: outside this count mechanism study; not run | — | — |']
    for r in grouped: lines.append(f'| {r["arm"]} | {r["mean_accuracy_pct"]:.4f} | 0 |')
    lines += ['',f'The preselected draw0 search gives uniform minus official {contrast["effect_pp"]:.4f} percentage points, with a descriptive 95% paired bootstrap interval {contrast["descriptive_ci95_pp"]}. These are six seeds on fresh full-all rows [120000,130000), four chains and 262,144 moves per chain. No mask stream was selected using graph or accuracy outcomes. All 120,000 returned boards were independently checked.','',
        'For each fixed bank, the exact expected number of surviving nonpeers is the sum of products of unexposed-pair probabilities. The conditional expectation was evaluated independently by direct multiplication and log accumulation. The formula and its union bound require independence between row masks, not between cell pairs. Per-bank N1000/N3000 results and Monte Carlo deviations from exact expectation are descriptive.','',
        f'At N1000, {sum(r["exact_recoveries"] for r in expected_rows if r["n"]==1000):,} of 6,144 randomized banks recovered the exact graph. At N3000, all 6,144 recovered it. The conditional theorem gives a numerical failure bound of {sum(r["exact_conditional_expected_extras"] for r in expected_rows if r["n"]==3000):.4g} for any of the six fixed banks to fail under one fresh ideal uniform-mask draw at N3000. This tiny bound comes from the theorem, not from estimating a rare failure probability using zero observed failures.','',
        'Counting used only the 324 equal-digit pairs of each complete board, always gated by observing both cells. An independent masked-feature matrix implementation validated every saved support; dense clue-only recounts checked the first and last draw in every bank at all three prefixes, and hidden-value substitution gates passed. Geometry enters the post-fit audit only.','',
        'The one-sided test was frozen before outcomes. A nonsignificant result in its stated direction must not be presented as equality or as a confirmatory opposite-direction effect. Uniform masks may define non-unique training puzzles; the held-out test puzzles are unchanged.','',
        '[Protocol](protocol-masks.json) · [Mask sampler](mask_count.py) · [Independent count audit](audit_mask_counts.py) · [Search](mask_search.py) · [Final audit](audit_masks.py) · [Conditional theorem](RANDOM_MASK_COVERAGE_THEOREM.md) · [Public results](../../../results/research/ripper/masks.json)','']
    (HERE/'REPORT-masks.md').write_text('\n'.join(lines))
    print(json.dumps({'primary':primary,'grouped':grouped,'secondary':contrast,'expectations':expected_rows}))


if __name__ == '__main__': main()
