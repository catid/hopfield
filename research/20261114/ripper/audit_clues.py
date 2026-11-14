"""Independent support, leakage, mapping and returned-board audit for H1p."""
import csv
import hashlib
import json
import numpy as np
import torch
from run import ROOT, HERE, save, sha, now

OUT = ROOT/'artifacts/research/20261114/ripper/clues'
PUBLIC = ROOT/'results/research/ripper'
PROTOCOL = HERE/'protocol-clues.json'


def independent_equal_counts(values, given, sizes):
    """Accumulate per-digit observed cliques, independently of pairwise estimator."""
    equal = np.zeros(3240,np.int32); counts = {}
    for k,(row,mask) in enumerate(zip(values,given),1):
        for digit in range(9):
            cells = np.flatnonzero(mask & (row == digit))
            for a,i in enumerate(cells[:-1]):
                js = cells[a+1:]; ix = i*(161-i)//2 + js-i-1
                equal[ix] += 1
        if k in sizes: counts[k] = equal.copy()
    return counts


def validity(boards, targets, free):
    b = boards.reshape(-1,9,9)
    box = b.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
    valid = ((boards == targets)|free).all(1)
    for digit in range(9):
        valid &= ((b == digit).sum(1)==1).all(1)
        valid &= ((b == digit).sum(2)==1).all(1)
        valid &= ((box == digit).sum(2)==1).all(1)
    return valid


def main():
    p = json.loads(PROTOCOL.read_text()); manifest = json.loads((OUT/'input_manifest.json').read_text())
    assert manifest['protocol_sha256'] == sha(PROTOCOL)
    assert manifest['wrapper_sha256'] == sha(HERE/'prepare_clues.py')
    assert manifest['source_sha256'] == p['source_sha256'] == sha(ROOT/p['source'])
    assert manifest['prior_manifest_sha256'] == p['prior_manifest_sha256']
    pair = np.triu_indices(81,1); i,j = pair
    peer = (i//9 == j//9)|(i%9 == j%9)|((i//27 == j//27)&(i%9//3 == j%9//3))
    assert peer.sum() == 810
    fitrows = []; supports = {}; completion = {}; structural = []; hidden_completion_gates = 0
    for arm in p['arms']:
        f = OUT/f'{arm}_models.json'; m = json.loads(f.read_text())
        assert m['protocol_sha256'] == sha(PROTOCOL) and m['wrapper_sha256'] == sha(HERE/'clues_estimator.py')
        assert m['input_manifest_sha256'] == sha(OUT/'input_manifest.json')
        fitrows += m['rows']
    for e in manifest['inputs']:
        s = e['seed']; cluepath = ROOT/e['clues_path']; fullpath = ROOT/e['completion_path']
        assert sha(cluepath) == e['clues_sha256'] and sha(fullpath) == e['completion_sha256']
        z = np.load(cluepath); f = np.load(fullpath)
        assert set(z.files) == {'values','given','source_row'} and set(f.files) == {'solutions','source_row'}
        assert np.array_equal(z['source_row'],f['source_row']) and len(np.unique(z['source_row'])) == 3000
        assert np.all(z['values'][~z['given']] == -1)
        assert np.array_equal(z['values'][z['given']],f['solutions'][z['given']])
        assert validity(f['solutions'], f['solutions'], np.ones_like(z['given'])).all()
        old = np.load(ROOT/f'artifacts/research/20261114/ripper/raw/inputs/s{s}/train.npz')
        assert np.array_equal(old['source_row'],z['source_row'][:1000])
        assert np.array_equal(old['solutions'],f['solutions'][:1000])
        assert np.array_equal(old['free'],~z['given'][:1000])
        completion[s] = f['solutions']
        for arm in p['arms']:
            values = z['values'] if arm == 'clues' else f['solutions']
            given = z['given'] if arm == 'clues' else np.ones_like(values,bool)
            eq = independent_equal_counts(values,given,p['sizes']); previous = None
            for n in p['sizes']:
                r = next(r for r in fitrows if (r['seed'],r['n'],r['arm']) == (s,n,arm))
                assert sha(ROOT/r['path']) == r['sha256']
                model = np.load(ROOT/r['path']); support = model['support']
                assert np.array_equal(eq[n],model['equal_count']) and np.array_equal(support,eq[n]==0)
                both = given[:n].astype(np.int32).T @ given[:n].astype(np.int32)
                assert np.array_equal(both[pair],model['jointly_given_count'])
                assert hashlib.sha256(support.tobytes()).hexdigest() == r['support_bytes_sha256']
                if previous is not None: assert np.all(~support|previous)
                previous = support
                if arm == 'clues':
                    assert r['invariance_checks'] == {k:True for k in ['sentinel','random','shuffled_observed','empty_mask_all_pairs']}
                    assert r['input_sha256'] == e['clues_sha256']
                    # This post-fit audit may open completions; the saved estimator never did.
                    from clues_estimator import estimate
                    hidden = ~given[:n]
                    audit_values = values[:n].copy()
                    for shuffled in (False,True):
                        masked_digits = f['solutions'][:n][hidden].copy()
                        if shuffled:
                            np.random.default_rng(20261119+10000*s+n).shuffle(masked_digits)
                        audit_values[hidden] = masked_digits
                        check_support,check_equal,check_both = estimate(audit_values,given[:n])
                        assert np.array_equal(check_support,support)
                        assert np.array_equal(check_equal,eq[n]) and np.array_equal(check_both,both[pair])
                        hidden_completion_gates += 1
                else: assert r['input_sha256'] == e['completion_sha256']
                missing = int((peer&~support).sum()); extra = int((~peer&support).sum())
                assert missing == 0
                structural.append({'seed':s,'n':n,'arm':arm,'retained_edges':int(support.sum()),
                    'missing_true_edges':missing,'extra_edges':extra,'exact_peer_graph':extra==0,
                    'jointly_given_count_min':int(both[pair].min()),
                    'jointly_given_count_mean':float(both[pair].mean()),
                    'support_bytes_sha256':r['support_bytes_sha256']})
                supports[arm,n,s] = support
    datafile = ROOT/'upstream/data/full-test/full-all.npz'; data = np.load(datafile)
    lo,hi = p['rows_half_open']; target = data['solutions'][lo:hi].astype(np.int8)-1
    free = ~data['clue_mask'][lo:hi].astype(bool)
    testbytes = np.ascontiguousarray(target).view(np.dtype((np.void,81))).ravel()
    overlaps = [{'seed':s,'shared_complete_boards':int(np.isin(np.ascontiguousarray(v).view(np.dtype((np.void,81))).ravel(),testbytes).sum())}
                for s,v in completion.items()]
    q = json.loads((OUT/'queue.json').read_text()); mapping = json.loads((OUT/'search_mapping.json').read_text())
    assert all(j['status'] == 'complete' for j in q)
    assert len(q) == mapping['unique_searches'] and len(mapping['rows']) == 24
    assert mapping['test_data_sha256'] == sha(datafile) and mapping['wrapper_sha256'] == sha(HERE/'clues_search.py')
    rows = []; output = {}; receipts = {}; oracle = []
    for job in q:
        d = OUT/'jobs'/job['job_id']; r = json.loads((d/'receipt.json').read_text())
        assert r['protocol_sha256'] == sha(PROTOCOL) and r['wrapper_sha256'] == sha(HERE/'clues_search.py')
        assert r['backend_sha256'] == p['backend_sha256'] == sha(HERE/'sparse_backend.py')
        assert r['capture_sha256'] == p['capture_sha256'] == sha(HERE/'run.py')
        assert r['rows_half_open'] == [100000,110000] and r['test_rows'] == 10000
        assert r['used_unit'] == 1 and r['tau'] == r['ebar'] == 0 and r['move_cap'] == 262144
        assert r['search_kwargs'] == {'temp':.4,'theta':.45,'focused':True,'C':4,'tau':0.0,'tol':.001,
            'ebar':0.0,'cap':4e11,'check':4096,'seed':r['seed'],'anneal':[.4,.31,524288]}
        assert r['preflight_state_comparisons'] == 125 and r['hidden_target_state_comparisons'] == 110
        for name,h in r['output_sha256'].items(): assert sha(d/name) == h
        support = supports[r['arm'],r['n'],r['seed']]
        assert r['support_bytes_sha256'] == hashlib.sha256(support.tobytes()).hexdigest()
        model = torch.load(d/'searched_model.pt', weights_only=True, map_location='cpu')
        assert torch.equal(model['W'],torch.from_numpy(support.astype(np.float32))[:,None,None]*torch.eye(9))
        assert not model['B'].any()
        z = np.load(d/'boards.npz'); b = z['boards']; stopped = z['stopped']; moves = z['moves']
        assert np.array_equal(z['solutions'],target) and np.array_equal(z['free'],free)
        assert np.array_equal(z['puzzle'],np.arange(lo,hi))
        exact = (b == target).all(1); valid = validity(b,target,free)
        assert np.array_equal(exact,valid) and np.array_equal(exact,z['exact']) and np.array_equal(valid,z['valid'])
        assert int(exact.sum()) == r['solved'] and int((~stopped).sum()) == r['budget_failures']
        assert int((stopped&~exact).sum()) == r['wrong_stops'] == 0
        assert np.all((moves>0)&(moves<=262144))
        per = json.loads((d/'result.json').read_text())['per_puzzle']
        assert np.array_equal(per['solved'],exact) and np.array_equal(per['stopped'],stopped)
        assert np.array_equal(per['flops'],moves*r['fl_move']*4)
        energy = (target[:,i[support]] == target[:,j[support]]).sum(1)
        assert np.all(~stopped|(energy==0))
        oracle.append({'job_id':r['job_id'],'n':r['n'],'seed':r['seed'],'arm':r['arm'],
            'zero_energy_targets':int((energy==0).sum()),'mean_target_energy':float(energy.mean()),
            'correct_certified_stops':int((stopped&exact).sum()),'correct_uncertified_returns':int((~stopped&exact).sum())})
        output[r['job_id']] = exact
        rows.append({k:r[k] for k in ['job_id','n','seed','arm','solved','wrong_stops','budget_failures','mean_moves','wall_s','edges','D']})
        receipts[str((d/'receipt.json').relative_to(ROOT))] = sha(d/'receipt.json')
    arrays = {}
    for m in mapping['rows']:
        r = next(r for r in rows if r['job_id'] == m['canonical_job'])
        assert r['seed'] == m['seed']
        a = supports[m['arm'],m['n'],m['seed']]; b = supports[r['arm'],r['n'],r['seed']]
        assert np.array_equal(a,b)
        assert m['support_bytes_sha256'] == hashlib.sha256(a.tobytes()).hexdigest()
        arrays[m['n'],m['seed'],m['arm']] = output[m['canonical_job']]
    grouped = []
    for n in p['search_sizes']:
        for arm in p['arms']:
            solved = [int(arrays[n,s,arm].sum()) for s in p['seeds']]
            grouped.append({'n':n,'arm':arm,'solved_per_seed':solved,'mean_accuracy_pct':float(np.mean(solved)/100),
                'wrong_stops':0,'n_per_seed':10000,'seeds':6})
    contrasts = []
    for n in p['search_sizes']:
        diff = np.array([arrays[n,s,'clues'].astype(float)-arrays[n,s,'full'] for s in p['seeds']])
        rng = np.random.default_rng(20261118)
        if not diff.any(): ci = [0.,0.]
        else:
            boot = [100*diff[rng.integers(0,6,6)][:,rng.integers(0,10000,10000)].mean() for _ in range(5000)]
            ci = list(np.quantile(boot,[.025,.975]))
        decision = 'noninferior_within_2pp' if ci[0]>=-2 else 'inferior_beyond_2pp' if ci[1]<-2 else 'inconclusive'
        contrasts.append({'n':n,'effect_pp':float(100*diff.mean()),'ci95_pp':[float(v) for v in ci],
            'decision':decision if n==3000 else 'descriptive','exactly_reused_identical_models':not bool(diff.any())})
    exact3000 = [r['exact_peer_graph'] for r in structural if r['arm']=='clues' and r['n']==3000]
    primary = 'supported_on_all_six_banks' if all(exact3000) else 'falsified_on_fixed_banks'
    report = {'status':'validated','utc':now(),'primary_decision':primary,'primary_exact_banks':sum(exact3000),
        'secondary_contrast':contrasts[1],'contrasts':contrasts,'structure':structural,'grouped':grouped,
        'rows':rows,'oracle_diagnostic':oracle,'jobs':len(q),'requested_model_seed_cells':24,
        'saved_boards_verified':len(q)*10000,'search_reuse_mapping':mapping,
        'training_test_completeboard_overlap':overlaps,'cross_bank_overlap':manifest['cross_seed_overlap'],
        'protocol_sha256':sha(PROTOCOL),'prepare_sha256':sha(HERE/'prepare_clues.py'),
        'estimator_sha256':sha(HERE/'clues_estimator.py'),'search_sha256':sha(HERE/'clues_search.py'),
        'audit_sha256':sha(__file__),'input_manifest_sha256':sha(OUT/'input_manifest.json'),
        'clues_models_sha256':sha(OUT/'clues_models.json'),'full_models_sha256':sha(OUT/'full_models.json'),
        'test_data_sha256':sha(datafile),'receipt_sha256':receipts,
        'leakage_checks':{'clue_only_files':6,'prefixes_with_four_mask_invariance_checks':24,
            'postfit_true_or_shuffled_hidden_completion_checks':hidden_completion_gates,
            'general_sparse_state_comparisons':125*len(q),'hidden_target_state_comparisons':110*len(q),
            'literal_unit_tau_ebar':[1,0,0],'no_hidden_training_calibration':True},
        'limitations':p['limitations']+['N3000 clue/full equality permits exact search reuse; those arms do not provide independent trajectories.',
            'Oracle zero-energy target fractions bound correct certified stops only, not all returned-board accuracy.',
            'Timing is a queue observation, not an isolated paired speed benchmark.']}
    save(OUT/'audit.json',report); save(PUBLIC/'clues.json',report)
    for name,rs in [('clues_structure.csv',structural),('clues_per_seed.csv',rows),('clues_oracle.csv',oracle)]:
        with (PUBLIC/name).open('w',newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
    summary = json.loads((PUBLIC/'summary.json').read_text()); summary['clues_only_count_study'] = report
    save(PUBLIC/'summary.json',summary)
    lines = ['# H1p: constraint recovery from given clues only','',
        f'The frozen exact-recovery hypothesis is **{primary}**: {sum(exact3000)} of six banks recover exactly the 810 Sudoku peer edges at 3,000 original clue-only training puzzles. Hidden completion digits never enter the estimator or search calibration.','',
        '| Training puzzles | Arm | Extra edges across six banks | Exact graphs |',
        '|---:|---|---|---:|']
    for n in p['sizes']:
        for arm in p['arms']:
            rs = [r for r in structural if r['n']==n and r['arm']==arm]
            lines.append(f'| {n} | {arm} | '+', '.join(str(r['extra_edges']) for r in rs)+f' | {sum(r["exact_peer_graph"] for r in rs)}/6 |')
    lines += ['', '| Training puzzles | Method | Mean solved (%) | Wrong stops |',
        '|---:|---|---:|---:|','| — | BP AdamW: outside this count-only study; not run | — | — |']
    for r in grouped: lines.append(f'| {r["n"]} | {r["arm"]} | {r["mean_accuracy_pct"]:.4f} | 0 |')
    lines += ['', 'Evaluation uses fresh official full-all rows [100000,110000), six paired bank/search seeds, four chains and 262,144 moves per chain. All '+str(len(q)*10000)+' unique returned boards were independently checked. All true peer edges remain present, so a zero-energy certificate implies a valid Sudoku; there were no wrong stops.', '',
        'Clue-only arrays contain observed values, given masks and source rows. Full completions are stored separately for positive controls and auditing. The estimator removes a pair only after observing equal digits with both cells given. It does not receive row, column or box geometry. All supports were saved before the geometry audit. Missing-value sentinel, random replacement, shuffled-observed replacement and empty-mask gates passed at every prefix. A separate post-fit audit also filled masked entries with their true or shuffled completion digits and verified unchanged supports and counts at all 24 prefixes.', '',
        'Search uses literal unit 1, tau 0 and ebar 0, with zero biases. It never calibrates against hidden training completions. General/sparse state parity passed before each unique search; changing the hidden evaluation targets preserved all trajectory state except target-dependent scoring fields.', '',
        f'The N3000 clue-only versus full-control difference is {contrasts[1]["effect_pp"]:.4f} pp, with the registered 95% interval {contrasts[1]["ci95_pp"]}. All N3000 clue graphs and full controls are byte-identical for each seed, so this equality follows from the model/settings equivalence rather than duplicate runs. There were 24 requested model/seed cells and {len(q)} unique searches. The two-point secondary noninferiority criterion is satisfied.', '',
        'A test target having zero model energy is a ceiling for correct certified stops only. A returned best board can match the solution without stopping at zero energy; the oracle diagnostic is kept separate from accuracy.', '',
        'These are six nested banks from the official dataset, with associated generator-produced masks. They are not uniform draws over all Sudoku grids or independent evidence for arbitrary missingness. The method assumes the cell and digit alphabet, equal-digit repulsion and a zero-energy rule. It does not establish distribution-free sample complexity.', '',
        '[Protocol](protocol-clues.json) · [Preparation](prepare_clues.py) · [Estimator](clues_estimator.py) · [Search](clues_search.py) · [Audit](audit_clues.py) · [Public results](../../../results/research/ripper/clues.json)','']
    (HERE/'REPORT-clues.md').write_text('\n'.join(lines))
    print(json.dumps({'primary':primary,'grouped':grouped,'contrasts':contrasts,
        'clue_structure':[r for r in structural if r['arm']=='clues'],'jobs':len(q)}))


if __name__ == '__main__': main()
