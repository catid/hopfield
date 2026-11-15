"""Independent H1bu source, nested masks, noise, LP and stopping-rule audit."""
from concurrent.futures import ProcessPoolExecutor
import csv, hashlib, json, time
import numpy as np
from stability_stopping_study import ROOT, HERE, OUT, PUBLIC, REGISTRY, PROTOCOL, PARENT, PREFIXES, sha, save, npz, verify
from audit_extra_coverage import statistics
from audit_partial_degree import grid
from audit_laplace_rate import independent_rate_costs
from audit_balanced_mask_math import reference_lines
from audit_higher_noise import reference_observe, realized_noise


def identity_audit(public=False):
    p=verify();base=PUBLIC if public else OUT;register=PUBLIC if public else REGISTRY
    parent=npz(PARENT/'excluded_ids.npz');fresh=npz(PARENT/'selected_sources.npz')['all_source_row'];excluded=np.unique(np.concatenate([parent['excluded'],fresh]));ripper=np.unique(np.concatenate([parent['ripper_prior'],fresh]))
    data=npz(register/'excluded_ids.npz');assert np.array_equal(excluded,data['excluded']) and len(excluded)==2368426 and np.array_equal(ripper,data['ripper_prior']) and len(ripper)==2132001 and np.isin(ripper,excluded).all()
    ledger=json.loads((register/'ledger.json').read_text())
    assert ledger['parent_protocol_sha256']==sha(HERE/'protocol-higher-noise.json') and ledger['parent_source_audit_sha256']==sha(PARENT/'source-audit.json') and ledger['parent_selected_ids_sha256']==sha(PARENT/'selected_sources.npz') and ledger['parent_exclusions_sha256']==sha(PARENT/'excluded_ids.npz') and ledger['excluded_file_sha256']==sha(register/'excluded_ids.npz')
    selected=npz(base/'selected_sources.npz');draws=selected['source_rows'];seal=json.loads((base/'identity-seal.json').read_text());manifest=json.loads((base/'input_manifest.json').read_text());eligible=np.setdiff1d(np.arange(p['pool_size'],dtype=np.int64),excluded,assume_unique=True)
    assert len(eligible)==1463568 and draws.shape==(32,360) and draws.dtype==np.int64
    assert seal['status']=='sealed_before_selected_labels' and seal['protocol_sha256']==manifest['protocol_sha256']==sha(PROTOCOL) and seal['selected_sources_sha256']==manifest['selected_sources_sha256']==sha(base/'selected_sources.npz')
    assert manifest['identity_seal_sha256']==sha(base/'identity-seal.json') and manifest['all32_full_masks_sealed_before_materialization'] and len(manifest['predraw_receipts_sha256'])==32
    assert [e['seed'] for e in manifest['inputs']]==list(range(32))
    for seed in range(32):
        expected=np.random.default_rng(p['source_seed_base']+seed).choice(eligible,360,replace=False);assert np.array_equal(expected,draws[seed]) and len(np.unique(draws[seed]))==360
        d=base/f's{seed}';predraw=json.loads((d/'predraw-receipt.json').read_text());assert predraw['seed']==seed and predraw['protocol_sha256']==sha(PROTOCOL) and predraw['identity_seal_sha256']==sha(base/'identity-seal.json') and predraw['masks_sha256']==sha(d/'masks.npz')
        assert predraw['status']=='sealed_before_selected_labels' and manifest['predraw_receipts_sha256'][str(seed)]==sha(d/'predraw-receipt.json')
    unique=np.unique(draws);assert np.array_equal(unique,selected['all_source_row']) and not np.intersect1d(unique,excluded).size
    assert len(unique)==seal['new_unique_source_ids'] and draws.size-len(unique)==seal['crossbank_repeated_instances']
    assert json.loads((base/'all_source_manifest.json').read_text())['inputs']==manifest['inputs']
    return dict(new_unique_source_ids=len(unique),source_instances=int(draws.size),crossbank_repeated_instances=int(draws.size-len(unique)),within_bank_repeats=0,declared_exclusion_overlap=0,
                updated_ripper_ledger_union=2132001+len(unique),updated_combined_exclusion_union=2368426+len(unique),clean_unique_oracle_cells=81*len(unique),clean_materialized_cell_instances=933120),draws,manifest


def sources():
    p=verify(True);identity,draws,manifest=identity_audit();wanted={};hashes={}
    for entry in manifest['inputs']:
        seed=entry['seed'];path=ROOT/entry['completion_path'];assert entry['completion_sha256']==sha(path);hashes[entry['completion_path']]=sha(path);z=npz(path)
        assert set(z)=={'solutions','source_row'} and np.array_equal(z['source_row'],draws[seed]);boards=z['solutions'];assert boards.shape==(360,81) and boards.dtype==np.int8 and ((boards>=0)&(boards<9)).all()
        grids=boards.reshape(-1,9,9);boxes=grids.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
        for digit in range(9):assert ((grids==digit).sum(1)==1).all() and ((grids==digit).sum(2)==1).all() and ((boxes==digit).sum(2)==1).all()
        for index,board in zip(draws[seed],boards,strict=True):
            index=int(index)
            if index in wanted:assert np.array_equal(wanted[index],board)
            else:wanted[index]=board
    source=ROOT/p['source'];assert sha(source)==p['source_csv_sha256'];remaining=set(wanted)
    with source.open(newline='') as f:
        reader=csv.reader(f);column=next(reader).index('answer')
        for index,fields in enumerate(reader):
            if index not in remaining:continue
            board=np.array([int(c)-1 for c in fields[column]],np.int8);assert np.array_equal(board,wanted[index]);remaining.remove(index)
            if not remaining:break
    assert not remaining
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),input_manifest_sha256=sha(OUT/'input_manifest.json'),identity_seal_sha256=sha(OUT/'identity-seal.json'),all_source_manifest_sha256=sha(OUT/'all_source_manifest.json'),
                **identity,independent_csv_answers_verified=len(wanted),new_donor_queries=0,all32_full_masks_sealed_before_selected_labels=True,completion_sha256=hashes,scope=p['scope'])
    save(OUT/'source-audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='completion_sha256'}),flush=True)


def reference_masks(seed):
    rng = np.random.default_rng(seed)
    lines = reference_lines()
    mask = np.ones((360, 81), bool)
    for cycle in range(4):
        permutation = list(map(int, rng.permutation(81)))
        order = list(map(int, rng.permutation(90)))
        for offset, line in enumerate(order):
            for cell in lines[line]:
                mask[90*cycle+offset, permutation[cell]] = False
    return mask


def reference_stopping(graphs):
    assert len(graphs) == 4
    available = [graph is not None for graph in graphs]
    equal = []
    for earlier, later in zip(graphs[:-1], graphs[1:]):
        equal.append(None if earlier is None or later is None else all(int(a) == int(b) for a, b in zip(earlier, later, strict=True)))
    chosen = 3
    for index in [1, 2]:
        if available[index-1] and available[index] and equal[index-1]:
            chosen = index
            break
    return dict(stop_index=chosen, stop_rows=[90, 180, 270, 360][chosen], available_prefixes=available, consecutive_equal=equal)


def literal_account(mask, stop_rows):
    assert stop_rows in [180, 270, 360]
    sets = {}
    for n in [90, 180, 270, 360]:
        sets[n] = {(row, int(cell)) for row in range(n) for cell in np.flatnonzero(mask[row])}
        assert len(sets[n]) == n*72
    assert sets[90] < sets[180] < sets[270] < sets[360]
    union = set().union(*sets.values())
    assert union == sets[360]
    return dict(full_physical_noisy_union=len(union), virtual_stopped_queries=len(sets[stop_rows]), prefix_queries={str(n): len(sets[n]) for n in sets})


def complete_gate(mask, clean, noise, expected):
    from stability_stopping_study import observe
    changed = ((clean+(np.arange(81) % 8+1)[None, :]) % 9).astype(np.int8)
    assert (changed != clean).all()
    gates = []
    for n in PREFIXES:
        actual = observe(mask[:n], changed[:n], {k: v[:n] for k, v in noise.items()})
        old = expected[n]
        given = old['given']
        assert np.array_equal(actual['given'], given) and np.array_equal(actual['joint_count'], old['joint_count'])
        assert (actual['values'][given] != old['values'][given]).all() and not np.array_equal(actual['equal_count'], old['equal_count'])
        gates.append(dict(rows=n, status='passed', all_source_and_queried_values_changed=True, masks_and_joint_counts_unchanged=True,
                          equality_counts_changed=True, original_source_sha256=hashlib.sha256(clean[:n].tobytes()).hexdigest(), changed_source_sha256=hashlib.sha256(changed[:n].tobytes()).hexdigest()))
    return gates


def audit_bank(seed, public=False):
    from stability_stopping_study import stopping_rule
    p = verify()
    directory = (PUBLIC if public else OUT)/f's{seed}'
    mask = npz(directory/'masks.npz')['balanced']
    assert np.array_equal(mask, reference_masks(p['balanced_seed_base']+seed)) and mask.shape == (360, 81) and (mask.sum(1) == 72).all()
    predraw = json.loads((directory/'predraw-receipt.json').read_text())
    assert predraw['masks_sha256'] == sha(directory/'masks.npz') and predraw['seed'] == seed and predraw['protocol_sha256'] == sha(PROTOCOL)
    clean = noise = None
    expected = {}
    if not public:
        entry = json.loads((OUT/'input_manifest.json').read_text())['inputs'][seed]
        assert sha(ROOT/entry['completion_path']) == entry['completion_sha256']
        clean = npz(ROOT/entry['completion_path'])['solutions']
        noise = npz(directory/'potential_noise.npz')
        rng = np.random.default_rng(p['noise_seed_base']+seed)
        assert np.array_equal(rng.random((360, 81)), noise['iid_uniforms']) and np.array_equal(rng.integers(1, 9, (360, 81), dtype=np.int8), noise['iid_offsets'])
    ii, jj = np.triu_indices(81, 1)
    truth = (ii//9 == jj//9) | (ii % 9 == jj % 9) | ((ii//27 == jj//27) & (ii % 9//3 == jj % 9//3))
    rows = []
    graphs = []
    for pi, n in enumerate(PREFIXES):
        prefix = f'r{n:03d}'
        arrays = npz(directory/f'{prefix}-models.npz')
        receipt = json.loads((directory/f'{prefix}-receipt.json').read_text())
        index = 4*seed+pi
        assert receipt['index'] == index and receipt['seed'] == seed and receipt['rows'] == n and receipt['protocol_sha256'] == sha(PROTOCOL)
        assert len(receipt['records']) == 40 and receipt['models_sha256'] == sha(directory/f'{prefix}-models.npz') and receipt['predraw_receipt_sha256'] == sha(directory/'predraw-receipt.json')
        given = mask[:n]
        joint = (given[:, ii] & given[:, jj]).sum(0)
        assert (joint == 71*(pi+1)).all() and (given.sum(0) == 80*(pi+1)).all() and np.array_equal(joint, arrays['joint_count'])
        if not public:
            assert receipt['observations_sha256'] == sha(directory/f'{prefix}-observations.npz') and receipt['noise_sha256'] == sha(directory/'potential_noise.npz')
            observed = npz(directory/f'{prefix}-observations.npz')
            expected[n] = reference_observe(given, clean[:n], {k: v[:n] for k, v in noise.items()})
            for key, value in expected[n].items():
                assert np.array_equal(value, observed[key])
            assert np.array_equal(expected[n]['equal_count'], arrays['equal_count'])
            for record in receipt['records']:
                assert record['log_sha256'] == sha(directory/f'{prefix}-d{record["degree"]:02d}.log')
        cost = independent_rate_costs(joint, arrays['equal_count'])
        assert np.array_equal(cost, arrays['cost'])
        checked = grid(arrays, receipt['records'], cost, p['solver'], index)
        degree = checked['selected_degree']
        primal = np.empty(0, np.int64) if degree is None else arrays[f'd{degree:02d}_primal_twice']
        result = statistics(primal, truth)
        graphs.append(primal.copy() if primal.shape == (3240,) and all(int(v) in [0, 2] for v in primal) else None)
        fixed = statistics(arrays['d20_primal_twice'], truth) if receipt['records'][19]['certificate']['certified'] else statistics(np.empty(0, np.int64), truth)
        top = np.zeros(3240, bool)
        top[[edge for _, edge in sorted((int(cost[e]), e) for e in range(3240))[:810]]] = True
        assert np.array_equal(top, arrays['top810_support'])
        rows.append(dict(index=index, seed=seed, rows=n, **checked, **result, pair_coqueries=71*(pi+1), queries_per_cell=80*(pi+1),
                         fixed_degree20=fixed, top810=statistics(2*top.astype(np.int64), truth)))
    decision = reference_stopping(graphs)
    assert decision == stopping_rule(graphs)
    assert json.loads((directory/'stopping.json').read_text()) == dict(seed=seed, protocol_sha256=sha(PROTOCOL), **decision)
    chosen = rows[decision['stop_index']]
    stopped = dict(seed=seed, **decision, classification=chosen['classification'], selected_degree=chosen['selected_degree'],
                   support_sha256=chosen['support_sha256'], missing_true=chosen['missing_true'], extra=chosen['extra'],
                   fixed360_classification=rows[3]['classification'], early_wrong_stable=decision['stop_rows'] < 360 and chosen['classification'] == 'resolved_failure_nonexact')
    gates = []
    realized = {}
    if not public:
        for n in PREFIXES:
            assert np.array_equal(expected[n]['values'], expected[360]['values'][:n]) and np.array_equal(expected[n]['given'], expected[360]['given'][:n])
            realized[str(n)] = realized_noise(mask[:n], clean[:n], {k: v[:n] for k, v in noise.items()})
        gates = complete_gate(mask, clean, noise, expected)
    files = ['masks.npz', 'predraw-receipt.json', 'stopping.json']+[f'r{n:03d}-{suffix}' for n in PREFIXES for suffix in ['models.npz', 'receipt.json']]
    if not public:
        files += ['potential_noise.npz']+[f'r{n:03d}-observations.npz' for n in PREFIXES]
    return dict(status='validated', seed=seed, protocol_sha256=sha(PROTOCOL), auditor_sha256=sha(__file__), rows=rows, stopped=stopped,
                query_accounting=literal_account(mask, decision['stop_rows']), complete_label_gates=gates, realized_noise=realized,
                artifact_sha256={name: sha(directory/name) for name in files})


def aggregate(rows, stopped):
    assert [r['index'] for r in rows] == list(range(128)) and [s['seed'] for s in stopped] == list(range(32))
    groups = []
    for n in [90, 180, 270, 360]:
        part = [r for r in rows if r['rows'] == n]
        assert len(part) == 32
        groups.append(dict(rows=n, **{kind: sum(r['classification'] == kind for r in part) for kind in ['success', 'resolved_failure_nonexact', 'resolved_failure_fractional', 'unresolved']},
                           selected_degree20=sum(r['selected_degree'] == 20 for r in part), fixed_degree20_exact=sum(r['fixed_degree20']['classification'] == 'success' for r in part),
                           top810_exact=sum(r['top810']['classification'] == 'success' for r in part)))
    good = sum(s['classification'] == 'success' for s in stopped)
    unknown = sum(s['classification'] == 'unresolved' for s in stopped)
    total = sum(s['stop_rows'] for s in stopped)
    histogram = {str(n): sum(s['stop_rows'] == n for s in stopped) for n in [180, 270, 360]}
    assert total == 8640+90*(histogram['360']-histogram['180'])
    endpoints = dict(stopped_exact=dict(lower=good, upper=good+unknown, required=31, total=32,
                     decision='supported' if good >= 31 else 'falsified' if good+unknown < 31 else 'inconclusive'),
                     virtual_rows=dict(total_rows=total, total_banks=32, mean_rows_numerator=total, mean_rows_denominator=32, maximum_total_rows=8640,
                     decision='supported' if total <= 8640 else 'falsified'))
    paired = dict(stopped_wins=[], fixed360_wins=[], both_exact=[], both_resolved_failure=[], any_unresolved=[])
    for s in stopped:
        a, b = s['classification'], s['fixed360_classification']
        category = 'any_unresolved' if 'unresolved' in [a, b] else 'both_exact' if a == b == 'success' else 'stopped_wins' if a == 'success' else 'fixed360_wins' if b == 'success' else 'both_resolved_failure'
        paired[category].append(s['seed'])
    return dict(grouped=groups, endpoints=endpoints, stop_histogram=histogram, paired_changes=paired,
                early_wrong_stable_banks=[s['seed'] for s in stopped if s['early_wrong_stable']],
                unavailable_prefixes=sum(not available for s in stopped for available in s['available_prefixes']))


def public_job(seed):
    return audit_bank(seed, True)


def public_audit():
    start = time.monotonic()
    p = verify()
    saved = json.loads((PUBLIC/'summary.json').read_text())
    assert saved['protocol_sha256'] == sha(PROTOCOL)
    identity, _, _ = identity_audit(True)
    for key, value in identity.items():
        assert saved['source_audit'][key] == value
    for path, digest in saved['artifact_sha256'].items():
        assert sha(PUBLIC/path) == digest
    with ProcessPoolExecutor(max_workers=8) as pool:
        audits = list(pool.map(public_job, range(32)))
    rows = [row for a in audits for row in a['rows']]
    stops = [a['stopped'] for a in audits]
    assert rows == saved['rows'] and stops == saved['stopped']
    result = aggregate(rows, stops)
    for key, value in result.items():
        assert saved[key] == value
    q = dict(full_physical_noisy_union=sum(a['query_accounting']['full_physical_noisy_union'] for a in audits),
             virtual_stopped_queries=sum(a['query_accounting']['virtual_stopped_queries'] for a in audits),
             prefix_standalone_all32={str(n): sum(a['query_accounting']['prefix_queries'][str(n)] for a in audits) for n in PREFIXES},
             clean_unique_oracle_cells=identity['clean_unique_oracle_cells'], clean_materialized_cell_instances=933120, official_noisy_label_sequence=11369559)
    assert q == saved['query_accounting'] and q['full_physical_noisy_union'] == 829440 and q['virtual_stopped_queries'] == 72*result['endpoints']['virtual_rows']['total_rows']
    assert saved['certified_LPs'] == sum(r['certified_grid_LPs'] for r in rows) and saved['mutations_rejected'] == sum(r['mutations_rejected'] for r in rows)
    assert saved['analytic_zero_proofs'] == 128 and saved['complete_label_gates'] == sum(len(a['complete_label_gates']) for a in saved['raw_audits']) == 128
    print(json.dumps(dict(status='passed', **result, query_accounting=q, source_identity=identity,
                         certified_LPs=saved['certified_LPs'], mutations_rejected=saved['mutations_rejected'], wall_seconds=time.monotonic()-start,
                         scope='Fresh-ID and mask replay, every prefix score/proof/stopping choice and physical/virtual accounting. Original CSV/noise/label provenance requires charged raw audit.')))


if __name__ == '__main__':
    public_audit()
