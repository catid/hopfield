"""Independent H1bw source, repeated-label decoder, LP and paired-loss audit."""
from concurrent.futures import ProcessPoolExecutor
import csv, hashlib, json, time
import numpy as np
from repeat_query_tradeoff_study import ROOT, HERE, OUT, PUBLIC, REGISTRY, PROTOCOL, PARENT, ARMS, ARM_ROWS, sha, save, npz, verify
from audit_extra_coverage import statistics
from audit_partial_degree import grid
from audit_laplace_rate import independent_rate_costs
from audit_balanced_mask_math import reference_lines
from audit_higher_noise import reference_corrupt


def identity_audit(public=False):
    p=verify();base=PUBLIC if public else OUT;register=PUBLIC if public else REGISTRY
    parent=npz(PARENT/'excluded_ids.npz');fresh=npz(PARENT/'selected_sources.npz')['all_source_row'];excluded=np.unique(np.concatenate([parent['excluded'],fresh]));ripper=np.unique(np.concatenate([parent['ripper_prior'],fresh]))
    data=npz(register/'excluded_ids.npz');assert np.array_equal(excluded,data['excluded']) and len(excluded)==2397093 and np.array_equal(ripper,data['ripper_prior']) and len(ripper)==2160668 and np.isin(ripper,excluded).all()
    ledger=json.loads((register/'ledger.json').read_text())
    assert ledger['parent_protocol_sha256']==sha(HERE/'protocol-fixed-270-confirm.json') and ledger['parent_source_audit_sha256']==sha(PARENT/'source-audit.json') and ledger['parent_selected_ids_sha256']==sha(PARENT/'selected_sources.npz') and ledger['parent_exclusions_sha256']==sha(PARENT/'excluded_ids.npz') and ledger['excluded_file_sha256']==sha(register/'excluded_ids.npz')
    selected=npz(base/'selected_sources.npz');draws=selected['source_rows'];seal=json.loads((base/'identity-seal.json').read_text());manifest=json.loads((base/'input_manifest.json').read_text());eligible=np.setdiff1d(np.arange(p['pool_size'],dtype=np.int64),excluded,assume_unique=True)
    assert len(eligible)==1434901 and draws.shape==(64,270) and draws.dtype==np.int64
    assert seal['status']=='sealed_before_selected_labels' and seal['protocol_sha256']==manifest['protocol_sha256']==sha(PROTOCOL) and seal['selected_sources_sha256']==manifest['selected_sources_sha256']==sha(base/'selected_sources.npz')
    assert manifest['identity_seal_sha256']==sha(base/'identity-seal.json') and manifest['all64_full_masks_sealed_before_materialization'] and len(manifest['predraw_receipts_sha256'])==64
    assert [e['seed'] for e in manifest['inputs']]==list(range(64))
    for seed in range(64):
        expected=np.random.default_rng(p['source_seed_base']+seed).choice(eligible,270,replace=False);assert np.array_equal(expected,draws[seed]) and len(np.unique(draws[seed]))==270
        d=base/f's{seed}';predraw=json.loads((d/'predraw-receipt.json').read_text());assert predraw['seed']==seed and predraw['protocol_sha256']==sha(PROTOCOL) and predraw['identity_seal_sha256']==sha(base/'identity-seal.json') and predraw['masks_sha256']==sha(d/'masks.npz')
        assert predraw['status']=='sealed_before_selected_labels' and manifest['predraw_receipts_sha256'][str(seed)]==sha(d/'predraw-receipt.json')
    unique=np.unique(draws);assert np.array_equal(unique,selected['all_source_row']) and not np.intersect1d(unique,excluded).size
    assert len(unique)==seal['new_unique_source_ids'] and draws.size-len(unique)==seal['crossbank_repeated_instances']
    assert json.loads((base/'all_source_manifest.json').read_text())['inputs']==manifest['inputs']
    return dict(new_unique_source_ids=len(unique),source_instances=int(draws.size),crossbank_repeated_instances=int(draws.size-len(unique)),within_bank_repeats=0,declared_exclusion_overlap=0,
                updated_ripper_ledger_union=2160668+len(unique),updated_combined_exclusion_union=2397093+len(unique),clean_unique_oracle_cells=81*len(unique),clean_materialized_cell_instances=1399680),draws,manifest


def sources():
    p=verify(True);identity,draws,manifest=identity_audit();wanted={};hashes={}
    for entry in manifest['inputs']:
        seed=entry['seed'];path=ROOT/entry['completion_path'];assert entry['completion_sha256']==sha(path);hashes[entry['completion_path']]=sha(path);z=npz(path)
        assert set(z)=={'solutions','source_row'} and np.array_equal(z['source_row'],draws[seed]);boards=z['solutions'];assert boards.shape==(270,81) and boards.dtype==np.int8 and ((boards>=0)&(boards<9)).all()
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
                **identity,independent_csv_answers_verified=len(wanted),new_donor_queries=0,all64_full_masks_sealed_before_selected_labels=True,completion_sha256=hashes,scope=p['scope'])
    save(OUT/'source-audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='completion_sha256'}),flush=True)


def reference_masks(seed):
    rng = np.random.default_rng(seed)
    lines = reference_lines()
    mask = np.ones((270, 81), bool)
    for cycle in range(3):
        permutation = list(map(int, rng.permutation(81)))
        order = list(map(int, rng.permutation(90)))
        for offset, line in enumerate(order):
            for cell in lines[line]:
                mask[90*cycle+offset, permutation[cell]] = False
    return mask


def literal_account(mask):
    # A repeated measurement has its own pass identity, even on the same cell.
    single = {(row, int(cell), 0) for row in range(270) for cell in np.flatnonzero(mask[row])}
    triple = {(row, int(cell), query) for row in range(90) for cell in np.flatnonzero(mask[row]) for query in range(3)}
    assert len(single) == len(triple) == 19440
    assert len(single & triple) == 6480 and len(triple-single) == 12960
    return dict(full_physical_noisy_union=len(single | triple), single270_deployment=len(single),
                triple90_deployment=len(triple), shared_first_query90=len(single & triple), extra_repeat_queries=len(triple-single))


def reference_decode(first, second, third):
    answer = np.empty(first.shape, np.int8)
    for index in np.ndindex(first.shape):
        labels = [int(first[index]), int(second[index]), int(third[index])]
        frequency = {digit: labels.count(digit) for digit in set(labels)}
        most = max(frequency.values())
        answer[index] = labels[0] if most == 1 else next(digit for digit, count in frequency.items() if count == most)
    return answer


def reference_observations(mask, clean, noise):
    queries = {}
    for name, n in [('first', 270), ('repeat1', 90), ('repeat2', 90)]:
        queries[name] = reference_corrupt(clean[:n], {key: noise[name+'_'+key] for key in ['iid_uniforms', 'iid_offsets']})
    decoded = reference_decode(queries['first'][:90], queries['repeat1'], queries['repeat2'])
    result = {}
    for arm, labels in [('single270', queries['first']), ('triple90', decoded)]:
        n = len(labels)
        given = mask[:n]
        values = np.full(labels.shape, -1, np.int8)
        joint = np.zeros((81, 81), np.int32)
        equal = np.zeros_like(joint)
        for row in range(n):
            cells = np.flatnonzero(given[row])
            values[row, cells] = labels[row, cells]
            block = np.ix_(cells, cells)
            joint[block] += 1
            retained = labels[row, cells]
            equal[block] += retained[:, None] == retained[None, :]
        ii, jj = np.triu_indices(81, 1)
        result[arm] = dict(given=given, values=values, joint_count=joint[ii, jj], equal_count=equal[ii, jj])
    return result, queries, decoded


def complete_gate(mask, clean, noise, expected):
    from repeat_query_math import observations
    changed = ((clean+(np.arange(81) % 8+1)[None, :]) % 9).astype(np.int8)
    assert (changed != clean).all()
    altered = observations(mask, changed, noise)
    gates = []
    for arm in ARMS:
        actual, old = altered[arm], expected[arm]
        given = old['given']
        n = len(given)
        assert np.array_equal(actual['given'], given) and np.array_equal(actual['joint_count'], old['joint_count'])
        assert (actual['values'][given] != old['values'][given]).all() and not np.array_equal(actual['equal_count'], old['equal_count'])
        gates.append(dict(arm=arm, rows=n, status='passed', all_source_and_queried_decoded_values_changed=True,
                          masks_and_joint_counts_unchanged=True, equality_counts_changed=True,
                          original_source_sha256=hashlib.sha256(clean[:n].tobytes()).hexdigest(),
                          changed_source_sha256=hashlib.sha256(changed[:n].tobytes()).hexdigest()))
    return gates


def noise_diagnostics(mask, clean, queries, decoded):
    result = {}
    for name, labels in list(queries.items())+[('decoded', decoded)]:
        given = mask[:len(labels)]
        truth = clean[:len(labels)]
        bad = labels != truth
        result[name] = dict(full_cells=labels.size, queried_cells=int(given.sum()),
                            full_cell_errors=int(bad.sum()), queried_cell_errors=int(bad[given].sum()),
                            queried_error_offsets=[int(((labels-truth) % 9 == offset)[given].sum()) for offset in range(1, 9)])
    first, second, third = queries['first'][:90], queries['repeat1'], queries['repeat2']
    ties = (first != second) & (first != third) & (second != third)
    result['ties'] = dict(all_cells=int(ties.sum()), queried_cells=int(ties[mask[:90]].sum()))
    return result


def audit_bank(seed, public=False):
    p = verify()
    directory = (PUBLIC if public else OUT)/f's{seed}'
    mask = npz(directory/'masks.npz')['balanced']
    assert np.array_equal(mask, reference_masks(p['balanced_seed_base']+seed)) and mask.shape == (270, 81) and (mask.sum(1) == 72).all()
    predraw = json.loads((directory/'predraw-receipt.json').read_text())
    assert predraw['masks_sha256'] == sha(directory/'masks.npz') and predraw['seed'] == seed and predraw['protocol_sha256'] == sha(PROTOCOL)
    clean = noise = queries = decoded = None
    expected = {}
    if not public:
        entry = json.loads((OUT/'input_manifest.json').read_text())['inputs'][seed]
        assert sha(ROOT/entry['completion_path']) == entry['completion_sha256']
        clean = npz(ROOT/entry['completion_path'])['solutions']
        noise = npz(directory/'potential_noise.npz')
        assert len(noise) == 6
        for name, key, n in [('first', 'noise_seed_base', 270), ('repeat1', 'repeat1_seed_base', 90), ('repeat2', 'repeat2_seed_base', 90)]:
            rng = np.random.default_rng(p[key]+seed)
            assert np.array_equal(rng.random((n, 81)), noise[name+'_iid_uniforms'])
            assert np.array_equal(rng.integers(1, 9, (n, 81), dtype=np.int8), noise[name+'_iid_offsets'])
        expected, queries, decoded = reference_observations(mask, clean, noise)
    ii, jj = np.triu_indices(81, 1)
    truth = (ii//9 == jj//9) | (ii % 9 == jj % 9) | ((ii//27 == jj//27) & (ii % 9//3 == jj % 9//3))
    rows = []
    for arm_index, arm in enumerate(ARMS):
        n = ARM_ROWS[arm]
        arrays = npz(directory/f'{arm}-models.npz')
        receipt = json.loads((directory/f'{arm}-receipt.json').read_text())
        index = 2*seed+arm_index
        assert receipt['index'] == index and receipt['seed'] == seed and receipt['rows'] == n and receipt['arm'] == arm and receipt['protocol_sha256'] == sha(PROTOCOL)
        assert len(receipt['records']) == 40 and receipt['models_sha256'] == sha(directory/f'{arm}-models.npz') and receipt['predraw_receipt_sha256'] == sha(directory/'predraw-receipt.json')
        given = mask[:n]
        joint = (given[:, ii] & given[:, jj]).sum(0)
        assert (joint == 71*(n//90)).all() and (given.sum(0) == 80*(n//90)).all() and np.array_equal(joint, arrays['joint_count'])
        if not public:
            assert receipt['observations_sha256'] == sha(directory/f'{arm}-observations.npz') and receipt['noise_sha256'] == sha(directory/'potential_noise.npz')
            observed = npz(directory/f'{arm}-observations.npz')
            for key, value in expected[arm].items():
                assert np.array_equal(value, observed[key])
            assert np.array_equal(expected[arm]['equal_count'], arrays['equal_count'])
            for record in receipt['records']:
                assert record['log_sha256'] == sha(directory/f'{arm}-d{record["degree"]:02d}.log')
        cost = independent_rate_costs(joint, arrays['equal_count'])
        assert np.array_equal(cost, arrays['cost'])
        checked = grid(arrays, receipt['records'], cost, p['solver'], index)
        degree = checked['selected_degree']
        primal = np.empty(0, np.int64) if degree is None else arrays[f'd{degree:02d}_primal_twice']
        result = statistics(primal, truth)
        fixed = statistics(arrays['d20_primal_twice'], truth) if receipt['records'][19]['certificate']['certified'] else statistics(np.empty(0, np.int64), truth)
        top = np.zeros(3240, bool)
        top[[edge for _, edge in sorted((int(cost[e]), e) for e in range(3240))[:810]]] = True
        assert np.array_equal(top, arrays['top810_support'])
        rows.append(dict(index=index, seed=seed, arm=arm, rows=n, **checked, **result, pair_coqueries=71*(n//90), queries_per_cell=240,
                         distinct_rows_per_cell=80*(n//90), fixed_degree20=fixed, top810=statistics(2*top.astype(np.int64), truth)))
    gates = []
    realized = {}
    if not public:
        assert np.array_equal(expected['single270']['values'][:90][mask[:90]], queries['first'][:90][mask[:90]])
        gates = complete_gate(mask, clean, noise, expected)
        realized = noise_diagnostics(mask, clean, queries, decoded)
    files = ['masks.npz', 'predraw-receipt.json']+[f'{arm}-{suffix}' for arm in ARMS for suffix in ['models.npz', 'receipt.json']]
    if not public:
        files += ['potential_noise.npz']+[f'{arm}-observations.npz' for arm in ARMS]
    return dict(status='validated', seed=seed, protocol_sha256=sha(PROTOCOL), auditor_sha256=sha(__file__), rows=rows,
                query_accounting=literal_account(mask), complete_label_gates=gates, realized_noise=realized,
                artifact_sha256={name: sha(directory/name) for name in files})


def aggregate(rows):
    assert [r['index'] for r in rows] == list(range(128))
    groups = []
    for arm in ARMS:
        part = [r for r in rows if r['arm'] == arm]
        assert len(part) == 64
        groups.append(dict(arm=arm, rows=ARM_ROWS[arm], **{kind: sum(r['classification'] == kind for r in part) for kind in ['success', 'resolved_failure_nonexact', 'resolved_failure_fractional', 'unresolved']},
                           selected_degree20=sum(r['selected_degree'] == 20 for r in part), fixed_degree20_exact=sum(r['fixed_degree20']['classification'] == 'success' for r in part),
                           top810_exact=sum(r['top810']['classification'] == 'success' for r in part)))
    single, triple = groups
    low, high = triple['success'], triple['success']+triple['unresolved']
    endpoints = dict(triple_exact=dict(lower=low, upper=high, required=61, total=64,
                     decision='supported' if low >= 61 else 'falsified' if high < 61 else 'inconclusive'))
    paired = dict(triple90_wins=[], single270_wins=[], both_exact=[], both_resolved_failure=[], any_unresolved=[])
    low_loss = high_loss = 0
    failures = ['resolved_failure_nonexact', 'resolved_failure_fractional']
    for seed in range(64):
        first, second = [rows[2*seed+i]['classification'] for i in range(2)]
        assert first in failures+['success', 'unresolved'] and second in failures+['success', 'unresolved']
        low_loss += first == 'success' and second in failures
        high_loss += first in ['success', 'unresolved'] and second != 'success'
        category = 'any_unresolved' if 'unresolved' in [first,second] else 'both_exact' if first==second=='success' else 'triple90_wins' if second=='success' else 'single270_wins' if first=='success' else 'both_resolved_failure'
        paired[category].append(seed)
    endpoints['paired_loss'] = dict(lower=low_loss, upper=high_loss, maximum=3, total=64,
                                    decision='supported' if high_loss <= 3 else 'falsified' if low_loss > 3 else 'inconclusive')
    return dict(grouped=groups, endpoints=endpoints, paired_changes=paired)


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
        audits = list(pool.map(public_job, range(64)))
    rows = [row for a in audits for row in a['rows']]
    assert rows == saved['rows']
    result = aggregate(rows)
    for key, value in result.items():
        assert saved[key] == value
    q = {key:sum(a['query_accounting'][key] for a in audits) for key in ['full_physical_noisy_union', 'single270_deployment', 'triple90_deployment', 'shared_first_query90', 'extra_repeat_queries']}
    q.update(clean_unique_oracle_cells=identity['clean_unique_oracle_cells'], clean_materialized_cell_instances=1399680, official_noisy_label_sequence=14687319)
    assert q == saved['query_accounting'] and q['full_physical_noisy_union'] == 2073600
    assert q['single270_deployment'] == q['triple90_deployment'] == 1244160 and q['shared_first_query90'] == 414720 and q['extra_repeat_queries'] == 829440
    assert saved['certified_LPs'] == sum(r['certified_grid_LPs'] for r in rows) and saved['mutations_rejected'] == sum(r['mutations_rejected'] for r in rows)
    assert saved['analytic_zero_proofs'] == 128 and saved['complete_label_gates'] == sum(len(a['complete_label_gates']) for a in saved['raw_audits']) == 128
    print(json.dumps(dict(status='passed', **result, query_accounting=q, source_identity=identity,
                         certified_LPs=saved['certified_LPs'], mutations_rejected=saved['mutations_rejected'], wall_seconds=time.monotonic()-start,
                         scope='Fresh-ID/mask replay, both arm score/proof grids, paired endpoints and physical union. Original CSV/noise/decoder provenance requires charged raw audit.')))


if __name__ == '__main__':
    public_audit()


