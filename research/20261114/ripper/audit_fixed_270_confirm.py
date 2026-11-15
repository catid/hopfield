"""Independent H1bv source, nested masks, noise, LP and paired-endpoint audit."""
from concurrent.futures import ProcessPoolExecutor
import csv, hashlib, json, time
import numpy as np
from fixed_270_confirm_study import ROOT, HERE, OUT, PUBLIC, REGISTRY, PROTOCOL, PARENT, PREFIXES, sha, save, npz, verify
from audit_extra_coverage import statistics
from audit_partial_degree import grid
from audit_laplace_rate import independent_rate_costs
from audit_balanced_mask_math import reference_lines
from audit_higher_noise import reference_observe, realized_noise


def identity_audit(public=False):
    p=verify();base=PUBLIC if public else OUT;register=PUBLIC if public else REGISTRY
    parent=npz(PARENT/'excluded_ids.npz');fresh=npz(PARENT/'selected_sources.npz')['all_source_row'];excluded=np.unique(np.concatenate([parent['excluded'],fresh]));ripper=np.unique(np.concatenate([parent['ripper_prior'],fresh]))
    data=npz(register/'excluded_ids.npz');assert np.array_equal(excluded,data['excluded']) and len(excluded)==2379905 and np.array_equal(ripper,data['ripper_prior']) and len(ripper)==2143480 and np.isin(ripper,excluded).all()
    ledger=json.loads((register/'ledger.json').read_text())
    assert ledger['parent_protocol_sha256']==sha(HERE/'protocol-stability-stopping.json') and ledger['parent_source_audit_sha256']==sha(PARENT/'source-audit.json') and ledger['parent_selected_ids_sha256']==sha(PARENT/'selected_sources.npz') and ledger['parent_exclusions_sha256']==sha(PARENT/'excluded_ids.npz') and ledger['excluded_file_sha256']==sha(register/'excluded_ids.npz')
    selected=npz(base/'selected_sources.npz');draws=selected['source_rows'];seal=json.loads((base/'identity-seal.json').read_text());manifest=json.loads((base/'input_manifest.json').read_text());eligible=np.setdiff1d(np.arange(p['pool_size'],dtype=np.int64),excluded,assume_unique=True)
    assert len(eligible)==1452089 and draws.shape==(64,270) and draws.dtype==np.int64
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
                updated_ripper_ledger_union=2143480+len(unique),updated_combined_exclusion_union=2379905+len(unique),clean_unique_oracle_cells=81*len(unique),clean_materialized_cell_instances=1399680),draws,manifest


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
    shorter = {(row, int(cell)) for row in range(180) for cell in np.flatnonzero(mask[row])}
    full = {(row, int(cell)) for row in range(270) for cell in np.flatnonzero(mask[row])}
    assert shorter < full and len(shorter) == 12960 and len(full) == 19440
    return dict(full_physical_noisy_union=len(shorter | full), prefix180_deployment=len(shorter), additional270_queries=len(full-shorter))


def complete_gate(mask, clean, noise, expected):
    from fixed_270_confirm_study import observe
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
    p = verify()
    directory = (PUBLIC if public else OUT)/f's{seed}'
    mask = npz(directory/'masks.npz')['balanced']
    assert np.array_equal(mask, reference_masks(p['balanced_seed_base']+seed)) and mask.shape == (270, 81) and (mask.sum(1) == 72).all()
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
        assert np.array_equal(rng.random((270, 81)), noise['iid_uniforms']) and np.array_equal(rng.integers(1, 9, (270, 81), dtype=np.int8), noise['iid_offsets'])
    ii, jj = np.triu_indices(81, 1)
    truth = (ii//9 == jj//9) | (ii % 9 == jj % 9) | ((ii//27 == jj//27) & (ii % 9//3 == jj % 9//3))
    rows = []
    for pi, n in enumerate(PREFIXES):
        prefix = f'r{n:03d}'
        arrays = npz(directory/f'{prefix}-models.npz')
        receipt = json.loads((directory/f'{prefix}-receipt.json').read_text())
        index = 2*seed+pi
        assert receipt['index'] == index and receipt['seed'] == seed and receipt['rows'] == n and receipt['protocol_sha256'] == sha(PROTOCOL)
        assert len(receipt['records']) == 40 and receipt['models_sha256'] == sha(directory/f'{prefix}-models.npz') and receipt['predraw_receipt_sha256'] == sha(directory/'predraw-receipt.json')
        given = mask[:n]
        joint = (given[:, ii] & given[:, jj]).sum(0)
        assert (joint == 71*(n//90)).all() and (given.sum(0) == 80*(n//90)).all() and np.array_equal(joint, arrays['joint_count'])
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
        fixed = statistics(arrays['d20_primal_twice'], truth) if receipt['records'][19]['certificate']['certified'] else statistics(np.empty(0, np.int64), truth)
        top = np.zeros(3240, bool)
        top[[edge for _, edge in sorted((int(cost[e]), e) for e in range(3240))[:810]]] = True
        assert np.array_equal(top, arrays['top810_support'])
        rows.append(dict(index=index, seed=seed, rows=n, **checked, **result, pair_coqueries=71*(n//90), queries_per_cell=80*(n//90),
                         fixed_degree20=fixed, top810=statistics(2*top.astype(np.int64), truth)))
    gates = []
    realized = {}
    if not public:
        for n in PREFIXES:
            assert np.array_equal(expected[n]['values'], expected[270]['values'][:n]) and np.array_equal(expected[n]['given'], expected[270]['given'][:n])
            realized[str(n)] = realized_noise(mask[:n], clean[:n], {k: v[:n] for k, v in noise.items()})
        gates = complete_gate(mask, clean, noise, expected)
    files = ['masks.npz', 'predraw-receipt.json']+[f'r{n:03d}-{suffix}' for n in PREFIXES for suffix in ['models.npz', 'receipt.json']]
    if not public:
        files += ['potential_noise.npz']+[f'r{n:03d}-observations.npz' for n in PREFIXES]
    return dict(status='validated', seed=seed, protocol_sha256=sha(PROTOCOL), auditor_sha256=sha(__file__), rows=rows,
                query_accounting=literal_account(mask), complete_label_gates=gates, realized_noise=realized,
                artifact_sha256={name: sha(directory/name) for name in files})


def aggregate(rows):
    assert [r['index'] for r in rows] == list(range(128))
    groups = []
    for n in [180, 270]:
        part = [r for r in rows if r['rows'] == n]
        assert len(part) == 64
        groups.append(dict(rows=n, **{kind: sum(r['classification'] == kind for r in part) for kind in ['success', 'resolved_failure_nonexact', 'resolved_failure_fractional', 'unresolved']},
                           selected_degree20=sum(r['selected_degree'] == 20 for r in part), fixed_degree20_exact=sum(r['fixed_degree20']['classification'] == 'success' for r in part),
                           top810_exact=sum(r['top810']['classification'] == 'success' for r in part)))
    short, full = groups
    bounds = dict(exact270=(full['success'], full['success']+full['unresolved'], 61),
                  paired_gain=(full['success']-short['success']-short['unresolved'], full['success']+full['unresolved']-short['success'], 8))
    endpoints = {name:dict(lower=low, upper=high, required=required, total=64,
                 decision='supported' if low>=required else 'falsified' if high<required else 'inconclusive') for name,(low,high,required) in bounds.items()}
    paired = dict(full270_wins=[], prefix180_wins=[], both_exact=[], both_resolved_failure=[], any_unresolved=[])
    for seed in range(64):
        short_kind, full_kind = [rows[2*seed+i]['classification'] for i in range(2)]
        category = 'any_unresolved' if 'unresolved' in [short_kind,full_kind] else 'both_exact' if short_kind==full_kind=='success' else 'full270_wins' if full_kind=='success' else 'prefix180_wins' if short_kind=='success' else 'both_resolved_failure'
        paired[category].append(seed)
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
    q = dict(full_physical_noisy_union=sum(a['query_accounting']['full_physical_noisy_union'] for a in audits),
             prefix180_deployment=sum(a['query_accounting']['prefix180_deployment'] for a in audits),
             additional270_queries=sum(a['query_accounting']['additional270_queries'] for a in audits),
             clean_unique_oracle_cells=identity['clean_unique_oracle_cells'], clean_materialized_cell_instances=1399680, official_noisy_label_sequence=12613719)
    assert q == saved['query_accounting'] and q['full_physical_noisy_union'] == 1244160 and q['prefix180_deployment'] == 829440 and q['additional270_queries'] == 414720
    assert saved['certified_LPs'] == sum(r['certified_grid_LPs'] for r in rows) and saved['mutations_rejected'] == sum(r['mutations_rejected'] for r in rows)
    assert saved['analytic_zero_proofs'] == 128 and saved['complete_label_gates'] == sum(len(a['complete_label_gates']) for a in saved['raw_audits']) == 128
    print(json.dumps(dict(status='passed', **result, query_accounting=q, source_identity=identity,
                         certified_LPs=saved['certified_LPs'], mutations_rejected=saved['mutations_rejected'], wall_seconds=time.monotonic()-start,
                         scope='Fresh-ID and mask replay, both prefix score/proof grids, paired endpoints and physical union. Original CSV/noise/label provenance requires charged raw audit.')))


if __name__ == '__main__':
    public_audit()
