"""H1bw: equal-query270 single observations versus90 triple-query decoded observations."""
from datetime import datetime, timezone
import argparse, json, os, shutil
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
from run import ROOT, HERE, sha, save
from partial_degree_study import npz, zero_witness
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate, top_count_support
from laplace_rate_math import rate_costs
from balanced_mask_math import line_complements
from higher_noise_study import selected_answers, NOISE_P
from repeat_query_math import observations

OUT = ROOT/'artifacts/research/20261114/ripper/repeat_query_tradeoff'
REGISTRY = OUT.parent/'repeat_query_tradeoff_registration'
PUBLIC = ROOT/'results/research/ripper/repeat_query_tradeoff'
PROTOCOL = HERE/'protocol-repeat-query-tradeoff.json'
PARENT = ROOT/'results/research/ripper/fixed_270_confirm'
ARMS = ['single270', 'triple90']
ARM_ROWS = dict(single270=270, triple90=90)


def make_masks(seed):
    rng = np.random.default_rng(seed)
    template = line_complements()
    parts = []
    for cycle in range(3):
        cells = rng.permutation(81)
        lines = rng.permutation(90)
        part = np.zeros((90, 81), bool)
        part[:, cells] = template[lines]
        parts.append(part)
    return np.concatenate(parts)


def verify(raw=False):
    p = json.loads(PROTOCOL.read_text())
    for rel, digest in (p['source_sha256'] | p['input_sha256']).items():
        assert sha(ROOT/rel) == digest, rel
    if raw:
        for rel, digest in p['raw_input_sha256'].items():
            assert sha(ROOT/rel) == digest, rel
    return p


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not REGISTRY.exists() and not PUBLIC.exists()
    gate = HERE/'repeat-query-tradeoff-preflight.json'
    checked = json.loads(gate.read_text())
    assert checked['status'] == 'passed'
    for rel, digest in checked['source_sha256'].items():
        assert sha(ROOT/rel) == digest
    seed_audit = HERE/'repeat-query-tradeoff-seed-audit.json'
    assert json.loads(seed_audit.read_text())['status'] == 'passed' and sha(seed_audit) == checked['seed_audit_sha256']
    channel = HERE/'repeat-query-channel.json'
    assert json.loads(channel.read_text())['status'] == 'passed' and sha(channel) == checked['exact_channel_sha256']
    old = json.loads((HERE/'protocol-fixed-270-confirm.json').read_text())
    previous_ids = npz(PARENT/'excluded_ids.npz')
    selected = npz(PARENT/'selected_sources.npz')['all_source_row']
    excluded = np.union1d(previous_ids['excluded'], selected)
    ripper = np.union1d(previous_ids['ripper_prior'], selected)
    assert len(excluded) == 2397093 and len(ripper) == 2160668 and np.isin(ripper, excluded).all()
    previous = json.loads((PARENT/'source-audit.json').read_text())
    assert previous['updated_ripper_ledger_union'] == len(ripper) and previous['updated_combined_exclusion_union'] == len(excluded)
    REGISTRY.mkdir()
    np.savez_compressed(REGISTRY/'excluded_ids.npz', excluded=excluded, ripper_prior=ripper)
    ledger = dict(status='sealed_before_new_draws', scope='Declared combined Ripper/H4ak exclusions through H1bv only; not global repository untouchedness.',
                  excluded_unique=len(excluded), ripper_prior_unique=len(ripper), eligible_rows=old['pool_size']-len(excluded),
                  parent_protocol_sha256=sha(HERE/'protocol-fixed-270-confirm.json'), parent_source_audit_sha256=sha(PARENT/'source-audit.json'),
                  parent_selected_ids_sha256=sha(PARENT/'selected_sources.npz'), parent_exclusions_sha256=sha(PARENT/'excluded_ids.npz'), excluded_file_sha256=sha(REGISTRY/'excluded_ids.npz'))
    save(REGISTRY/'ledger.json', ledger)
    proposal = HERE/'repeat-query-tradeoff-proposal.json'
    design = json.loads(proposal.read_text())
    inputs = [gate, seed_audit, HERE/'repeat-query-channel.json', proposal, REGISTRY/'excluded_ids.npz', REGISTRY/'ledger.json', HERE/'protocol-fixed-270-confirm.json',
              PARENT/'excluded_ids.npz', PARENT/'selected_sources.npz', PARENT/'source-audit.json', PARENT/'summary.json']
    p = dict(design, frozen_utc=datetime.now(timezone.utc).isoformat(), status='frozen_before_new_source_draws_masks_noise_and_LPs',
             source=old['source'], source_csv_sha256=old['source_csv_sha256'], sampling=design['source'], noise_probability_hex=NOISE_P.hex(),
             source_seed_base=design['seeds']['source'], balanced_seed_base=design['seeds']['balanced_mask'], noise_seed_base=design['seeds']['noise'],
             repeat1_seed_base=design['seeds']['repeat1'], repeat2_seed_base=design['seeds']['repeat2'],
             degrees=old['degrees'], candidate_degrees=old['candidate_degrees'], costs=old['costs'], objective=old['objective'], selector=old['selector'], proof=old['proof'], prior=old['prior'],
             solver=dict(old['solver'], random_seed_base=design['seeds']['solver_base']), source_sha256=checked['source_sha256'],
             input_sha256={str(path.relative_to(ROOT)): sha(path) for path in inputs}, raw_input_sha256={old['source']: old['source_csv_sha256']})
    save(PROTOCOL, p)
    print(json.dumps(dict(protocol_sha256=sha(PROTOCOL), excluded_ids=len(excluded), eligible_rows=ledger['eligible_rows'])))


def select():
    p = verify()
    assert not OUT.exists()
    excluded = npz(REGISTRY/'excluded_ids.npz')['excluded']
    eligible = np.setdiff1d(np.arange(p['pool_size'], dtype=np.int64), excluded, assume_unique=True)
    assert len(eligible) == 1434901
    draws = np.stack([np.random.default_rng(p['source_seed_base']+bank).choice(eligible, 270, replace=False) for bank in range(64)])
    OUT.mkdir()
    np.savez_compressed(OUT/'selected_sources.npz', source_rows=draws, all_source_row=np.unique(draws))
    save(OUT/'identity-seal.json', dict(status='sealed_before_selected_labels', protocol_sha256=sha(PROTOCOL), utc=datetime.now(timezone.utc).isoformat(),
         selected_sources_sha256=sha(OUT/'selected_sources.npz'), new_unique_source_ids=len(np.unique(draws)), crossbank_repeated_instances=draws.size-len(np.unique(draws))))


def predraw(seed, p=None):
    p = verify() if p is None else p
    assert (OUT/'identity-seal.json').exists() and not (OUT/'input_manifest.json').exists()
    directory = OUT/f's{seed}'
    directory.mkdir(exist_ok=False)
    np.savez_compressed(directory/'masks.npz', balanced=make_masks(p['balanced_seed_base']+seed))
    save(directory/'predraw-receipt.json', dict(status='sealed_before_selected_labels', seed=seed, protocol_sha256=sha(PROTOCOL), utc=datetime.now(timezone.utc).isoformat(),
         identity_seal_sha256=sha(OUT/'identity-seal.json'), masks_sha256=sha(directory/'masks.npz')))


def materialize():
    p = verify()
    predraws = {}
    for seed in range(64):
        directory = OUT/f's{seed}'
        receipt = json.loads((directory/'predraw-receipt.json').read_text())
        assert receipt['seed'] == seed and receipt['protocol_sha256'] == sha(PROTOCOL) and receipt['masks_sha256'] == sha(directory/'masks.npz')
        predraws[str(seed)] = sha(directory/'predraw-receipt.json')
    source = ROOT/p['source']
    assert sha(source) == p['source_csv_sha256']
    draws = npz(OUT/'selected_sources.npz')['source_rows']
    wanted = {}
    solutions = np.zeros((64, 270, 81), np.int8)
    for seed in range(64):
        for offset, row in enumerate(draws[seed]):
            wanted.setdefault(int(row), []).append((seed, offset))
    remaining = set(wanted)
    for index, board in selected_answers(source, wanted):
        assert board.shape == (81,)
        for seed, offset in wanted[index]:
            solutions[seed, offset] = board
        remaining.remove(index)
        if not remaining:
            break
    assert not remaining
    entries = []
    for seed in range(64):
        path = OUT/f'training-s{seed}.npz'
        np.savez_compressed(path, solutions=solutions[seed], source_row=draws[seed])
        entries.append(dict(seed=seed, role='training', completion_path=str(path.relative_to(ROOT)), completion_sha256=sha(path)))
    manifest = dict(protocol_sha256=sha(PROTOCOL), all64_full_masks_sealed_before_materialization=True, predraw_receipts_sha256=predraws,
                    identity_seal_sha256=sha(OUT/'identity-seal.json'), selected_sources_sha256=sha(OUT/'selected_sources.npz'), inputs=entries)
    save(OUT/'input_manifest.json', manifest)
    save(OUT/'all_source_manifest.json', dict(**manifest, purpose='Future exclusions must include these 64 fresh recipient banks.'))


def bank(seed, p=None):
    p = verify() if p is None else p
    source_audit = json.loads((OUT/'source-audit.json').read_text())
    assert source_audit['status'] == 'validated' and source_audit['protocol_sha256'] == sha(PROTOCOL) and source_audit['input_manifest_sha256'] == sha(OUT/'input_manifest.json')
    directory = OUT/f's{seed}'
    mask = npz(directory/'masks.npz')['balanced']
    entry = json.loads((OUT/'input_manifest.json').read_text())['inputs'][seed]
    assert sha(ROOT/entry['completion_path']) == entry['completion_sha256']
    clean = npz(ROOT/entry['completion_path'])['solutions']
    noise = {}
    for name, base, n in [('first', p['noise_seed_base'], 270), ('repeat1', p['repeat1_seed_base'], 90), ('repeat2', p['repeat2_seed_base'], 90)]:
        rng = np.random.default_rng(base+seed)
        noise[name+'_iid_uniforms'] = rng.random((n, 81))
        noise[name+'_iid_offsets'] = rng.integers(1, 9, (n, 81), dtype=np.int8)
    np.savez_compressed(directory/'potential_noise.npz', **noise)
    observed_arms = observations(mask, clean, noise)
    ii, jj = np.triu_indices(81, 1)
    for arm_index, arm in enumerate(ARMS):
        index = 2*seed+arm_index
        prefix = arm
        rows = ARM_ROWS[arm]
        observed = observed_arms[arm]
        np.savez_compressed(directory/f'{prefix}-observations.npz', **observed)
        cost = rate_costs(observed['joint_count'], observed['equal_count'])
        arrays = dict(cost=cost, joint_count=observed['joint_count'], equal_count=observed['equal_count'], top810_support=top_count_support(np.ones(3240, bool), cost))
        arrays.update({'d00_'+key: value for key, value in zero_witness(81, cost).items()})
        records = []
        for degree in range(1, 41):
            log = directory/f'{prefix}-d{degree:02d}.log'
            x, y, solver = solve(81, degree, ii, jj, cost, p['solver']['random_seed_base']-2026111502+index*41+degree, log)
            if solver['solver_optimal'] and solver['value_valid'] and solver['dual_valid']:
                certificate, witness = attempt_certificate(81, degree, ii, jj, cost, x, y)
            else:
                certificate = dict(certified=False, reason='solver_status_or_validity')
                witness = {key: np.empty(0, np.int64) for key in ['primal_twice', 'node_dual_twice', 'upper_dual_twice']}
            arrays.update({f'd{degree:02d}_'+key: value for key, value in dict(primal_values=x, node_dual_values=y, **witness).items()})
            records.append(dict(degree=degree, solver=solver, certificate=certificate, log_sha256=sha(log)))
        np.savez_compressed(directory/f'{prefix}-models.npz', **arrays)
        save(directory/f'{prefix}-receipt.json', dict(index=index, seed=seed, arm=arm, rows=rows, protocol_sha256=sha(PROTOCOL), records=records,
             models_sha256=sha(directory/f'{prefix}-models.npz'), observations_sha256=sha(directory/f'{prefix}-observations.npz'),
             noise_sha256=sha(directory/'potential_noise.npz'), predraw_receipt_sha256=sha(directory/'predraw-receipt.json')))
    from audit_repeat_query_tradeoff import audit_bank
    checked = audit_bank(seed)
    save(directory/'audit.json', checked)
    print(json.dumps(dict(seed=seed, complete=True)), flush=True)


def predraw_all():
    p = verify()
    for seed in range(64):
        predraw(seed, p)
    print(json.dumps(dict(status='all_masks_sealed', banks=64)), flush=True)


def worker(worker_id):
    assert 0 <= worker_id < 8
    p = verify()
    for seed in range(worker_id, 64, 8):
        bank(seed, p)


def summary():
    from audit_repeat_query_tradeoff import aggregate
    p = verify()
    assert not PUBLIC.exists()
    PUBLIC.mkdir(parents=True)
    audits = []
    files = {}
    for seed in range(64):
        directory = OUT/f's{seed}'
        checked = json.loads((directory/'audit.json').read_text())
        assert checked['status'] == 'validated' and checked['protocol_sha256'] == sha(PROTOCOL) and checked['auditor_sha256'] == sha(HERE/'audit_repeat_query_tradeoff.py')
        for name, digest in checked['artifact_sha256'].items():
            assert sha(directory/name) == digest
        audits.append(checked)
        destination = PUBLIC/f's{seed}'
        destination.mkdir()
        for name in ['masks.npz', 'predraw-receipt.json']+[f'{arm}-{suffix}' for arm in ARMS for suffix in ['models.npz', 'receipt.json']]:
            shutil.copyfile(directory/name, destination/name)
            files[f's{seed}/{name}'] = sha(directory/name)
    rows = [row for checked in audits for row in checked['rows']]
    outcomes = aggregate(rows)
    for name in ['selected_sources.npz', 'identity-seal.json', 'input_manifest.json', 'all_source_manifest.json', 'source-audit.json']:
        shutil.copyfile(OUT/name, PUBLIC/name)
    for name in ['excluded_ids.npz', 'ledger.json']:
        shutil.copyfile(REGISTRY/name, PUBLIC/name)
    source_audit = json.loads((OUT/'source-audit.json').read_text())
    accounting = {key:sum(a['query_accounting'][key] for a in audits) for key in ['full_physical_noisy_union', 'single270_deployment', 'triple90_deployment', 'shared_first_query90', 'extra_repeat_queries']}
    accounting.update(clean_unique_oracle_cells=source_audit['clean_unique_oracle_cells'], clean_materialized_cell_instances=1399680, official_noisy_label_sequence=14687319)
    assert accounting['full_physical_noisy_union'] == 2073600
    assert accounting['single270_deployment'] == accounting['triple90_deployment'] == 1244160
    assert accounting['shared_first_query90'] == 414720 and accounting['extra_repeat_queries'] == 829440
    result = dict(status='validated', protocol_sha256=sha(PROTOCOL), scope=p['scope'], rows=rows, **outcomes,
                  raw_audits=audits, query_accounting=accounting, source_audit=source_audit,
                  certified_LPs=sum(r['certified_grid_LPs'] for r in rows), mutations_rejected=sum(r['mutations_rejected'] for r in rows),
                  analytic_zero_proofs=128, complete_label_gates=128, artifact_sha256=files, all_banks_retained=True)
    save(OUT/'audit.json', result)
    save(PUBLIC/'summary.json', result)
    print(json.dumps(dict(endpoints=result['endpoints'], grouped=result['grouped'], query_accounting=accounting)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'select', 'predraw-all', 'materialize', 'source-audit', 'worker', 'summary'])
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.seed)
    elif args.action == 'predraw-all':
        predraw_all()
    elif args.action == 'source-audit':
        from audit_repeat_query_tradeoff import sources
        sources()
    else:
        globals()[args.action]()
