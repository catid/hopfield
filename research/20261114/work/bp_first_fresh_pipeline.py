"""H2bo source gate or fresh label-blind complete-cohort answer pipeline."""
import copy
import json
from pathlib import Path
import time
import numpy as np
import study as S
from bounded_archive_writer import ArchiveWriter
from bp_first_fresh_batch import run as batch
from bp_first_geometry_accounting import batch_cost
from bp_first_geometry_parity import verify as parity, array_sha

ARMS = ('pl', 'old_binary', 'connectivity_veto', 'original_lexical16',
        'unified16', 'unified48', 'unified64', 'unified96')


def drain(writer):
    started = time.perf_counter()
    while writer.pending:
        writer._take_oldest()
    assert not writer.pending
    return time.perf_counter() - started


def load_models(p, mode):
    torch, _, _ = S.imports()
    models, doubles, receipts = {}, {}, []
    for seed in range(3):
        models[seed], doubles[seed] = {}, {}
        for label, source_arm in ([('bp', 'pl'), ('lp', 'old_binary')]
                                  if mode == 'BP_geometry_LP96' else [('lp', 'old_binary')]):
            torch.cuda.synchronize()
            started = time.perf_counter()
            model = S.model_for(seed, 180, 'original', 'cuda')
            model.load_state_dict(torch.load(S.ROOT / p['models'][str(seed)][source_arm]['path'],
                                            map_location='cuda', weights_only=True))
            models[seed][label] = model
            doubles[seed][label] = copy.deepcopy(model).double()
            torch.cuda.synchronize()
            receipts.append({'seed': seed, 'model': label,
                'checkpoint_sha256': p['models'][str(seed)][source_arm]['sha256'],
                'seconds': time.perf_counter() - started})
    return models, doubles, receipts


def payloads(result, destination, tag, protocol_hash, retain_couplings, offset):
    arrays = {'answer_' + k: v for k, v in result['answer'].items()}
    arrays.update(geometry_valid=result['geometry_valid'], subset_indices=result['subset_indices'])
    for prefix, group in [('bp_', result['bp_prefix']), ('subset_', result['lp_subset'])]:
        if group is not None:
            arrays.update({prefix + k: v for k, v in group.items()})
    # Source gates verify actual couplings against frozen source arrays. Fresh
    # pair0 retains actual arrays for both arms; later copies carry typed hashes.
    # Final labels-last audit compares every replay against pair0.
    arrays.update({k: v for k, v in result['audit'].items() if k != 'couplings' or retain_couplings})
    answer_item = (destination / (tag + '.npz'), arrays)
    traces = []
    core = result['core']
    if core is not None:
        for v in range(2, 8):
            block = core['blocks'][v]
            raw = block['raw']
            content = {'protocol_sha256': protocol_hash, 'arm': ARMS[v], 'offset': offset,
                       'trace': raw['trace'] if raw else [],
                       'initial_terminal_checks': block['initial_terminal_records'],
                       'final_terminal_checks': raw.get('terminal_check_records', []) if raw else []}
            data = {'trace_utf8': np.frombuffer(json.dumps(content, separators=(',', ':')).encode(), dtype=np.uint8),
                    'valid_prefix_indices': block['valid_prefix_indices'],
                    'states_before': raw['states_before'] if raw else np.zeros((0, 0, 180, 180), bool),
                    'initial_terminal': block['initial_terminal'],
                    'pre_terminal_stopped': raw.get('pre_terminal_stopped', np.zeros(0, bool)) if raw else np.zeros(0, bool),
                    'pre_terminal_block_cap': raw.get('pre_terminal_block_cap', np.zeros(0, bool)) if raw else np.zeros(0, bool)}
            traces.append((destination / (tag + '-' + ARMS[v] + '-trace.npz'), data))
    return answer_item, traces


def diagnostic_source_parity(answer, source_job, bi):
    """Verify every newly added fresh-audit diagnostic on the fixed source gate."""
    with np.load(source_job / f'batch{bi:02d}.npz', allow_pickle=False) as source:
        comparisons = []
        if answer['bp_prefix'] is not None:
            comparisons.extend([(answer['audit']['bp_energy64'], source['energy64'][0], 1e-6),
                (answer['audit']['bp_minimum_flip_margin32'], source['native_minimum_flip_margin32'][0], 1e-9)])
        if answer['core'] is not None:
            ix = (np.arange(len(answer['answer']['prediction'])) if answer['bp_prefix'] is None
                  else answer['subset_indices'])
            comparisons.extend([(answer['audit']['LP_minimum_flip_margin64'], source['minimum_flip_margin64'][1:, ix], 1e-9),
                                (answer['audit']['LP_energy64'], source['energy64'][1:, ix], 1e-6)])
        for actual, expected, tolerance in comparisons:
            assert actual.shape == expected.shape
            assert np.allclose(actual, expected, rtol=0, atol=tolerance, equal_nan=True)
    return {'status': 'passed', 'new_diagnostic_array_comparisons': len(comparisons)}


def cohort(p, protocol_hash, mode, tokens_by_shard, shards, directory,
           watchdog, progress, barrier, batch_selection=None, source_gate=False,
           pair=None, retain_couplings=False):
    torch, _, _ = S.imports()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    setup_started = time.perf_counter()
    writer = ArchiveWriter(asynchronous=True, max_pending=2, memory_gib=8)
    writer_setup = time.perf_counter() - setup_started
    drain_seconds = parity_seconds = payload_seconds = 0.
    rows = []
    try:
        # The writer has no submitted task, so model readiness also has no
        # excluded archive competitor. These fresh loads are fully charged.
        assert not writer.pending
        barrier.gate('readiness-begin')
        torch.cuda.synchronize()
        readiness_started = time.perf_counter()
        models, doubles, readiness = load_models(p, mode)
        torch.cuda.synchronize()
        readiness_seconds = time.perf_counter() - readiness_started
        assert readiness_seconds >= sum(r['seconds'] for r in readiness)
        barrier.gate('readiness-done')
        watchdog()
        jobs = {j['id']: j for j in p['source_jobs']}
        with torch.no_grad():
            for seed in range(3):
                for shard in shards:
                    job_id = f'shard{shard:02d}-s{seed}'
                    source_job = S.ROOT / 'artifacts/research/20261114/work/loop80_confirmation/runs' / job_id
                    for bi in (batch_selection[shard] if batch_selection is not None else range(8)):
                        watchdog()
                        drained = drain(writer)
                        drain_seconds += drained
                        # No asynchronous task is allowed across this boundary.
                        assert not writer.pending
                        sequence = len(rows)
                        barrier.gate(f'batch{sequence:04d}-begin')
                        host_tokens = tokens_by_shard[shard][bi * 32:(bi + 1) * 32]
                        answer = batch(mode, host_tokens, models[seed], doubles[seed], watchdog,
                                       after_answer=lambda: barrier.gate(f'batch{sequence:04d}-answer-done'))
                        cost = batch_cost(answer['timing'])
                        tag = f'{job_id}-batch{bi:02d}'
                        before = time.perf_counter()
                        answer_item, trace_items = payloads(answer, directory, tag, protocol_hash, retain_couplings, bi * 32)
                        payload_seconds += time.perf_counter() - before
                        # Two owned tasks may overlap each other and this batch's
                        # untimed source audit. Both are drained before more GPU work.
                        writer.submit([answer_item])
                        if trace_items:
                            writer.submit(trace_items)
                        before = time.perf_counter()
                        if source_gate:
                            checked = parity(answer, host_tokens, source_job, bi, jobs[job_id]['batch_sha256'])
                            checked['fresh_diagnostics'] = diagnostic_source_parity(answer, source_job, bi)
                        else:
                            checked = {'status': 'fresh_arrays_sealed_without_outcome_join',
                                       'actual_couplings_sha256': array_sha(answer['audit']['couplings']),
                                       'tokens_sha256': array_sha(host_tokens),
                                       'full_model_inputs': len(host_tokens),
                                       'actual_subset_inputs': answer['timing']['actual_subset_inputs'],
                                       'labels_distances_source_outcomes_read': False}
                        checked_seconds = time.perf_counter() - before
                        parity_seconds += checked_seconds
                        row = {'job_id': job_id, 'seed': seed, 'shard': shard, 'batch': bi,
                               'file': answer_item[0].name, 'trace_files': [q[0].name for q in trace_items],
                               'timing': answer['timing'], 'source_parity': checked if source_gate else None,
                               'fresh_array_receipt': None if source_gate else checked,
                               'source_parity_seconds': checked_seconds,
                               'archive_drain_seconds_before_answer': drained,
                               'archive_tasks_at_answer_start': 0,
                               'answer_seconds_checked': cost}
                        rows.append(row)
                        S.save(directory / (tag + '.json'), row)
                        progress(mode=mode, batches_complete=len(rows), seed=seed, shard=shard, batch=bi,
                                 model_inputs_complete=32 * len(rows), actual_subset_inputs=sum(x['timing']['actual_subset_inputs'] for x in rows))
                        del answer, answer_item, trace_items
                        watchdog()
        files = writer.close()
        barrier.gate('archives-done')
        byname = {Path(r['path']).name: r for r in files}
        for row in rows:
            row['file_sha256'] = byname[row['file']]['sha256']
            row['trace_sha256'] = {name: byname[name]['sha256'] for name in row['trace_files']}
        result = {'status': 'complete', 'mode': mode, 'protocol_sha256': protocol_hash,
                  'model_inputs': len(rows) * 32, 'shards': list(shards),
                  'batch_selection': batch_selection, 'source_gate': source_gate,
                  'pair': pair, 'retained_couplings': retain_couplings, 'rows': rows,
                  'model_readiness': readiness,
                  'model_readiness_seconds': readiness_seconds,
                  'readiness_started_monotonic': readiness_started,
                  'readiness_finished_monotonic': readiness_started + readiness_seconds,
                  'readiness_orchestration_seconds': readiness_seconds - sum(r['seconds'] for r in readiness),
                  'batch_answer_seconds': sum(r['timing']['answer_seconds'] for r in rows),
                  'complete_answer_seconds': readiness_seconds + sum(r['timing']['answer_seconds'] for r in rows),
                  'actual_subset_inputs': sum(r['timing']['actual_subset_inputs'] for r in rows),
                  'archive_writer_setup_seconds': writer_setup, 'archive_boundary_drain_seconds': drain_seconds,
                  'source_parity_seconds': parity_seconds, 'payload_seconds': payload_seconds,
                  'diagnostic_seconds': sum(r['timing']['diagnostic_seconds'] for r in rows),
                  'archive_writer_metrics': writer.metrics, 'cohort_wall_seconds': time.perf_counter() - started,
                  'all_writes_closed_hashed_and_joined': writer.closed,
                  'archive_tasks_during_answer_segments': 0,
                  'host_barrier': {'directory': str(barrier.directory.relative_to(S.ROOT)),
                                   'gpu': barrier.gpu, 'events': barrier.events,
                                   'wait_seconds': sum(r['wait_seconds'] for r in barrier.events)},
                  'coupling_evidence': 'Source gates compare every actual coupling array to H2bh. Fresh pair0 retains actual full arrays for both arms; later repeats retain observed typed-array hashes and all states, checked after complete sealing.'}
        S.save(directory / 'result.json', result)
        del models, doubles
        torch.cuda.empty_cache()
        return result
    finally:
        if not writer.closed:
            writer.close()
