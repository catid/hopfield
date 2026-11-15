"""H2bo independent synchronized complete-cost and receipt validation."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_batches(receipt):
    shards = receipt['shards']
    assert len(shards) == len(set(shards)) and all(0 <= s < 16 for s in shards)
    selection = receipt['batch_selection']
    if selection is not None:
        assert set(map(int, selection)) == set(shards)
    result = set()
    for seed in range(3):
        for shard in shards:
            batches = selection[str(shard)] if selection is not None else list(range(8))
            assert len(batches) == len(set(batches)) and all(0 <= b < 8 for b in batches)
            result.update((seed, shard, batch) for batch in batches)
    return result


def verify_costs(root,directory,ph,cohorts):
    for r in cohorts:
        assert r['status'] == 'complete' and r['protocol_sha256'] == ph
        assert r['all_writes_closed_hashed_and_joined'] and r['archive_tasks_during_answer_segments'] == 0
        assert r['archive_writer_metrics']['maximum_pending'] <= 2 and r['archive_writer_metrics']['maximum_rss_kib'] <= 8 * 1024**2
        expected = expected_batches(r)
        assert {(x['seed'], x['shard'], x['batch']) for x in r['rows']} == expected and len(r['rows']) == len(expected)
        ready = {(x['seed'], x['model']) for x in r['model_readiness']}
        assert ready == {(seed, model) for seed in range(3) for model in (['lp'] if r['mode'] == 'LP96' else ['bp', 'lp'])}
        assert len(ready) == len(r['model_readiness'])
        assert all(np.isfinite(x['seconds']) and x['seconds'] > 0 for x in r['model_readiness'])
        assert abs(sum(x['seconds'] for x in r['model_readiness']) + r['readiness_orchestration_seconds'] - r['model_readiness_seconds']) < 1e-8
        assert r['readiness_orchestration_seconds'] >= 0
        assert r['model_inputs'] == 32 * len(expected)
        barrier = r['host_barrier']
        if r['kind'] == 'timing':
            assert barrier['gpu'] == (0 if r['pair'] < 2 else 1)
        events = barrier['events']
        required_stages = ['readiness-begin', 'readiness-done']
        for index in range(len(r['rows'])):
            required_stages += [f'batch{index:04d}-begin', f'batch{index:04d}-answer-done']
        required_stages += ['archives-done']
        assert [event['stage'] for event in events] == required_stages
        marker_times = {}
        command = 'gate' if r['kind'] == 'gate' else 'timing'
        for event in events:
            times = []
            for side, gpu in [('own', barrier['gpu']), ('peer', 1 - barrier['gpu'])]:
                file = root / barrier['directory'] / event[side + '_marker']
                assert sha(file) == event[side + '_sha256']
                marker = json.loads(file.read_text())
                assert marker['stage'] == event['stage'] and marker['gpu'] == gpu and marker['protocol_sha256'] == ph
                worker = json.loads((directory / f'{command}-worker{gpu}.json').read_text())
                assert marker['pid'] == worker['pid']
                assert np.isfinite(marker['arrived_monotonic'])
                times.append(marker['arrived_monotonic'])
            assert 0 <= event['wait_seconds'] <= 60.01
            marker_times[event['stage']] = times
        assert abs(sum(e['wait_seconds'] for e in events) - barrier['wait_seconds']) < 1e-8
        assert r['readiness_started_monotonic'] >= max(marker_times['readiness-begin'])
        assert r['readiness_finished_monotonic'] <= marker_times['readiness-done'][0]
        assert abs(r['readiness_finished_monotonic'] - r['readiness_started_monotonic'] - r['model_readiness_seconds']) < 1e-8
        costs = []
        for index, row in enumerate(r['rows']):
            t = row['timing']
            c = t['components']
            required = {'input_transfer', 'hidden_readout_and_flags', 'LP96_complete_prefix', 'LP96_answer_return', 'LP96_original_margin_and_return', 'final_numerical_acceptance'}
            required |= ({'BP_native_solve', 'BP_cap_flags', 'BP_answer_return', 'BP_original_margin_and_return', 'geometry_screen_and_indices', 'LP_subset_gather', 'subset_scatter_and_provenance'} if r['mode'] != 'LP96' else {'model_provenance'})
            assert required <= c.keys() and all(np.isfinite(v) and v >= 0 for v in c.values())
            assert t['n'] == 32 and 0 <= t['actual_subset_inputs'] <= 32
            if r['mode'] == 'BP_geometry_LP96':
                for key in ['LP96_complete_prefix', 'LP96_answer_return', 'LP96_original_margin_and_return']:
                    assert (c[key] == 0) == (t['actual_subset_inputs'] == 0)
            assert t['answer_seconds'] > 0 and t['unattributed_in_segment_seconds'] >= 0
            assert abs(sum(c.values()) + t['unattributed_in_segment_seconds'] - t['answer_seconds']) < 1e-8
            assert row['archive_tasks_at_answer_start'] == 0 and row['archive_drain_seconds_before_answer'] >= 0
            before = marker_times[f'batch{index:04d}-begin']
            after = marker_times[f'batch{index:04d}-answer-done']
            assert t['answer_started_monotonic'] >= max(before)
            assert t['answer_finished_monotonic'] <= after[0]
            assert t['diagnostic_started_monotonic'] >= max(after)
            assert abs(t['answer_finished_monotonic'] - t['answer_started_monotonic'] - t['answer_seconds']) < 1e-8
            assert abs(t['post_answer_barrier_wait_seconds'] - events[3 + 2 * index]['wait_seconds']) < 1e-8
            costs.append(t['answer_seconds'])
        assert abs(sum(costs) - r['batch_answer_seconds']) < 1e-8
        assert abs(sum(costs) + r['model_readiness_seconds'] - r['complete_answer_seconds']) < 1e-8
    pairs=[]
    for pair in range(4):
        control=next(r for r in cohorts if r['kind']=='timing' and r['pair']==pair and r['mode']=='LP96')
        candidate=next(r for r in cohorts if r['kind']=='timing' and r['pair']==pair and r['mode']=='BP_geometry_LP96')
        a,b=control['complete_answer_seconds'],candidate['complete_answer_seconds']
        pairs.append({'pair':pair,'LP96_seconds':a,'candidate_seconds':b,'saving_fraction':Fraction(1)-Fraction(b)/Fraction(a)})
    ordered=sorted(r['saving_fraction'] for r in pairs);median=(ordered[1]+ordered[2])/2
    for row in pairs:row['saving']=float(row.pop('saving_fraction'))
    return {'pairs':pairs,'median_saving':float(median),'exact_median_fraction':str(median),
            'minimum_saving':.10,'decision':'supported' if median>=Fraction(1,10) else 'falsified'}
