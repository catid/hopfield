"""Seal every H2bn cohort receipt before independent graph/outcome joins."""
import json
import study as S

DEST = S.OUT / 'bp_first_geometry_cascade'


def seal():
    target = DEST / 'sealed-replays.json'
    assert not target.exists()
    ph = S.sha(DEST / 'protocol.json')
    files = {}
    for gpu in [0, 1]:
        for kind in ['gate', 'timing']:
            path = DEST / f'{kind}-worker{gpu}.json'
            worker = json.loads(path.read_text())
            assert worker['status'] == 'complete' and worker['protocol_sha256'] == ph
            assert worker['worker_wall_seconds'] <= (300 if kind == 'gate' else 1500)
            files[str(path.relative_to(DEST))] = S.sha(path)
        for name in [f'gate-gpu{gpu}.json', f'timing-gpu{gpu}.json', f'pairs-gpu{gpu}.json']:
            files[name] = S.sha(DEST / name)
        for folder in ['source-gates', 'warmup']:
            for mode in ['LP96', 'BP_geometry_LP96']:
                path = DEST / folder / f'gpu{gpu}-{mode}' / 'result.json'
                receipt = json.loads(path.read_text())
                assert receipt['status'] == 'complete' and receipt['protocol_sha256'] == ph
                files[str(path.relative_to(DEST))] = S.sha(path)
    for pair in range(4):
        for mode in ['LP96', 'BP_geometry_LP96']:
            path = DEST / 'runs' / f'pair{pair}-{mode}' / 'result.json'
            receipt = json.loads(path.read_text())
            assert receipt['status'] == 'complete' and receipt['protocol_sha256'] == ph
            assert receipt['model_inputs'] == 12288
            files[str(path.relative_to(DEST))] = S.sha(path)
    assert len(files) == 26
    result = {'status': 'complete', 'protocol_sha256': ph, 'sealed_utc': S.utc(),
              'cohort_and_worker_receipt_sha256': files,
              'timing_pairs': 4, 'model_inputs_per_paired_arm': 12288,
              'labels_or_shortestness_outcomes_joined': False,
              'scope': 'Complete source/timing/warmup receipts bind every archived batch and trace by hash; all underlying NPZ bytes are rechecked by the independent audit.'}
    S.save(target, result)
    print(json.dumps({'status': 'complete', 'sealed_sha256': S.sha(target), 'receipts': len(files)}, indent=2))


if __name__ == '__main__':
    seal()
