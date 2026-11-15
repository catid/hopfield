"""Post-run H2bn evidence extraction; complete routing/failure public supplement."""
import csv
from functools import lru_cache
import json
from pathlib import Path
import shutil
import time
import numpy as np
import study as S
import fast_npz as Z

DEST = S.OUT / 'bp_first_geometry_cascade'
SOURCE = S.OUT / 'loop80_confirmation'
PUBLIC = DEST / 'public-staging'
ORIGINAL = S.OUT / 'cross_certificate_screen/public-fixtures'


def extract():
    started = time.perf_counter()
    audit = json.loads((DEST / 'audit.json').read_text())
    assert audit['status'] == 'passed'
    protocol = json.loads((DEST / 'protocol.json').read_text())
    original = json.loads((ORIGINAL / 'manifest.json').read_text())
    with (DEST / 'per-model-input.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    fixed = {(shard * 256, seed) for shard in range(16) for seed in range(3)}
    routed = {(int(r['input_index']), int(r['model_seed'])) for r in rows
              if r['mode'] == 'BP_geometry_LP96' and r['routed'] == '1'}
    failures = {(int(r['input_index']), int(r['model_seed'])) for r in rows
                if r['mode'] == 'BP_geometry_LP96' and r['success'] == '0'}
    selected = fixed | routed | failures
    source_map = {(r['input_index'], r['model_seed']): r
                  for r in original['fixed_cases'] + original['negative_or_retention_loss_cases']}
    assert selected <= source_map.keys()
    PUBLIC.mkdir(exist_ok=False)
    fixtures = PUBLIC / 'state-fixtures'
    fixtures.mkdir()
    for name in ['protocol.json', 'synthetic-gate.json', 'sealed-replays.json', 'audit.json', 'audit-worker.json',
                 'per-model-input.csv', 'live-gate.json', 'live-timing.json', 'launcher-gate.json', 'launcher-timing.json']:
        shutil.copyfile(DEST / name, PUBLIC / name)
    cohorts = []
    for gpu in [0, 1]:
        for name in [f'gate-worker{gpu}.json', f'timing-worker{gpu}.json', f'gate-gpu{gpu}.json', f'timing-gpu{gpu}.json', f'pairs-gpu{gpu}.json']:
            shutil.copyfile(DEST / name, PUBLIC / name)
    for folder in ['source-gates', 'warmup', 'runs']:
        for source in sorted((DEST / folder).glob('*/result.json')):
            relative = source.relative_to(DEST)
            target = PUBLIC / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            cohorts.append({'file': str(relative), 'sha256': S.sha(source)})
    # Preserve every original byte/hash while avoiding thousands of tiny public
    # files. This changes packaging only, after the scientific run is sealed.
    marker_bytes = {str(q.relative_to(DEST)): q.read_text()
                    for q in sorted((DEST / 'barriers').rglob('*.json'))}
    S.save(PUBLIC / 'barrier-markers.json', {'encoding': 'original UTF-8 text, byte-exact',
                                         'markers': marker_bytes})
    records = []
    timed_rows = {}
    for pair in range(4):
        for mode in ['LP96', 'BP_geometry_LP96']:
            folder = DEST / 'runs' / f'pair{pair}-{mode}'
            receipt = json.loads((folder / 'result.json').read_text())
            timed_rows[pair, mode] = {(r['job_id'], r['batch']): r for r in receipt['rows']}
    @lru_cache(maxsize=8)
    def final_arrays(path, digest):
        assert S.sha(path) == digest
        with np.load(path, allow_pickle=False) as z:
            return {key: z[key] for key in z.files if key.startswith('answer_')}
    for input_index, seed in sorted(selected):
        shard, local = divmod(input_index, 256)
        bi, within = divmod(local, 32)
        job = f'shard{shard:02d}-s{seed}'
        source_record = source_map[input_index, seed]
        source_path = ORIGINAL / source_record['file']
        assert S.sha(source_path) == source_record['sha256']
        arrays = {}
        for mode, prefix in [('LP96', 'control'), ('BP_geometry_LP96', 'candidate')]:
            fields = {}
            file_receipts = []
            for pair in range(4):
                folder = DEST / 'runs' / f'pair{pair}-{mode}'
                row = timed_rows[pair, mode][job, bi]
                path = folder / row['file']
                for key, values in final_arrays(path, row['file_sha256']).items():
                    fields.setdefault(key, []).append(values[within])
                file_receipts.append({'pair': pair, 'batch_file': str(path.relative_to(DEST)),
                                      'batch_sha256': row['file_sha256']})
            for key, values in fields.items():
                arrays[prefix + '_' + key] = np.stack(values)
            arrays[prefix + '_provenance_utf8'] = np.frombuffer(json.dumps(file_receipts).encode(), dtype=np.uint8)
        filename = f'i{input_index}-s{seed}.npz'
        Z.save(fixtures / filename, **arrays)
        records.append({'file': filename, 'sha256': S.sha(fixtures / filename),
                        'input_index': input_index, 'model_seed': seed,
                        'original_fixture': source_record['file'], 'original_fixture_sha256': source_record['sha256'],
                        'fixed_control': (input_index, seed) in fixed,
                        'actually_routed': (input_index, seed) in routed,
                        'candidate_failure': (input_index, seed) in failures})
    S.save(fixtures / 'manifest.json', {'protocol_sha256': S.sha(DEST / 'protocol.json'),
        'audit_sha256': S.sha(DEST / 'audit.json'), 'source_fixture_manifest_sha256': S.sha(ORIGINAL / 'manifest.json'),
        'source_fixtures_public_relative': '../../cross-certificate-screen/public-fixtures',
        'fixed_controls': len(fixed), 'complete_routed_cases': len(routed),
        'complete_candidate_failure_cases': len(failures), 'records': records,
        'selection': 'Union of first input per shard/model, every actual geometry route and every candidate failure. Four pairs and both actual arms per selected case.',
        'scope': 'Selected complete actual final boards/counters/certificates bound to existing original source coupling fixtures. Complete internal trajectory/source parity was independently checked on Work, not reconstructed by these final-state fixtures.'})
    S.save(PUBLIC / 'cohort-receipts.json', {'receipts': cohorts, 'extractor_sha256': S.sha(Path(__file__)),
        'extraction_seconds': time.perf_counter() - started, 'final_array_cache': final_arrays.cache_info()._asdict(),
        'barrier_marker_count': len(marker_bytes),
        'packaging_scope': 'Post-run I/O only. Cache holds at most eight final-answer batches; original barrier bytes are consolidated without changing evidence. Not an inference speed claim.'})
    print(json.dumps({'status': 'extracted', 'fixture_cases': len(records), 'routed_cases': len(routed),
                      'candidate_failures': len(failures), 'cohort_receipts': len(cohorts)}, indent=2))


if __name__ == '__main__':
    extract()
