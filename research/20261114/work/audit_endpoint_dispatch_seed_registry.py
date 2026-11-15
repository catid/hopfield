"""H2br exact seed reservation and complete prior registry; no draws."""
import json
import study as S

BASE = 2026111532
DEST = S.OUT / 'bp_first_endpoint_dispatch'


def main():
    DEST.mkdir(exist_ok=True)
    path = DEST / 'seed-registry.json'
    assert not path.exists()
    protocols, hits = [], []
    for source in sorted(S.HERE.glob('*protocol*.json')):
        data = json.loads(source.read_text())
        found = str(BASE) in json.dumps(data)
        protocols.append({'path': str(source.relative_to(S.ROOT)), 'sha256': S.sha(source),
                          'contains_reserved_seed_base': found})
        if found:
            hits.append(str(source))
    assert not hits, hits
    manifests = []
    for source in sorted([S.OUT / 'datasets.json', *S.OUT.glob('*/datasets.json')]):
        if source.is_file() and source.parent != DEST:
            data = json.loads(source.read_text())
            if 'datasets' in data:
                manifests.append({'path': str(source.relative_to(S.ROOT)), 'sha256': S.sha(source),
                                  'dataset_records': len(data['datasets'])})
    result = {'status': 'passed', 'reserved_seed_base': BASE, 'existing_protocol_references': 0,
              'protocol_registry': protocols, 'prior_data_manifests': manifests,
              'fresh_inputs_drawn': 0, 'source_sha256': S.sha(__file__),
              'scope': 'Exact preapproved seed reservation, no seed-outcome search or fresh data generation. Every extant Work protocol and prior dataset manifest is hash-bound for later complete nonoverlap checks.'}
    S.save(path, result)
    print(json.dumps({'status': 'passed', 'base': BASE, 'protocols': len(protocols),
                      'prior_data_manifests': len(manifests), 'fresh_inputs_drawn': 0}, indent=2))


if __name__ == '__main__':
    main()
