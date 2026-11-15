"""Materialize frozen oracle interventions without rerunning acquisition."""
import hashlib
import json
import numpy as np
from run import ROOT, HERE, sha, save, now
from confidence_oracle_common import OUT, PROTOCOL, load_protocol
from confidence_oracle_policy import interventions


def main():
    p = load_protocol(require_models=False)
    assert not OUT.exists()
    OUT.mkdir()
    old = ROOT/'artifacts/research/20261114/ripper/confidence_budget'
    previous = json.loads((old/'audit.json').read_text())
    models = []
    for seed in p['seeds']:
        receipt_path = old/f's{seed}/receipt.json'
        assert sha(receipt_path) == previous['artifact_sha256'][str(receipt_path.relative_to(ROOT))]
        receipt = json.loads(receipt_path.read_text())
        entry = next(m for m in receipt['models'] if m['arm'] == 'evidence')
        source = ROOT/entry['path']
        assert sha(source) == entry['sha256']
        base = np.load(source)['weight16']
        base.setflags(write=False)
        before = hashlib.sha256(base.tobytes()).hexdigest()
        assigned = interventions(base)
        assert hashlib.sha256(base.tobytes()).hexdigest() == before
        dest = OUT/f's{seed}'
        dest.mkdir()
        for arm, weight in assigned.items():
            path = dest/f'{arm}-n1110.npz'
            np.savez_compressed(path, support=weight > 0, weight16=weight)
            models.append({'arm': arm, 'seed': seed, 'n': 1110,
                           'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                           'base_path': entry['path'], 'base_sha256': entry['sha256'],
                           'weight16_bytes_sha256': hashlib.sha256(weight.tobytes()).hexdigest(),
                           'support_bytes_sha256': hashlib.sha256((weight > 0).tobytes()).hexdigest()})
    save(OUT/'models.json', {'status': 'prepared', 'utc': now(), 'protocol_sha256': sha(PROTOCOL),
         'preparation_sha256': sha(__file__), 'policy_sha256': sha(HERE/'confidence_oracle_policy.py'),
         'new_queries': 0, 'new_source_banks': 0, 'models': models})
    print(json.dumps({'prepared_models': len(models), 'new_queries': 0}), flush=True)


if __name__ == '__main__':
    main()
