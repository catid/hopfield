"""Count estimator: no Sudoku geometry and no hidden completion access."""
import argparse
import hashlib
import json
import numpy as np
from run import ROOT, HERE, save, sha, now

OUT = ROOT/'artifacts/research/20261114/ripper/clues'


def estimate(values, given):
    assert values.shape == given.shape and values.shape[1] == 81
    i,j = np.triu_indices(81,1)
    both = given[:,i] & given[:,j]
    equal = (values[:,i] == values[:,j]) & both
    return ~equal.any(0), equal.sum(0).astype(np.int32), both.sum(0).astype(np.int32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', choices=['clues','full'], required=True); a = ap.parse_args()
    p = json.loads((HERE/'protocol-clues.json').read_text()); manifest = json.loads((OUT/'input_manifest.json').read_text())
    assert manifest['protocol_sha256'] == sha(HERE/'protocol-clues.json')
    rows = []
    for e in manifest['inputs']:
        seed = e['seed']; key = 'clues' if a.arm == 'clues' else 'completion'
        src = ROOT/e[key+'_path']; assert sha(src) == e[key+'_sha256']
        z = np.load(src)
        if a.arm == 'clues':
            assert set(z.files) == {'values','given','source_row'}
            values, given = z['values'], z['given']; assert np.all(values[~given] == -1)
        else:
            assert set(z.files) == {'solutions','source_row'}
            values = z['solutions']; given = np.ones_like(values, bool)
        previous = None
        for n in p['sizes']:
            support, equal, together = estimate(values[:n], given[:n])
            if previous is not None: assert np.all(~support | previous)
            previous = support
            checks = {}
            if a.arm == 'clues':
                rng = np.random.default_rng(20261118+10000*seed+n)
                for label in ['sentinel','random','shuffled_observed']:
                    v = values[:n].copy(); hidden = ~given[:n]
                    if label == 'sentinel': v[hidden] = -99
                    elif label == 'random': v[hidden] = rng.integers(0,9,hidden.sum())
                    else: v[hidden] = rng.permutation(np.resize(v[given[:n]], hidden.sum()))
                    s,q,t = estimate(v, given[:n])
                    assert np.array_equal(s,support) and np.array_equal(q,equal) and np.array_equal(t,together)
                    checks[label] = True
                empty,_,_ = estimate(values[:n], np.zeros_like(given[:n]))
                assert empty.all(); checks['empty_mask_all_pairs'] = True
            dest = OUT/f'models/{a.arm}-n{n}-s{seed}.npz'; dest.parent.mkdir(exist_ok=True)
            assert not dest.exists(); np.savez_compressed(dest, support=support, equal_count=equal, jointly_given_count=together)
            rows.append({'seed':seed,'n':n,'arm':a.arm,'path':str(dest.relative_to(ROOT)),
                'sha256':sha(dest),'support_bytes_sha256':hashlib.sha256(support.tobytes()).hexdigest(),
                'retained_edges':int(support.sum()),'input_sha256':sha(src),'invariance_checks':checks})
    save(OUT/f'{a.arm}_models.json', {'status':'fit_without_geometry','utc':now(),
        'protocol_sha256':sha(HERE/'protocol-clues.json'),'wrapper_sha256':sha(__file__),
        'input_manifest_sha256':sha(OUT/'input_manifest.json'),'rows':rows})
    print(json.dumps({'status':'fit_without_geometry','arm':a.arm,'models':len(rows),'all_invariance_checks_passed':True}))


if __name__ == '__main__': main()
