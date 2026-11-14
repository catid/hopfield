"""Extend the frozen raw banks and physically separate clues from completions."""
import csv
import json
import numpy as np
from run import ROOT, HERE, save, sha, now

OUT = ROOT / 'artifacts/research/20261114/ripper/clues'


def main():
    p = json.loads((HERE/'protocol-clues.json').read_text())
    assert np.__version__ == '2.3.4'
    src = ROOT/p['source']; assert sha(src) == p['source_sha256']
    prior = ROOT/p['prior_manifest']; assert sha(prior) == p['prior_manifest_sha256']
    old = json.loads(prior.read_text())
    assert old['source_rows'] == old['unique_complete_boards'] == p['pool_size']
    OUT.mkdir(exist_ok=False)
    banks = {}; wanted = {}
    for seed in p['seeds']:
        rows = np.random.default_rng(20261116+seed).permutation(p['pool_size'])[:3000]
        banks[seed] = {'source_row': rows, 'values': np.full((3000,81), -1, np.int8),
                       'given': np.zeros((3000,81), bool), 'solutions': np.zeros((3000,81), np.int8)}
        for i, row in enumerate(rows): wanted.setdefault(int(row), []).append((seed,i))
    remaining = set(wanted)
    with src.open(newline='') as f:
        for row_index, row in enumerate(csv.DictReader(f)):
            if row_index not in wanted: continue
            answer = np.frombuffer(row['answer'].encode('ascii'), np.uint8).astype(np.int8)-ord('1')
            observed = np.frombuffer(row['question'].replace('.','0').encode('ascii'), np.uint8).astype(np.int8)-ord('1')
            given = observed >= 0
            assert len(answer) == len(observed) == 81 and np.all((answer >= 0)&(answer < 9))
            assert np.array_equal(answer[given], observed[given])
            for seed,i in wanted[row_index]:
                d = banks[seed]; d['solutions'][i] = answer; d['values'][i] = observed; d['given'][i] = given
            remaining.remove(row_index)
            if not remaining: break
    assert not remaining
    entries = []
    for seed,d in banks.items():
        e = next(e for e in old['inputs'] if e['seed'] == seed)
        assert sha(ROOT/e['path']) == e['sha256']
        z = np.load(ROOT/e['path'])
        assert np.array_equal(d['source_row'][:1000], z['source_row'])
        assert np.array_equal(d['source_row'][:1000], z['unique_pool_index'])
        assert np.array_equal(d['solutions'][:1000], z['solutions'])
        assert np.array_equal(~d['given'][:1000], z['free'])
        b = d['solutions'].reshape(-1,9,9)
        boxes = b.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
        for digit in range(9):
            assert np.all((b == digit).sum(1)==1) and np.all((b == digit).sum(2)==1)
            assert np.all((boxes == digit).sum(2)==1)
        a = OUT/f'clue_inputs/s{seed}.npz'; a.parent.mkdir(exist_ok=True)
        c = OUT/f'completion_controls/s{seed}.npz'; c.parent.mkdir(exist_ok=True)
        np.savez_compressed(a, values=d['values'], given=d['given'], source_row=d['source_row'])
        np.savez_compressed(c, solutions=d['solutions'], source_row=d['source_row'])
        entries.append({'seed':seed, 'n':3000, 'clues_path':str(a.relative_to(ROOT)),
            'clues_sha256':sha(a), 'completion_path':str(c.relative_to(ROOT)), 'completion_sha256':sha(c),
            'prior_prefix_verified':True, 'given_per_board':{'min':int(d['given'].sum(1).min()),
            'max':int(d['given'].sum(1).max()), 'mean':float(d['given'].sum(1).mean())}})
    overlaps = [{'seed_a':a,'seed_b':b,'shared_boards':int(len(np.intersect1d(banks[a]['source_row'], banks[b]['source_row'])))}
                for a in range(6) for b in range(a+1,6)]
    save(OUT/'input_manifest.json', {'status':'prepared', 'utc':now(), 'protocol_sha256':sha(HERE/'protocol-clues.json'),
         'wrapper_sha256':sha(__file__), 'source_sha256':sha(src), 'numpy':np.__version__,
         'prior_manifest_sha256':sha(prior), 'inputs':entries, 'cross_seed_overlap':overlaps,
         'support_inspection_performed':False})
    print(json.dumps({'status':'prepared','seeds':6,'rows_per_seed':3000,'overlaps':overlaps}))


if __name__ == '__main__': main()
