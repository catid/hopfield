"""Public-only H4w support, outcome, uncertainty and exact-proof checks.

No raw data, optimizer or search imports. The statistical implementation is
separate from the study's batched matrix implementation.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from verify_sudoku_soundness import check_trace

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ROOT/'results/research/foureyes/canonical_study'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    s = json.loads((PUBLIC/'summary.json').read_text())
    cert = json.loads((PUBLIC/'certificates.json').read_text())
    protocol = ROOT/'research/20261114/foureyes/canonical_study.json'
    assert sha(protocol)==s['protocol_sha256']
    assert sha(PUBLIC/'summary.json')==cert['summary_sha256']
    assert sha(PUBLIC/'models.npz')==cert['models_sha256']
    assert sha(PUBLIC/'audited-outcome-bits.npz')==s['outcome_bits_sha256']
    for path,digest in cert['source_gate_exports'].items():
        assert sha(ROOT/path)==digest
    with np.load(PUBLIC/'models.npz',allow_pickle=False) as data:
        models = {k:data[k] for k in data.files}
    with np.load(PUBLIC/'audited-outcome-bits.npz',allow_pickle=False) as data:
        good = np.unpackbits(data['valid'],axis=1,count=10000).astype(np.int8)
        stopped = np.unpackbits(data['stopped'],axis=1,count=10000).astype(bool)
        indexes = {str(k):i for i,k in enumerate(data['job_ids'])}
    assert good.shape==stopped.shape==(148,10000)
    ii,jj = np.triu_indices(81,1)
    truth = (ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3))
    lookup = {}
    structural = {(r['bank'],r['arm']):r for r in s['source_audit']['models']}
    witnesses = {w['job']:w for w in cert['witnesses']}
    count_steps = 0
    for m,o,c in zip(s['model_mappings'],s['outcomes'],cert['rows'],strict=True):
        bank,arm = m['bank'],m['arm'];support = models[arm+'_support'][bank]
        assert support.shape==(3240,) and support.dtype==np.bool_
        assert hashlib.sha256(support.tobytes()).hexdigest()==m['support_sha256']
        if arm.startswith('canonical_'):
            assert np.array_equal(support,models[arm+'_coefficients'][bank]>0)
        elif arm=='adamw':
            j = models['adamw_coefficients'][bank];p = np.sort(j[j>0]);k = np.diff(p).argmax()
            assert np.array_equal(support,j>(p[k]+p[k+1])/2)
        else:
            counts = models['counts'][bank];v = np.unique(counts)
            eligible = [k for k in range(len(v)-1) if v[k+1]<=np.median(counts)]
            k = max(eligible,key=lambda k:int(v[k+1]-v[k])) if eligible else None
            threshold = (v[k]+v[k+1])//2 if k is not None else 0
            assert np.array_equal(support,counts<=threshold)
        struct = structural[bank,arm]
        assert (int(support.sum()),int((truth&~support).sum()),int((~truth&support).sum()))==(struct['retained'],struct['missing'],struct['extra'])
        i = indexes[m['actual']];lookup[bank,arm] = good[i]
        assert abs(good[i].mean()*100-o['accuracy_pct'])<1e-12
        assert int((stopped[i]&~good[i].astype(bool)).sum())==o['wrong_stops']==c['observed_wrong_stops']
        assert (bank,arm,m['actual'])==(c['bank'],c['arm'],c['job'])
        proved,steps,units = check_trace(support,c['trace']);count_steps += steps
        assert sorted(proved)==c['certified_units'] and (len(proved)==27)==c['sound']
        if c['sound']:
            assert not o['wrong_stops']
        elif 'witness_job' in c:
            w = witnesses[c['witness_job']]
            board,solution,free = np.asarray(w['board']),np.asarray(w['solution']),np.asarray(w['free'])
            assert board.dtype.kind in 'iu' and solution.dtype.kind in 'iu'
            assert board.shape==solution.shape==free.shape==(81,) and free.dtype==np.bool_
            assert ((board>=0)&(board<9)).all() and np.array_equal(board[~free],solution[~free])
            assert all(set(solution[u])==set(range(9)) for u in units)
            assert any(len(set(board[u]))!=9 for u in units)
            assert np.all(board[ii[support]]!=board[jj[support]])
            assert w['puzzle_index']==390000+w['local_index']
            assert s['input_sha256'][w['source_archive']]==w['source_sha256']
    assert len(lookup)==256
    for group in s['grouped']:
        arm = group['arm'];rows = [r for r in s['outcomes'] if r['arm']==arm]
        assert abs(np.stack([lookup[b,arm] for b in range(64)]).mean()*100-group['accuracy_pct'])<1e-12
        assert sum(r['wrong_stops'] for r in rows)==group['wrong_stops']
        structures = [structural[b,arm] for b in range(64)]
        assert sum(r['missing']==0 for r in structures)==group['all_true_retained']
        assert sum(r['missing']==r['extra']==0 for r in structures)==group['exact_graphs']
    for group in cert['grouped']:
        rows = [r for r in cert['rows'] if r['arm']==group['arm']]
        assert sum(r['sound'] for r in rows)==group['sound']
        assert sum('witness_job' in r for r in rows)==group['unsound_witness']
        assert sum(not r['sound'] and 'witness_job' not in r for r in rows)==group['unresolved']
    delta = np.stack([lookup[b,'canonical_half']-lookup[b,'restricted'] for b in range(64)])
    rng = np.random.default_rng(20261115999);draws = []
    for draw in range(5000):
        banks = rng.integers(0,64,64);inputs = rng.integers(0,10000,10000)
        sums = delta[banks].sum(axis=0,dtype=np.int64)
        value = 100.*sums[inputs].sum()/(64*10000)
        if draw<16:
            assert value==100.*delta[banks[:,None],inputs].sum()/(64*10000)
        draws.append(value)
    interval = np.quantile(draws,[.025,.975])
    assert np.array_equal(interval,s['primary']['paired_two_way95_interval_pp'])
    assert abs(delta.mean()*100-s['primary']['gain_pp'])<1e-12
    wrong = sum(r['wrong_stops'] for r in s['outcomes'] if r['arm']=='canonical_half')
    assert wrong==s['primary']['wrong_stops']
    decision = 'falsified' if wrong or interval[1]<-2 else 'supported' if interval[0]>=-2 else 'inconclusive'
    assert decision==s['primary']['decision']
    print(json.dumps(dict(status='passed',logical_cells=256,unique_searches=148,exact_count_identity_steps=count_steps,
        explicit_witnesses=len(witnesses),interval_pp=interval.tolist(),decision=decision)))


if __name__=='__main__':
    main()
