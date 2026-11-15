"""Public-only independent proof and witness replay for all H4ab supports."""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
from verify_sudoku_soundness import check_trace

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PUBLIC=ROOT/'results/research/foureyes/strong_support_soundness'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit():
    protocol=HERE/'strong_support_soundness.json';p=json.loads(protocol.read_text())
    for rel,h in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==h,rel
    c=json.loads((PUBLIC/'certificates.json').read_text())
    assert c['protocol_sha256']==sha(protocol)and c['producer_sha256']==sha(HERE/'strong_support_soundness.py')
    parent_path=ROOT/'results/research/foureyes/strong_support_study/summary.json'
    assert c['parent_sha256']==sha(parent_path);parent=json.loads(parent_path.read_text())
    assert sha(PUBLIC/'supports.npz')==c['supports_sha256']
    with np.load(PUBLIC/'supports.npz',allow_pickle=False)as z:
        assert np.array_equal(z['banks'],np.arange(64));supports=z['supports'].copy()
    assert supports.shape==(64,3240)and supports.dtype==bool
    with np.load(ROOT/'results/research/foureyes/canonical_confirm/models.npz',allow_pickle=False)as z:coeff=z['canonical_weak_coefficients']
    pairs=[(i,j)for i in range(81)for j in range(i+1,81)]
    true=np.array([i//9==j//9 or i%9==j%9 or(i//27==j//27 and i%9//3==j%9//3)for i,j in pairs])
    counts=dict(sound_proof=0,unsound_witness=0,unresolved=0);identities=set();rows=[];steps=mutations=0
    for bank,(r,support)in enumerate(zip(c['rows'],supports,strict=True)):
        assert r['bank']==bank
        j=coeff[bank];positive=sorted(float(x)for x in j if x>0);n=len(positive)
        cut=(positive[n//2]if n%2 else(positive[n//2-1]+positive[n//2])/2)/4
        rebuilt=np.array([x>0 and x>=cut for x in j]);assert np.array_equal(support,rebuilt)and cut==r['threshold']
        ident=hashlib.sha256(support.tobytes()).hexdigest();assert ident==r['support_sha256'];identities.add(ident)
        old=next(x for x in parent['rows']if x['bank']==bank and x['arm']=='strongonly')
        assert hashlib.sha256((support.astype(np.uint16)*256).tobytes()).hexdigest()==r['weight256_sha256']==old['weight256_sha256']
        proved,nsteps,units=check_trace(support,r['trace']);steps+=nsteps
        assert sorted(proved)==r['certified_units']and r['sound']==(len(proved)==27)
        assert r['missing_true']==int((true&~support).sum())and r['extra']==int((~true&support).sum())
        assert r['observed_wrong_stops']==old['wrong_stops']
        if r['sound']:
            classification='sound_proof';assert r['counterexample_search']is None and old['wrong_stops']==0
        else:
            search=r['counterexample_search'];assert search['status']in ['explicit_counterexample','unresolved_budget','no_witness_in_complete_search']
            classification='unresolved'
            if search['status']=='explicit_counterexample':
                board=np.asarray(search['board']);assert board.shape==(81,)and board.dtype.kind in 'iu'and((board>=0)&(board<9)).all()
                assert all(board[i]!=board[k]for(i,k),keep in zip(pairs,support)if keep)
                assert any(len(set(board[u]))!=9 for u in units)
                a,b=search['forced_equal_pair'];idx=pairs.index((a,b));assert true[idx]and not support[idx]and board[a]==board[b]
                classification='unsound_witness'
        assert classification==r['classification'];counts[classification]+=1
        if nsteps:
            broken=copy.deepcopy(r['trace']);next(x for x in broken if x['rule']=='count_identity')['source'].pop()
            try:check_trace(support,broken)
            except AssertionError:mutations+=1
            else:raise AssertionError('invalid proof accepted')
        rows.append(dict(bank=bank,classification=classification,certified_units=len(proved),count_identity_steps=nsteps,missing_true=r['missing_true'],extra=r['extra'],support_sha256=ident))
    assert sum(counts.values())==64 and len(identities)==c['unique_supports']
    decision='falsified'if counts['unsound_witness']else'supported'if counts['sound_proof']==64 else'inconclusive'
    result=dict(status='validated',protocol_sha256=sha(protocol),auditor_sha256=sha(__file__),certificates_sha256=sha(PUBLIC/'certificates.json'),supports_sha256=c['supports_sha256'],
                primary=dict(target='All64 graphs globally sound.',decision=decision),counts=counts,rows=rows,unique_supports=len(identities),count_identity_steps=steps,rejected_mutated_proofs=mutations,
                scope=p['scope'],geometry_role='Supplied for exact audit only; no graph modification or bank selection.',gpu_seconds=0,new_source_rows=0,new_solver_inputs=0)
    path=PUBLIC/'summary.json'
    if path.exists():assert json.loads(path.read_text())==result
    return result


if __name__=='__main__':
    r=audit();print(json.dumps(dict(status=r['status'],primary=r['primary'],counts=r['counts'],count_identity_steps=r['count_identity_steps'],rejected_mutations=r['rejected_mutated_proofs'],public_only=True)))
