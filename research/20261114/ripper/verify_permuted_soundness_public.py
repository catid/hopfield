"""Public-only independent replay of every H1aq proof, witness and decision."""
from pathlib import Path
import copy,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
PUBLIC=ROOT/'results/research/ripper'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    sys.path.insert(0,str(HERE.parent/'foureyes'));from verify_sudoku_soundness import check_trace
    summary=json.loads((PUBLIC/'permuted_soundness.json').read_text());cert=json.loads((PUBLIC/'permuted_soundness_certificates.json').read_text());parent=json.loads((PUBLIC/'permuted_noise.json').read_text());protocol=json.loads((HERE/'protocol-permuted-soundness.json').read_text())
    assert summary['status']=='validated'and summary['protocol_sha256']==cert['protocol_sha256']==sha(HERE/'protocol-permuted-soundness.json')
    assert summary['certificates_sha256']==sha(PUBLIC/'permuted_soundness_certificates.json')
    assert cert['parent_audit_sha256']==sha(PUBLIC/'permuted_noise.json')
    assert summary['supports_sha256']==cert['supports_sha256']==sha(PUBLIC/'permuted_soundness_supports.npz')
    for p,d in protocol['code_sha256'].items():assert sha(ROOT/p)==d
    with np.load(PUBLIC/'permuted_soundness_supports.npz',allow_pickle=False)as z:boards=z['supports'];assert np.array_equal(z['banks'],np.arange(32))
    assert boards.shape==(32,3240)and boards.dtype==bool
    pairs=[(i,j)for i in range(81)for j in range(i+1,81)];truth=np.array([i//9==j//9 or i%9==j%9 or(i//27==j//27 and i%9//3==j%9//3)for i,j in pairs]);counts={'sound_proof':0,'unsound_witness':0,'unresolved':0};steps=mutations=0;identities=set()
    for bank,(r,support,publicrow)in enumerate(zip(cert['rows'],boards,summary['rows'],strict=True)):
        assert r['bank']==publicrow['bank']==bank
        assert r['support_sha256']==hashlib.sha256(support.tobytes()).hexdigest();identities.add(r['support_sha256'])
        prior=next(a for a in parent['acquisition_audits']if a['seed']==bank)
        assert prior['support_bytes_sha256']==r['support_sha256']
        assert r['fit_sha256']==protocol['input_sha256'][f'artifacts/research/20261114/ripper/permuted_noise/s{bank}/fit.npz']
        proved,n,units=check_trace(support,r['trace']);steps+=n
        assert sorted(proved)==r['certified_units']and r['sound']==(len(proved)==27)
        assert r['missing_true_edges']==prior['missing_true_edges']==publicrow['missing_true_edges']==int((truth&~support).sum())
        assert r['extra_edges']==prior['extra_edges']==publicrow['extra_edges']==int((~truth&support).sum())
        assert publicrow['certified_units']==len(proved)and publicrow['count_identity_steps']==n
        if r['sound']:classification='sound_proof';assert r['counterexample_search']is None
        else:
            search=r['counterexample_search'];assert search['status']in ['explicit_counterexample','unresolved_budget','no_witness_in_complete_search']
            classification='unresolved'
            if search['status']=='explicit_counterexample':
                board=np.asarray(search['board']);assert board.shape==(81,)and board.dtype.kind in 'iu'and((board>=0)&(board<9)).all()
                assert all(board[i]!=board[j]for(i,j),keep in zip(pairs,support)if keep)
                assert any(len(set(board[cells]))!=9 for cells in units)
                a,b=search['forced_equal_pair'];index=pairs.index((a,b));assert truth[index]and not support[index]and board[a]==board[b]
                classification='unsound_witness'
        assert classification==r['classification']==publicrow['classification'];counts[classification]+=1
        if n:
            broken=copy.deepcopy(r['trace']);next(s for s in broken if s['rule']=='count_identity')['source'].pop()
            try:check_trace(support,broken)
            except AssertionError:mutations+=1
            else:raise AssertionError('invalid proof accepted')
    assert sum(counts.values())==32 and counts==summary['counts']
    assert len(identities)==summary['unique_supports']==cert['unique_supports']and steps==summary['count_identity_steps']and mutations==summary['rejected_mutated_proofs']
    decision='supported'if counts['sound_proof']>=29 else'falsified'if counts['unsound_witness']>=4 else'inconclusive';assert decision==summary['primary']['decision']
    print(json.dumps(dict(status='passed',counts=counts,decision=decision,count_identity_steps=steps,rejected_mutations=mutations,public_only=True)))
if __name__=='__main__':main()
