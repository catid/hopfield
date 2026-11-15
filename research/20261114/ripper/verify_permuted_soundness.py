"""Public graph proofs and explicit counterexamples, independently checked."""
import copy,csv,hashlib,json,sys
import numpy as np
from permuted_soundness import ROOT,HERE,FOUREYES,PARENT,OUT,PUBLIC,PROTOCOL,sha,save,verify

def audit():
    p=verify();sys.path.insert(0,str(FOUREYES));from verify_sudoku_soundness import check_trace
    c=json.loads((PUBLIC/'permuted_soundness_certificates.json').read_text());assert c['protocol_sha256']==sha(PROTOCOL)and c['producer_sha256']==sha(HERE/'permuted_soundness.py')
    assert c['parent_audit_sha256']==sha(PARENT/'audit.json')and c['supports_sha256']==sha(PUBLIC/'permuted_soundness_supports.npz')
    with np.load(PUBLIC/'permuted_soundness_supports.npz',allow_pickle=False)as z:
        supports=z['supports'];assert np.array_equal(z['banks'],np.arange(32))
    assert supports.shape==(32,3240)and supports.dtype==bool
    pairs=[(i,j)for i in range(81)for j in range(i+1,81)];rows=[];steps=mutations=0
    truth=np.array([i//9==j//9 or i%9==j%9 or(i//27==j//27 and i%9//3==j%9//3)for i,j in pairs])
    for r,support in zip(c['rows'],supports,strict=True):
        b=r['bank'];assert b==len(rows);assert r['support_sha256']==hashlib.sha256(support.tobytes()).hexdigest()and r['fit_sha256']==sha(PARENT/f's{b}'/'fit.npz')
        with np.load(PARENT/f's{b}'/'fit.npz',allow_pickle=False)as z:assert np.array_equal(z['support'],support)
        proved,nsteps,units=check_trace(support,r['trace']);steps+=nsteps
        assert sorted(proved)==r['certified_units']and r['sound']==(len(proved)==27)
        assert r['missing_true_edges']==int((truth&~support).sum())and r['extra_edges']==int((~truth&support).sum())
        if r['sound']:
            classification='sound_proof';assert r['counterexample_search']is None
        else:
            search=r['counterexample_search'];assert search['status']in ['explicit_counterexample','unresolved_budget','no_witness_in_complete_search']
            if search['status']=='explicit_counterexample':
                board=np.asarray(search['board']);assert board.shape==(81,)and board.dtype.kind in 'iu'and((board>=0)&(board<9)).all()
                assert all(board[i]!=board[j]for(i,j),keep in zip(pairs,support)if keep)
                assert any(len(set(board[cells]))!=9 for cells in units)
                a,b=search['forced_equal_pair'];assert board[a]==board[b]and truth[pairs.index((a,b))]and not support[pairs.index((a,b))]
                classification='unsound_witness'
            else:classification='unresolved'
        assert classification==r['classification']
        if nsteps:
            bad=copy.deepcopy(r['trace']);next(x for x in bad if x['rule']=='count_identity')['source'].pop()
            try:check_trace(support,bad)
            except AssertionError:mutations+=1
            else:raise AssertionError('invalid count identity accepted')
        rows.append(dict(bank=r['bank'],classification=classification,missing_true_edges=r['missing_true_edges'],extra_edges=r['extra_edges'],certified_units=len(proved),count_identity_steps=nsteps))
    assert len(rows)==32 and c['unique_supports']==len({r['support_sha256']for r in c['rows']})
    counts={k:sum(r['classification']==k for r in rows)for k in ['sound_proof','unsound_witness','unresolved']}
    primary=dict(target='At least29/32 supports imply Sudoku validity for every zero-energy9-coloring.',decision='supported'if counts['sound_proof']>=29 else'falsified'if counts['unsound_witness']>=4 else'inconclusive')
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),primary=primary,counts=counts,rows=rows,unique_supports=c['unique_supports'],count_identity_steps=steps,rejected_mutated_proofs=mutations,certificates_sha256=sha(PUBLIC/'permuted_soundness_certificates.json'),supports_sha256=c['supports_sha256'],new_queries=0,new_source_rows=0,new_solver_inputs=0,gpu_seconds=0,scope=p['scope'],geometry_role='Supplied only for exact audit; never used to modify supports or filter banks.')
    save(OUT/'audit.json',result);save(PUBLIC/'permuted_soundness.json',result)
    with (PUBLIC/'permuted_soundness_per_bank.csv').open('w',newline='')as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# H1aq: missing true edges versus global certificate soundness','',
        f'The target of at least 29 globally sound supports is **{primary["decision"]}**: {counts["sound_proof"]} proved sound, {counts["unsound_witness"]} explicitly unsound, and {counts["unresolved"]} unresolved out of all 32 H1ap fits.','',
        '| Audit category | Banks |','|---|---:|','| BP AdamW: outside this source-only study; not run | — |',
        f'| Complete independently replayed Sudoku-unit proof | {counts["sound_proof"]} |',f'| Explicit invalid zero-energy coloring | {counts["unsound_witness"]} |',f'| Unresolved | {counts["unresolved"]} |','',
        'All fixed H1ap supports are retained unchanged. Sudoku geometry is supplied only for this exact audit. A retained nine-clique forces all nine colors. Where certified units form two partitions of the same cells, digit-count identities can prove a final missing unit. Reaching all 27 units proves every zero-energy coloring over nine symbols is an ordinary valid Sudoku board, independent of any puzzle or clue set.','',
        f'The independent Counter-based verifier replayed {steps} count-identity steps and rejected {mutations} mutated proofs. This proof concerns the learned zero set; it does not assert that the graph equals the peer graph or that every capped solver return is valid.','',
        'For any unresolved support, the frozen deterministic exact-coloring search forces each missing true pair equal, with a shared two-CPU-second / 100,000-node ceiling per graph. Only a returned board independently satisfying every learned inequality and violating an ordinary Sudoku unit counts as unsound. Timeouts, incomplete closure and no-witness outcomes remain unresolved.','',
        'No learned fit, queried label, source board, puzzle row or GPU experiment was added or modified. This audit separates structural edge loss from the stronger claim of a false global certificate.','',
        '[Protocol](protocol-permuted-soundness.json) · [Proof and witness certificates](../../../results/research/ripper/permuted_soundness_certificates.json) · [Independent audit](verify_permuted_soundness.py) · [Results](../../../results/research/ripper/permuted_soundness.json)','']
    (HERE/'REPORT-permuted-soundness.md').write_text('\n'.join(lines));print(json.dumps(dict(primary=primary,counts=counts)))
if __name__=='__main__':audit()
