"""H1cb: exact truth gaps of all192 selected H1ca arms, without new fitting."""
from pathlib import Path
import argparse, datetime, json, os, platform, resource, signal, subprocess, sys, time
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
WALL_ORIGIN = time.monotonic()
import numpy as np
from run import ROOT, HERE, sha, save
from audit_degree_prior import check_certificate
PARENT = ROOT/'results/research/ripper/posterior_fresh_confirm'
OUT = ROOT/'artifacts/research/20261114/ripper/posterior_residual_gap'
PUBLIC = ROOT/'results/research/ripper/posterior_residual_gap'
PROTOCOL = HERE/'protocol-posterior-residual-gap.json'
ARMS = ['single270', 'hard_triple90', 'posterior_triple90']


def read(path):
    return json.loads(Path(path).read_text())


def truth_graph():
    ii, jj = np.triu_indices(81, 1)
    return (ii//9 == jj//9) | (ii % 9 == jj % 9) | ((ii//27 == jj//27) & (ii % 9//3 == jj % 9//3))


def evaluate(n, degree, cost, proof, truth):
    ii, jj = np.triu_indices(n, 1)
    assert truth.dtype == bool and truth.shape == cost.shape
    incidence = np.zeros(n, np.int64)
    np.add.at(incidence, ii, truth)
    np.add.at(incidence, jj, truth)
    assert (incidence == degree).all()
    assert check_certificate(n, degree, ii, jj, cost, *proof)
    optimum = sum(int(c)*int(x) for c, x in zip(cost, proof[0], strict=True))
    true_cost = 2*sum(int(c) for c in cost[truth])
    gap = true_cost-optimum
    assert gap >= 0
    exact = bool(np.array_equal(proof[0], 2*truth.astype(np.int64)))
    integral = bool(np.isin(proof[0], [0, 2]).all())
    if exact:
        assert gap == 0
    return dict(optimum_twice=optimum, truth_cost_twice=true_cost, gap_twice=gap,
                exact=exact, integral=integral,
                truth_objective_status='cooptimal' if gap == 0 else 'strictly_suboptimal')


def aggregate(rows):
    assert len(rows)==192 and [r['index'] for r in rows]==list(range(192))
    assert [r['arm'] for r in rows]==ARMS*64
    failed=[r for r in rows if r['arm']=='posterior_triple90' and not r['exact']]
    assert [r['seed'] for r in failed]==[6,7,57]
    strict=sum(r['gap_twice']>0 for r in failed)
    grouped=[]
    for arm in ARMS:
        part=[r for r in rows if r['arm']==arm]
        grouped.append(dict(arm=arm,total=len(part),exact=sum(r['exact'] for r in part),
                            strict=sum(r['gap_twice']>0 for r in part),cooptimal_nonexact=sum(r['gap_twice']==0 and not r['exact'] for r in part),
                            zero_gap=sum(r['gap_twice']==0 for r in part),selected_fractional=sum(not r['integral'] for r in part)))
    assert [r['exact'] for r in grouped]==[64,51,61]
    return dict(denominator=3,required_strict=2,strict_failures=strict,cooptimal_failures=3-strict,
                decision='supported' if strict>=2 else 'falsified',grouped=grouped,
                all192_retained=True,selected_certificates_rechecked=192,proof_mutations_rejected=576,new_LPs=0,new_labels=0,
                parent_physical_noisy_union=2073600,official_noisy_label_sequence=16760919)


def verify():
    p=read(PROTOCOL)
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==digest,rel
    return p


def freeze():
    assert not PROTOCOL.exists() and not PUBLIC.exists() and not OUT.exists()
    path=HERE/'posterior-residual-gap-preflight.json';gate=read(path)
    assert gate['status']=='passed' and gate['study_gaps_read'] is False
    for rel,digest in gate['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    parent=read(PARENT/'summary.json');design=read(HERE/'posterior-residual-gap-proposal.json')
    assert sha(PARENT/'summary.json')==design['parent_summary_sha256'] and sha(HERE/'protocol-posterior-fresh-confirm.json')==design['parent_protocol_sha256']
    assert len(parent['rows'])==192 and all(r['selected_degree']==20 and r['classification']!='unresolved' for r in parent['rows'])
    assert [r['success'] for r in parent['grouped']]==[64,51,61]
    assert [r['seed'] for r in parent['rows'] if r['arm']=='posterior_triple90' and r['classification']!='success']==[6,7,57]
    inputs=[path,HERE/'posterior-residual-gap-proposal.json',HERE/'protocol-posterior-fresh-confirm.json',PARENT/'summary.json',
            ROOT/'results/research/ripper/posterior_fresh_confirm_public_manifest.json']
    for seed in range(64):
        inputs += [PARENT/f's{seed}/{arm}-{suffix}' for arm in ARMS for suffix in ['models.npz','receipt.json']]
        inputs += [PARENT/f's{seed}/query-records.npz',PARENT/f's{seed}/query-record-audit.json']
    inputs += [PARENT/'query-record-seal.json']
    p=dict(design,status='frozen_before_truth_objective_gap_calculation',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           source_sha256=gate['source_sha256'],input_sha256={str(path.relative_to(ROOT)):sha(path) for path in inputs})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL),input_files=len(inputs))),flush=True)


def bank(seed,p):
    from posterior_fresh_confirm_study import arm_score
    directory=OUT/f's{seed}';directory.mkdir()
    prior=read(PARENT/'summary.json')['rows'];truth=truth_graph()
    with np.load(PARENT/f's{seed}/query-records.npz',allow_pickle=False) as z:queries={k:z[k] for k in z.files}
    rows=[]
    for ai,arm in enumerate(ARMS):
        index=3*seed+ai;old=prior[index]
        arrays=arm_score(queries,arm);cost=arrays['cost']
        with np.load(PARENT/f's{seed}/{arm}-models.npz',allow_pickle=False) as z:
            assert np.array_equal(cost,z['cost'])
            proof=[z['d20_'+key] for key in ['primal_twice','node_dual_twice','upper_dual_twice']]
        result=evaluate(81,20,cost,proof,truth)
        assert result['exact']==(old['classification']=='success')
        rows.append(dict(index=index,seed=seed,arm=arm,degree=20,parent_classification=old['classification'],**result))
    save(directory/'rows.json',rows)
    from audit_posterior_residual_gap import audit_bank
    checked=audit_bank(seed,False,p)
    save(directory/'audit.json',checked)
    print(json.dumps(dict(seed=seed,complete=True)),flush=True)


def worker(worker_id):
    assert 0<=worker_id<8;p=verify()
    for seed in range(worker_id,64,8):bank(seed,p)


def summary():
    import shutil
    p=verify();assert not PUBLIC.exists();PUBLIC.mkdir(parents=True)
    rows=[];audits=[];files={}
    for seed in range(64):
        directory=OUT/f's{seed}';audit=read(directory/'audit.json')
        assert audit['status']=='validated' and audit['seed']==seed and audit['protocol_sha256']==sha(PROTOCOL)
        assert audit['auditor_sha256']==sha(HERE/'audit_posterior_residual_gap.py') and audit['rows_sha256']==sha(directory/'rows.json')
        rows+=read(directory/'rows.json');audits.append(audit);target=PUBLIC/f's{seed}';target.mkdir()
        for name in ['rows.json','audit.json']:
            shutil.copyfile(directory/name,target/name);files[f's{seed}/{name}']=sha(target/name)
    assert sum(a['mutations_rejected'] for a in audits)==576
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),rows=rows,**aggregate(rows),artifact_sha256=files,scope=p['scope'])
    save(OUT/'audit.json',result);save(PUBLIC/'summary.json',result)
    print(json.dumps(dict(decision=result['decision'],strict_failures=result['strict_failures'],grouped=result['grouped'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','worker','summary']);parser.add_argument('--seed',type=int);args=parser.parse_args()
    worker(args.seed) if args.action=='worker' else globals()[args.action]()
