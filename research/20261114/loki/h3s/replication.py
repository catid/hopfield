"""Deterministic record-duplication invariance test and deduplicated derivative."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
import signal
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3s';OUT=ROOT/'artifacts/research/20261114/loki/h3s';PUBLIC=ROOT/'results/research/loki/h3s';BASE=ROOT/'artifacts/research/20261114/loki/h3r'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def freeze():
    assert not(SRC/'protocol.json').exists()
    files=[Path(__file__),BASE/'training.npz',BASE/'source-support.npz',BASE/'source-fits.json',ROOT/'research/20261114/loki/h3r/protocol.json',*[BASE/f'models/{a}/model.pt' for a in ['n14-full','exposure-filtered','true-graph-oracle']]]
    p=dict(id='H3s-record-replication-invariance',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),copies=[1,2,4],unique_source_patterns=5180,
        primary_hypothesis='Raw-record exposure fitting preserves every N15 target coefficient when identical source records are duplicated2x or4x.',
        decision='Support the finite invariance claim iff S and B and q/delta/tau/unit match across all declared multiplicities; falsify if any target coefficient or solver parameter differs. Deterministic equality, not a statistical test.',
        derivative='For each multiplicity explicitly deduplicate complete source patterns before calculating marginals and N. Require exactly5180 distinct patterns and test the same coefficient/parameter invariance. This is a named derivative, not a retroactive change to H3r.',
        exposure='Use exactly H3r E=sum(m_i*m_j)/N and threshold ln(364/.01), unchanged homogeneous lines and target settings. Repeated copies are not independent information.',
        provenance='The multiplicity sensitivity was analytically anticipated after H3r exposure scores were known. Scope is frozen before this deterministic replication audit; it is not an out-of-sample statistical discovery.',
        outcomes='No new GPU searches. If every constructed model exactly matches an H3r model, map its already fixed cohort/seed outcomes with explicit identity receipts; mappings are explanatory reuse, not independent replication.',
        budget=dict(cpu_wall_seconds=180,threads=1,gpu_seconds=0),input_sha256={str(path.relative_to(ROOT)):sha(path) for path in files},
        limits=['Invariance over copies1/2/4 of one finite complete source bank, not arbitrary noisy or near-duplicate inputs.',
            'The independence exposure heuristic has no calibrated significance or Bonferroni guarantee for constrained patterns.',
            'Deduplication fixes repeated-record counting but does not prove general transfer or source-bank robustness.'])
    save(SRC/'protocol.json',p);print(json.dumps(dict(status='frozen',protocol_sha256=sha(SRC/'protocol.json'))))
def fit(labels,lines):
    N=len(labels);m=labels.sum(0).astype(np.int64);co=labels.astype(np.int64).T@labels.astype(np.int64)
    offsets=[(r,c) for r in range(14) for c in range(-13,14) if r>0 or c>0];counts={o:0 for o in offsets};num={o:0 for o in offsets}
    # Literal placement aggregation, independent from H3r's bincount path.
    for a,b in itertools.combinations(range(196),2):
        o=(b//14-a//14,b%14-a%14);counts[o]+=int(co[a,b]);num[o]+=int(m[a])*int(m[b])
    threshold=math.log(len(offsets)/.01);expected={o:num[o]/N for o in offsets};kept={o:counts[o]==0 and expected[o]>=threshold for o in offsets}
    own_lines=[]
    for a,b in itertools.product(range(-2,3),repeat=2):
        if (a,b)==(0,0) or math.gcd(abs(a),abs(b))!=1 or a<0 or (a==0 and b<0):continue
        on=[v for (r,c),v in counts.items() if a*r+b*c==0]
        if sum(v==0 for v in on)>=3 and not any(v>0 for v in on):own_lines.append([a,b])
    assert own_lines==lines
    i,j=np.triu_indices(225,1);dr=j//15-i//15;dc=j%15-i%15;mask=np.array([kept.get((r,c),False) or any(a*r+b*c==0 for a,b in lines) for r,c in zip(dr,dc)])
    S=torch.zeros(225,225,2,2);S[i,j,1,1]=torch.tensor(15*mask-1,dtype=torch.float32);S[j,i,1,1]=S[i,j,1,1]
    model=dict(S=S,B=torch.zeros(225,2),q=15,delta=14.,tau=-105.,unit=25.909505208333332)
    return model,dict(records=N,unique_records=len(np.unique(labels,axis=0)),finite_zeros_retained=sum(kept.values()),threshold=threshold,forbidden_pairs=int(mask.sum())),counts,num,expected

def main():
    torch.set_num_threads(1);p=json.loads((SRC/'protocol.json').read_text());OUT.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    for path,digest in p['input_sha256'].items():assert sha(ROOT/path)==digest,path
    def timeout(*unused):raise TimeoutError('180-second CPU replication-audit envelope reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(180);start=time.perf_counter()
    labels=np.load(BASE/'training.npz')['n14_all'];assert len(labels)==len(np.unique(labels,axis=0))==5180
    linefit=json.loads((BASE/'source-fits.json').read_text());lines=linefit['retained_lines'];references={a:torch.load(BASE/f'models/{a}/model.pt',weights_only=False) for a in ['n14-full','exposure-filtered','true-graph-oracle']}
    rows=[];scores=[];models={};base_counts=None;base_num=None;base_exp=None;identity=[]
    equal=lambda a,b:all(torch.equal(a[k],b[k]) for k in ['S','B']) and all(a[k]==b[k] for k in ['q','delta','tau','unit'])
    for copies in p['copies']:
        duplicated=np.tile(labels,(copies,1))
        for variant,records in [('raw-records',duplicated),('source-deduplicated',np.unique(duplicated,axis=0))]:
            model,rec,counts,num,expected=fit(records,lines);models[variant,copies]=model
            if variant=='raw-records' and copies==1:base_counts=counts;base_num=num;base_exp=expected
            if variant=='raw-records':
                assert all(counts[o]==copies*base_counts[o] and num[o]==copies*copies*base_num[o] and expected[o]==copies*base_exp[o] for o in counts)
            else:assert counts==base_counts and num==base_num and expected==base_exp
            matches=[a for a,m in references.items() if equal(model,m)];assert matches
            reference='exposure-filtered' if 'exposure-filtered' in matches else 'n14-full';identity.append(dict(variant=variant,copies=copies,reference_arm=reference,exact_matches=matches))
            path=OUT/f'{variant}-x{copies}.pt';torch.save(model,path)
            rows.append(dict(variant=variant,copies=copies,**rec,reference_arm=reference,unchanged_from_x1=equal(model,models[variant,1]),model_sha256=sha(path)))
            for o in [(12,-13),(12,13),(13,-12),(13,12),(13,-13),(13,13)]:
                scores.append(dict(variant=variant,copies=copies,dr=o[0],dc=o[1],expected_count=expected[o],threshold=rec['threshold'],finite_zero_retained=counts[o]==0 and expected[o]>=rec['threshold']))
    primary=all(r['unchanged_from_x1'] for r in rows if r['variant']=='raw-records');derivative=all(r['unchanged_from_x1'] for r in rows if r['variant']=='source-deduplicated')
    for variant in ['raw-records','source-deduplicated']:
        for copies in [2,4]:
            changed=(models[variant,copies]['S'][:,:,1,1]!=models[variant,1]['S'][:,:,1,1]).numpy();assert (changed==changed.T).all()
            next(r for r in rows if r['variant']==variant and r['copies']==copies)['changed_unordered_pairs']=int(np.triu(changed,1).sum())
        next(r for r in rows if r['variant']==variant and r['copies']==1)['changed_unordered_pairs']=0
    assert derivative
    identity_receipt=dict(status='exact_identity_verified_before_outcome_mapping',utc=dt.datetime.now(dt.timezone.utc).isoformat(),mappings=identity,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__))
    save(OUT/'identity-receipt.json',identity_receipt)
    prior_path=ROOT/'results/research/loki/h3r/summary.json';prior=json.loads(prior_path.read_text());assert prior['status']=='validated';mapped=[]
    for item in identity:
        result=next(r for r in prior['rows'] if r['arm']==item['reference_arm'])
        mapped.append(dict(**item,original_accuracy_pct=result['original_accuracy_pct'],certificate_first_accuracy_pct=result['certificate_first_accuracy_pct'],coverage_pct=result['coverage_pct'],wrong_stops=result['wrong_stops'],independent_search=False))
    result=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),decision='supported' if primary else 'falsified',derivative_invariant=derivative,rows=rows,offset_scores=scores,
        identity_receipt=identity_receipt,mapped_H3r_outcomes=mapped,outcome_mapping_scope='Exact solver identity on the same H3r cohort/settings/seeds; no new search or independent replication.',
        source_sha256=sha(__file__),protocol_sha256=sha(SRC/'protocol.json'),input_sha256=p['input_sha256'],mapped_summary_sha256=sha(prior_path),seconds=time.perf_counter()-start,gpu_seconds=0,limits=p['limits'])
    assert result['seconds']<180;save(OUT/'summary.json',result);save(PUBLIC/'summary.json',result)
    for name,data in [('model_identity.csv',rows),('offset_exposure.csv',scores),('mapped_outcomes.csv',mapped)]:
        with (PUBLIC/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    signal.alarm(0);print(json.dumps(result))
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','audit']);args=ap.parse_args();freeze() if args.mode=='freeze' else main()
