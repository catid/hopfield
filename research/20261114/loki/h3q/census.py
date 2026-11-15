"""Exact post hoc certificate-capacity census; no model fitting or search."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3q';OUT=ROOT/'artifacts/research/20261114/loki/h3q';PUBLIC=ROOT/'results/research/loki/h3q'
BASE=ROOT/'artifacts/research/20261114/loki/h3o'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def freeze():
    assert not(SRC/'protocol.json').exists()
    files=[Path(__file__),BASE/'data.npz',BASE/'input-universe.npz',BASE/'solutions-n15.npy',BASE/'models/n14-full/model.pt',BASE/'models/true-graph-oracle/model.pt']
    p=dict(id='H3q-exact-population-certificate-census',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='descriptive_posthoc_scope_frozen_before_census',scope='All836069 unique valid1/2/3clue N15 superqueen inputs from all32516 valid complete solutions; no sampling and no new search.',
        question='What exact fraction of this finite input universe lacks any valid local certificate under the unchanged full-N14 hybrid, compared with the supplied oracle?',
        diagnostic_threshold='Report whether the exact no-valid-certificate input fraction exceeds0.1percent. This is a structural threshold, not the accuracy noninferiority test or a general fallback-decoder accuracy bound.',
        method='For each complete solution compute float64 single-cell margins. Any empty-cell deficit makes it uncertifiable under every given subset. Otherwise a1/2/3clue subset admits the solution exactly when it clamps every occupied-cell deficit. Enumerate every subset and accumulate per-input counts.',
        independent_validation='Recompute certificate counts for a frozen512-input diagnostic sample by literal occupied-pair fields and direct evaluation of every compatible solution under its unclamped cells. Rebuild full input completion counts and match the original exhaustive universe. Validate exact H3o/H3p cohort capacities after census without changing models.',
        sample_seed=2026112508,sample_size=512,budget=dict(cpu_wall_seconds=180,threads=1,gpu_seconds=0),
        limits=['Post hoc descriptive structural population diagnostic following H3o/H3p; no new accuracy evidence.',
            'Certificate existence, certificate correctness and finite-budget search success are distinct.',
            'Any conditional certificate-soundness argument assumes target q, valid givens and a forbidden graph containing all true attacks.',
            'Finite universe and fixed models; no claim about source-bank variation or arbitrary targets.'],
        input_sha256={str(path.relative_to(ROOT)):sha(path) for path in files})
    save(SRC/'protocol.json',p);print(json.dumps(dict(status='frozen',protocol_sha256=sha(SRC/'protocol.json'))))
def main():
    torch.set_num_threads(1);p=json.loads((SRC/'protocol.json').read_text());OUT.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    for path,digest in p['input_sha256'].items():assert sha(ROOT/path)==digest,path
    def timeout(*unused):raise TimeoutError('180-second CPU census envelope reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(180);start=time.perf_counter()
    solutions=np.load(BASE/'solutions-n15.npy');universe=dict(np.load(BASE/'input-universe.npz'));keys=universe['keys'];assert len(keys)==836069 and len(solutions)==32516
    positions=np.flatnonzero(solutions).reshape(len(solutions),15)%225;all_counts=np.zeros(226**3,np.int32);census={};diags=[]
    ids=np.arange(225);dr=np.abs(ids[:,None]//15-ids//15);dc=np.abs(ids[:,None]%15-ids%15)
    true=(dr==0)|(dc==0)|(dr==dc)|((dr==1)&(dc==2))|((dr==2)&(dc==1));np.fill_diagonal(true,False)
    for arm in ['n14-full','true-graph-oracle']:
        model=torch.load(BASE/f'models/{arm}/model.pt',weights_only=False);J=model['S'][:,:,1,1].numpy().astype(np.int64)
        assert model['q']==15 and model['delta']==14 and (J[true]==14).all()
        x=solutions.astype(np.float64);fields=x@J;deficit=(1-2*x)*fields<model['delta']-.001
        empty_bad=(deficit&~solutions).any(1);bad_counts=deficit.sum(1);cert=np.zeros(226**3,np.int32)
        for clues in [1,2,3]:
            subsets=np.array(list(itertools.combinations(range(15),clues)))
            for lo in range(0,len(solutions),512):
                hi=min(lo+512,len(solutions));cells=positions[lo:hi,None,:][:,:,subsets] if False else positions[lo:hi,subsets]
                encoded=sum((cells[:,:,a].astype(np.int64)+1)*226**(2-a) for a in range(clues))
                covered=deficit[lo:hi][np.arange(hi-lo)[:,None,None],cells].sum(2)==bad_counts[lo:hi,None]
                allowed=covered&~empty_bad[lo:hi,None]
                np.add.at(cert,encoded[allowed],1)
                if arm=='n14-full':np.add.at(all_counts,encoded.ravel(),1)
        assert (cert<=all_counts).all();census[arm]=cert[keys]
        diags.append(dict(arm=arm,solutions_with_empty_cell_deficits=int(empty_bad.sum()),occupied_deficit_histogram={str(v):int((bad_counts==v).sum()) for v in np.unique(bad_counts)}))
    assert np.array_equal(np.flatnonzero(all_counts),keys) and np.array_equal(all_counts[keys],universe['counts'])
    assert np.array_equal(census['true-graph-oracle'],universe['counts'])
    # Independent audit: occupied-pair field sums, then direct unclamped-margin checks over every compatible completion.
    audit_indices=np.random.default_rng(p['sample_seed']).permutation(len(keys))[:p['sample_size']]
    for arm in census:
        m=torch.load(BASE/f'models/{arm}/model.pt',weights_only=False);J=m['S'][:,:,1,1].numpy().astype(np.int64)
        literal=J[positions].sum(1);direct=(1-2*solutions.astype(np.int64))*literal
        assert np.array_equal(direct,(1-2*solutions.astype(float))*(solutions.astype(float)@J))
        for ix in audit_indices:
            key=int(keys[ix]);given=np.array([key//226**k%226-1 for k in [2,1,0] if key//226**k%226>0])
            compatible=solutions[:,given].all(1);margins=direct[compatible].copy();margins[:,given]=10**9
            assert int((margins>=m['delta']-.001).all(1).sum())==int(census[arm][ix])
    clues=(keys//226**2>0).astype(int)+(keys//226%226>0)+(keys%226>0);rows=[]
    for arm,counts in census.items():
        for clue in [0,1,2,3]:
            mask=np.ones(len(keys),bool) if clue==0 else clues==clue
            rows.append(dict(arm=arm,clues='all' if clue==0 else clue,inputs=int(mask.sum()),inputs_with_certificate=int((counts[mask]>0).sum()),inputs_without_certificate=int((counts[mask]==0).sum()),
                no_certificate_pct=100*float((counts[mask]==0).mean()),valid_completion_certificates=int(counts[mask].sum()),compatible_completions=int(universe['counts'][mask].sum())))
    cohorts=[]
    for study in ['h3o','h3p']:
        data=np.load(ROOT/f'artifacts/research/20261114/loki/{study}/data.npz');ix=np.searchsorted(keys,data['test_keys']);assert np.array_equal(keys[ix],data['test_keys'])
        summary=json.loads((ROOT/f'results/research/loki/{study}/summary.json').read_text())
        for arm,counts in census.items():
            record=next(r for r in summary['posthoc_valid_certificate_capacity'] if r['arm']==arm)
            assert record['inputs_with_any_valid_certificate']==int((counts[ix]>0).sum()) and record['valid_input_completion_certificates']==int(counts[ix].sum())
            cohorts.append(dict(study=study,arm=arm,inputs=len(ix),inputs_without_certificate=int((counts[ix]==0).sum()),valid_completion_certificates=int(counts[ix].sum())))
    np.savez_compressed(OUT/'per-input-capacity.npz',keys=keys,compatible_counts=universe['counts'],**census)
    primary=next(r for r in rows if r['arm']=='n14-full' and r['clues']=='all')
    result=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),scope=p['scope'],posthoc=True,rows=rows,source_solution_diagnostics=diags,
        structural_fraction_exceeds_0_1_percent=primary['no_certificate_pct']>.1,cohort_crosschecks=cohorts,independent_diagnostic_inputs=p['sample_size'],
        conditional_soundness='Every q=15 stopped state with valid givens is task-valid if the learned forbidden graph contains every true attack: any true conflict has a free occupied endpoint, whose removal cost is at most14-15<13.999 and therefore cannot stop. This concerns correctness conditional on stopping, not certificate availability or search success.',
        seconds=time.perf_counter()-start,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__),input_sha256=p['input_sha256'],
        capacity_archive_sha256=sha(OUT/'per-input-capacity.npz'),limits=p['limits'])
    assert result['seconds']<180
    save(OUT/'summary.json',result);save(PUBLIC/'summary.json',result)
    with (PUBLIC/'by_clues.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (PUBLIC/'inputs_without_certificate.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['input_key','clues','compatible_completions','hybrid_certificate_count'])
        for ix in np.flatnonzero(census['n14-full']==0):w.writerow([int(keys[ix]),int(clues[ix]),int(universe['counts'][ix]),0])
    signal.alarm(0);print(json.dumps(result))
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','census']);args=ap.parse_args();freeze() if args.mode=='freeze' else main()
