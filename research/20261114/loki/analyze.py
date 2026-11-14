"""Validate saved H3 outputs and apply the preregistered decision rule, CPU only."""
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/"research/20261114/loki"
OUT=ROOT/"artifacts/research/20261114/loki"
sys.path.insert(0,str(ROOT/"upstream"))
from cram import nqueens_data as D

HASHES={}
def sha(p):
    p=Path(p)
    with p.open('rb') as f:v=hashlib.file_digest(f,'sha256').hexdigest()
    HASHES[str(p.relative_to(ROOT))]=v
    return v
def read(p):
    sha(p);return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def cp_upper(k,n,alpha=.05):
    """One-sided exact binomial upper bound; stable binomial CDF inversion."""
    if k==n:return 1.
    if k==0:return -math.expm1(math.log(alpha)/n)
    ids=np.arange(k+1,dtype=float)
    comb=np.array([math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1) for i in range(k+1)])
    lo,hi=k/n,1.
    for _ in range(70):
        p=(lo+hi)/2
        terms=comb+ids*math.log(p)+(n-ids)*math.log1p(-p)
        m=terms.max();cdf=math.exp(m)*np.exp(terms-m).sum()
        if cdf>alpha:lo=p
        else:hi=p
    return (lo+hi)/2


def check(n,arm,cohort,seed,all_solutions):
    d=OUT/f'n{n}/{cohort}/{arm}/s{seed}'
    rec=read(d/'result.json');assert rec['status']=='complete'
    fit=read(OUT/f'n{n}/fits/{arm}/fit.json')
    assert fit['updates']==4000 and fit['model_sha256']==rec['model_sha256']==sha(OUT/f'n{n}/fits/{arm}/model.pt')
    assert rec['records_sha256']==sha(d/'records.npz')
    assert rec['protocol_sha256']==sha(SRC/'protocol.json')
    assert rec['source_sha256']==sha(SRC/'study.py')
    z=dict(np.load(d/'records.npz')); b=np.unpackbits(z['boards'],axis=-1)[...,:n*n].astype(bool)
    inputs=np.load(OUT/f'cohorts-n{n}.npz')[f'{cohort}_inputs']==2
    assert np.array_equal(z['inputs'],inputs)
    complete_keys={row.tobytes() for row in all_solutions}
    ok=np.array([[row.tobytes() in complete_keys and row[givens].all() for row in samples] for givens,samples in zip(inputs,b)])
    assert np.array_equal(ok,z['valid'])
    counts=np.array([all_solutions[:,given].all(1).sum() for given in inputs]);denom=np.minimum(counts,20)
    assert np.array_equal(counts,z['solution_counts']) and np.array_equal(denom,z['denom'])
    found=np.array([min(len({row.tobytes() for row in samples[valid]}),cap) for samples,valid,cap in zip(b,ok,denom)])
    pick=np.argmin(z['energy'],axis=1);n20=ok[np.arange(len(inputs)),pick]
    assert np.array_equal(found,z['found']) and np.array_equal(n20,z['n20'])
    assert np.array_equal(ok[:,0],z['n1'])
    assert ((z['moves']>=0)&(z['moves']<=20000)).all()
    assert (z['moves'][~z['stopped']]==20000).all()
    assert np.array_equal(z['flop'],2*(n*n)**2+2*n*n+1+z['moves']*(5*n*n+14))
    assert int((z['stopped']&~ok).sum())==rec['wrong_stops'] and int((~z['stopped']).sum())==rec['budget_failures']
    assert 100*float(n20.mean())==rec['n20_accuracy_pct']
    assert 100*float(found.sum()/denom.sum())==rec['coverage_pct']
    return rec,z


def bootstrap(a,b,denom):
    # Input clusters contain all three repeated search seeds.
    rng=np.random.default_rng(20261114);m=a['n20'].shape[1];acc=[];cov=[]
    ad=(a['n20'].astype(float)-b['n20'].astype(float)).mean(axis=0)
    cd=(a['found'].astype(float)-b['found'].astype(float)).mean(axis=0)
    for _ in range(20):
        ix=rng.integers(0,m,(100,m))
        acc.extend(ad[ix].mean(1)*100)
        cov.extend(cd[ix].sum(1)/denom[ix].sum(1)*100)
    return dict(accuracy_difference_pp=float(ad.mean()*100),coverage_difference_pp=float(cd.sum()/denom.sum()*100),
        accuracy_one_sided_95_lower_pp=float(np.quantile(acc,.05)),accuracy_one_sided_95_upper_pp=float(np.quantile(acc,.95)),
        coverage_one_sided_95_lower_pp=float(np.quantile(cov,.05)),coverage_one_sided_95_upper_pp=float(np.quantile(cov,.95)),
        replicates=2000,unit='unique input, with all three search seeds retained inside each bootstrap cluster',seed=20261114)


def main():
    start=time.perf_counter();p=read(SRC/'protocol.json');selection=read(OUT/'selection.json');chosen=selection['selected_arm']
    assert not selection['confirmation_used'] and not selection['test_used']
    for rel,h in p['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    for c in p['cohorts'].values():
        assert sha(ROOT/c['cohort_file'])==c['cohort_sha256']
        for rel,h in c['data_sha256'].items():assert sha(ROOT/rel)==h
    rows=[];evidence={};comparisons=[]
    for n in (10,8):
        all_solutions=np.asarray(D.completions(np.zeros(n*n,bool),n),bool);assert len(all_solutions)=={8:92,10:724}[n]
        arms=list(p['arms']) if n==10 else []
        for cohort,current in [('discovery',arms),('confirmation',[chosen,'sgd-pre','sgd-plain'])]:
            for arm in current:
                checked=[check(n,arm,cohort,s,all_solutions) for s in (0,1,2)]
                recs,arrays=zip(*checked)
                arr={k:np.stack([z[k] for z in arrays]) for k in ('n20','n1','found')}
                denom=arrays[0]['denom'];assert all(np.array_equal(z['denom'],denom) for z in arrays)
                evidence[n,cohort,arm]=(arr,denom)
                fit=read(OUT/f'n{n}/fits/{arm}/fit.json')
                rows.append(dict(n=n,cohort=cohort,arm=arm,unique_inputs=len(denom),search_seeds='0;1;2',
                    n20_accuracy_pct=float(arr['n20'].mean()*100),n1_accuracy_pct=float(arr['n1'].mean()*100),
                    coverage_pct=float(arr['found'].sum(axis=1).mean()/denom.sum()*100),
                    wrong_stops=sum(r['wrong_stops'] for r in recs),budget_failures=sum(r['budget_failures'] for r in recs),
                    mean_moves=float(np.mean([r['mean_moves'] for r in recs])),
                    search_seconds_total=sum(r['search_seconds'] for r in recs),fit_seconds=fit['complete_fit_seconds'],
                    final_pl=fit['final_loss'],unit=fit['unit'],delta=fit['delta']))
        for control in ('sgd-pre','sgd-plain'):
            a,denom=evidence[n,'confirmation',chosen];b,_=evidence[n,'confirmation',control]
            stats=bootstrap(a,b,denom)
            loss=(b['n20']&~a['n20']).any(axis=0);loss_count=int(loss.sum());upper=cp_upper(loss_count,len(loss))
            safe=upper<.001
            supported=safe and stats['coverage_one_sided_95_lower_pp']>-2
            falsified=stats['accuracy_one_sided_95_upper_pp']<-.1 or stats['coverage_one_sided_95_upper_pp']<-2
            assert not(supported and falsified)
            decision='supported' if supported else 'falsified' if falsified else 'inconclusive'
            comparisons.append(dict(n=n,arm=chosen,control=control,primary=n==10 and control=='sgd-pre',
                unique_inputs=len(loss),inputs_with_any_accuracy_regression=loss_count,
                regression_probability_exact95_upper=upper,conservative_accuracy95_lower_pp=-100*upper,
                accuracy_noninferiority_safeguard_passed=safe,decision=decision,**stats))
    primary=next(c for c in comparisons if c['primary'])
    source=read(OUT/'cpu-validation.json');assert source['status']=='passed' and source['source_sha256']==sha(SRC/'study.py')
    with (OUT/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status='validated',utc=dt.datetime.now(dt.timezone.utc).isoformat(),hypothesis='H3',decision=primary['decision'],
        selected_arm=chosen,discovery_selection=selection,rows=rows,comparisons=comparisons,
        distinct_models=10,fit_updates_each=4000,search_jobs=39,independent_board_validation='Saved boards tested against exhaustive92/724solution sets and input givens; counts/FLOPs/statuses recomputed',
        inference_seed_note='Three search seeds for each deterministic fitted model; not three independently trained models.',
        source_sha256=sha(Path(__file__)),new_training_or_inference=False,input_sha256=HASHES,audit_seconds=time.perf_counter()-start)
    save(OUT/'summary.json',result)
    lines=['# H3: AdamW versus geometry-derived preconditioning','',
        f"**Preregistered n=10 result: {primary['decision']}.** Selection on 512 discovery inputs chose `{chosen}`; confirmation used 3,179 different unique partial boards, each with 20 chains under each of 3 paired search seeds. Models use the same 97,554 fitting rows, 4,000 full-batch updates, zero initialization and exact pseudolikelihood. No test data were used.",'',
        'BP AdamW here differentiates the local conditional loss with respect to energy parameters. It does not backpropagate through search. The SGD controls use the frozen paper LR 30 with geometry preconditioning on/off; AdamW uses the selected LR and weight decay 0.01. All arms use the same normalized temperature and stopping policy, with statistics fitted from their own training rows.','',
        '| Size | Method | N20 accuracy (%) | Coverage (%) | Wrong stopped chains | Capped chains | Complete fit (s) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for n in (10,8):
        for arm in (chosen,'sgd-pre','sgd-plain'):
            r=next(r for r in rows if r['n']==n and r['cohort']=='confirmation' and r['arm']==arm)
            name='BP AdamW, no preconditioner' if arm==chosen else 'SGD + preconditioner' if arm=='sgd-pre' else 'SGD, no preconditioner'
            lines.append(f"| {n} | {name} | {r['n20_accuracy_pct']:.6f} | {r['coverage_pct']:.6f} | {r['wrong_stops']:,} | {r['budget_failures']:,} | {r['fit_seconds']:.3f} |")
    lines+=['',f"Against preconditioned SGD at n=10, AdamW changes accuracy by **{primary['accuracy_difference_pp']:+.6f} percentage points** and coverage by **{primary['coverage_difference_pp']:+.6f} points**. One-sided 95% bootstrap lower/upper bounds are [{primary['accuracy_one_sided_95_lower_pp']:.6f}, {primary['accuracy_one_sided_95_upper_pp']:.6f}] for accuracy and [{primary['coverage_one_sided_95_lower_pp']:.6f}, {primary['coverage_one_sided_95_upper_pp']:.6f}] for coverage. These endpoints form a 90% two-sided interval; they are the preregistered one-sided tests.",'',
        f"The accuracy safeguard found {primary['inputs_with_any_accuracy_regression']} unique inputs with at least one paired-seed regression. Its exact one-sided 95% upper probability is {100*primary['regression_probability_exact95_upper']:.6f}%, giving the conservative accuracy-difference lower bound {primary['conservative_accuracy95_lower_pp']:.6f} points. Support required this bound above −0.1 point and the coverage lower bound above −2 points. Absence of evidence is reported as inconclusive.",'',
        'Coverage counts distinct valid completions, capped at min(all completions, 20), and pools unique inputs by that denominator. N20 keeps the lowest-energy returned chain, including a capped chain’s best board under the unchanged paper convention. Search-seed repeats are clustered inside each unique input for uncertainty. The n8 result is a secondary transfer check with only 310 confirmation inputs, not a new LR selection.','',
        'Fit times are one observation per deterministic model on Loki Blackwell, with CUDA context already initialized, and include progress logging. They are not warmed benchmarks or H200 comparisons. The seven-arm discovery sweep took separate fits; selection overhead is excluded from the final model fit column.','',
        'The official published SGD choice was previously selected using the original validation set; it is a frozen external control here. The new AdamW choice used only this campaign’s discovery subset. The training and validation sets are disjoint by partial input, but completed solution labels can repeat across different partial inputs, as in the original GRAM task.','',
        '[Frozen protocol](protocol.json), [runner](study.py), [CPU auditor](analyze.py), [machine-readable results](../../../artifacts/research/20261114/loki/summary.json), [all rows](../../../artifacts/research/20261114/loki/results.csv).']
    # From research/date/host to workspace requires three parent segments.
    (SRC/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='validated',decision=result['decision'],primary=primary,hashed_files=len(HASHES),audit_seconds=result['audit_seconds'])))


if __name__=='__main__':main()
