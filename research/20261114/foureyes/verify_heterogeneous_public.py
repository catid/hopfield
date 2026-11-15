"""Portable H4ad graph/diagnostic replay; independent raw fitting audit separate."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PUBLIC=ROOT/'results/research/foureyes/heterogeneous_noise'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol=HERE/'heterogeneous_noise.json';p=json.loads(protocol.read_text());s=json.loads((PUBLIC/'summary.json').read_text())
    assert s['status']=='validated' and s['protocol_sha256']==sha(protocol)
    assert s['auditor_sha256']==sha(HERE/'audit_heterogeneous_noise.py')
    skipped=[]
    for path,digest in (p['source_sha256']|p['input_sha256']).items():
        f=ROOT/path
        if f.exists():assert sha(f)==digest,path
        else:
            assert path.startswith(('artifacts/','upstream/')),path
            skipped.append(path)
    assert sha(PUBLIC/'models.npz')==s['models_sha256']
    with np.load(PUBLIC/'models.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    assert a['arms'].tolist()==p['arms']==['adamw','canonical','restricted']
    assert np.array_equal(a['groups'],np.repeat(np.arange(16),2))
    assert a['conditions'].tolist()==p['conditions']*16
    hot=np.sort(np.random.default_rng(p['hotspot_seed']).choice(81,18,replace=False))
    assert np.array_equal(a['hot_cells'],hot)
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    ishot=np.isin(np.arange(81),hot)
    categories={'hot_hot':ishot[ii]&ishot[jj],'hot_cold':ishot[ii]^ishot[jj],'cold_cold':~ishot[ii]&~ishot[jj]}
    assert a['supports'].shape==(32,3,3240) and a['supports'].dtype==bool
    for key in ['adamw_coefficients','canonical_coefficients','canonical_gradient']:
        assert a[key].shape==(32,3240) and a[key].dtype==np.float64 and np.isfinite(a[key]).all()
    assert a['counts'].shape==(32,3240) and a['counts'].dtype==np.int64 and ((a['counts']>=0)&(a['counts']<=2400)).all()
    for key in ['free_counts','corruption_counts']:
        assert a[key].shape==(32,81) and a[key].dtype==np.int64 and ((a[key]>=0)&(a[key]<=2400)).all()
    assert (a['corruption_counts']<=a['free_counts']).all() and (a['corruption_counts'].sum(1)==21600).all()
    assert np.array_equal(a['free_counts'][::2],a['free_counts'][1::2])
    assert len(s['rows'])==96 and len(s['fit_audits'])==len(s['diagnostics'])==32
    for index,(group,condition) in enumerate(zip(a['groups'],a['conditions'],strict=True)):
        job=f'g{group:02d}-{condition}';fit=s['fit_audits'][index];d=s['diagnostics'][index]
        assert fit['job_id']==d['job_id']==job
        coeff=a['adamw_coefficients'][index];positive=np.sort(coeff[coeff>0]);meta=fit['adamw_metadata']
        if len(positive)>=2 and np.diff(positive).max()>0:
            k=int(np.argmax(np.diff(positive)));cut=(positive[k]+positive[k+1])/2
            adam=coeff>cut;assert meta['eligible'] and abs(meta['threshold']-cut)<1e-14
        else:adam=np.zeros(3240,bool);assert not meta['eligible']
        j=a['canonical_coefficients'][index];positive=j[j>0]
        if len(positive):cut=float(np.median(positive)/4);canonical=(j>0)&(j>=cut)
        else:cut=None;canonical=np.zeros(3240,bool)
        assert cut==fit['canonical_threshold']
        counts=a['counts'][index];values=np.unique(counts);eligible=np.flatnonzero(values[1:]<=np.median(counts))
        k=int(eligible[np.argmax(np.diff(values)[eligible])]) if len(eligible) else None
        threshold=int((values[k]+values[k+1])//2) if k is not None else 0
        assert threshold==fit['restricted_threshold']
        assert np.array_equal(np.stack([adam,canonical,counts<=threshold]),a['supports'][index])
        g=a['canonical_gradient'][index];penalty=9/30000
        kkt=np.where(j>0,g+penalty,np.where(j<0,g-penalty,np.maximum(abs(g)-penalty,0)))
        assert abs(float(abs(kkt).max())-fit['numpy_kkt_linf'])<1e-15
        assert abs(fit['numpy_objective']-fit['numpy_loss']-penalty*float(abs(j).sum()))<1e-12
        assert fit['updates']==256 and fit['finite_budget_not_optimum_claim']
        assert d['true_count_min']==int(counts[truth].min()) and d['true_count_max']==int(counts[truth].max())
        assert d['nonpeer_count_min']==int(counts[~truth].min()) and d['nonpeer_count_max']==int(counts[~truth].max())
        margin=int(counts[~truth].min()-counts[truth].max())
        assert d['separation_margin']==margin and d['any_low_count_scalar_threshold_separates']==(margin>0)
        for prefix,mask in [('hot',ishot),('cold',~ishot)]:
            assert d[prefix+'_free_exposures']==int(a['free_counts'][index,mask].sum())
            assert d[prefix+'_corruptions']==int(a['corruption_counts'][index,mask].sum())
        for arm,support in zip(a['arms'],a['supports'][index],strict=True):
            row=next(r for r in s['rows'] if r['job_id']==job and r['arm']==arm)
            missing=int((truth&~support).sum());extra=int((~truth&support).sum())
            assert row['group']==int(group) and row['condition']==condition and row['banks']==list(range(4*int(group),4*int(group)+4))
            assert row['missing_true']==missing and row['extra']==extra and row['retained']==int(support.sum())
            assert row['exact_graph']==(missing==extra==0) and row['all_true_retained']==(missing==0)
            assert row['support_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
            assert row['eligible']==(meta['eligible'] if arm=='adamw' else cut is not None if arm=='canonical' else True)
            for name,mask in categories.items():
                expected=dict(missing_true=int((truth&mask&~support).sum()),extra=int((~truth&mask&support).sum()),true_pairs=int((truth&mask).sum()),nonpeer_pairs=int((~truth&mask).sum()))
                assert row['hot_categories'][name]==expected
    for grouped in s['grouped']:
        rows=[r for r in s['rows'] if r['condition']==grouped['condition'] and r['arm']==grouped['arm']];assert len(rows)==16
        for total,item in [('exact_graphs','exact_graph'),('eligible_groups','eligible'),('all_true_retaining_groups','all_true_retained'),('missing_true_total','missing_true'),('extra_total','extra')]:assert grouped[total]==sum(r[item] for r in rows)
        assert grouped['ineligible_groups']==16-grouped['eligible_groups']
    target=next(r for r in s['grouped'] if r['condition']=='hot18_weight8' and r['arm']=='restricted')
    control=next(r for r in s['grouped'] if r['condition']=='uniform' and r['arm']=='restricted')
    success=target['exact_graphs']>=15 and target['missing_true_total']==0;control_success=control['exact_graphs']>=15 and control['missing_true_total']==0
    assert s['primary']['decision']==('supported' if success else 'falsified')
    assert s['primary']['exact_graphs']==target['exact_graphs'] and s['primary']['missing_true_total']==target['missing_true_total']
    assert s['primary']['uniform_control_target_met']==control_success and s['primary']['heterogeneous_only_failure']==(not success and control_success)
    witnesses=json.loads((PUBLIC/'isolation-witnesses.json').read_text())
    assert witnesses['status']=='validated' and len(witnesses['rows'])==16
    assert witnesses['summary_sha256']==sha(PUBLIC/'summary.json') and witnesses['models_sha256']==s['models_sha256']
    assert witnesses['protocol_sha256']==sha(protocol) and witnesses['producer_sha256']==sha(HERE/'diagnose_heterogeneous_supports.py')
    units=[list(range(9*r,9*r+9)) for r in range(9)]+[list(range(c,81,9)) for c in range(9)]+[[9*r+c for r in range(3*a,3*a+3) for c in range(3*b,3*b+3)] for a in range(3) for b in range(3)]
    count=0
    for group,row in enumerate(witnesses['rows']):
        support=a['supports'][2*group+1,2];degree=np.bincount(np.r_[ii[support],jj[support]],minlength=81)
        assert row['group']==group and row['isolated_cells']==np.flatnonzero(degree==0).tolist()
        assert row['exact_graph']==np.array_equal(support,truth)
        if row['witness'] is not None:
            w=row['witness'];clean=np.asarray(w['clean']);invalid=np.asarray(w['invalid'])
            source_path=f'artifacts/research/20261114/foureyes/heterogeneous_noise/inputs/g{group:02d}-hot18_weight8.npz'
            assert w['input_sha256']==s['source_audit']['input_sha256'][source_path]
            assert clean.shape==invalid.shape==(81,) and ((clean>=0)&(clean<9)).all() and ((invalid>=0)&(invalid<9)).all()
            assert all(len(set(clean[u]))==9 for u in units)
            assert np.flatnonzero(clean!=invalid).tolist()==[w['changed_cell']] and degree[w['changed_cell']]==0
            assert w['zero_energy']==int((invalid[ii[support]]==invalid[jj[support]]).sum())==0
            assert w['invalid_units']==[k for k,u in enumerate(units) if len(set(invalid[u]))<9] and w['invalid_units']
            count+=1
        else:assert row['exact_graph']
    assert count==witnesses['witnesses']
    for phase,record in s['launch'].items():
        assert record['status']=='complete' and record['failure'] is None and record['returncodes']==[0,0,0,0]
        assert record['wall_seconds']<p['budget'][phase+'_phase_wall_seconds']
    print(json.dumps(dict(status='passed',graphs=96,primary=s['primary'],exact_invalid_zero_energy_witnesses=count,
                         excluded_raw_or_upstream_inputs=len(skipped),scope='Public support, threshold, per-cell/edge diagnostics and counterexample replay. Raw corruptions, checkpoints, objective gradients and histories require separate local archive audit.')))


if __name__=='__main__':main()
