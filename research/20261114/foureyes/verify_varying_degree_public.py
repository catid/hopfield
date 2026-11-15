"""Portable H4al threshold and graph replay; raw fitting audit remains separate."""
from pathlib import Path
import hashlib
import io
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PUBLIC=ROOT/'results/research/foureyes/varying_degree/controls'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def npy_sha(array):
    stream=io.BytesIO();np.save(stream,array,allow_pickle=False)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def verify_synthetic_sources(p,s,a):
    from audit_varying_degree_controls import independent_graph,independent_colorings
    from audit_regularity_intensity import literal_corruption
    source=s['source_audit'];manifest=json.loads((PUBLIC/'generation-manifest.json').read_text())
    assert source==json.loads((PUBLIC/'source-audit.json').read_text())
    assert source['graphs_sha256']==s['graphs_sha256']==sha(PUBLIC/'graphs.npz')
    assert source['input_manifest_sha256']==sha(PUBLIC/'generation-manifest.json')
    seal=json.loads((PUBLIC/'graph-order-seal.json').read_text())
    assert seal['sealed_before_colorings_and_noise'] and seal['protocol_sha256']==sha(HERE/'varying_degree.json')
    assert manifest['graph_order_seal_sha256']==sha(PUBLIC/'graph-order-seal.json')
    with np.load(PUBLIC/'graphs.npz',allow_pickle=False) as z:graphs={k:z[k].copy() for k in z.files}
    assert graphs['truths'].shape==(128,3240) and graphs['truths'].dtype==bool
    assert graphs['groups'].tolist()==np.repeat(np.arange(64),2).tolist() and graphs['conditions'].tolist()==['cycle81','cliques9']*64
    assert len(manifest['entries'])==128 and len(source['rows'])==128
    ii,jj=np.triu_indices(81,1);duplicates=generated=0
    for index,e in enumerate(manifest['entries']):
        family=p['conditions'][index%2];order=np.random.default_rng(p['graph_seed_base']+index).permutation(81)
        assert e['index']==index and e['group']==index//2 and e['condition']==family
        assert np.array_equal(order,graphs['orders'][index])
        truth=independent_graph(order,family);assert np.array_equal(truth,graphs['truths'][index])
        degree=np.zeros(81,np.int64);np.add.at(degree,ii,truth);np.add.at(degree,jj,truth)
        assert (degree==p['true_degrees_for_evaluation_only'][family]).all()
        clean,receipt=independent_colorings(order,family,p['coloring_seed_base']+index)
        assert receipt==e['sampler_receipt'] and (clean[:,ii[truth]]!=clean[:,jj[truth]]).all()
        hot=np.sort(np.random.default_rng(p['heatmap_seed_base']+index).choice(81,18,replace=False))
        assert e['hot_cells']==hot.tolist()==a['hot_cells_by_input'][index].tolist()
        rng=np.random.default_rng(p['noise_rng_base']+index);uniforms=1-rng.random((2400,81));offsets=rng.integers(1,9,(2400,81),dtype=np.int64)
        noisy,_=literal_corruption(clean,np.ones((2400,81),bool),uniforms,offsets,hot,'hot18_weight32')
        assert ((noisy!=clean).sum(1)==9).all()
        assert np.array_equal((noisy[:,ii]==noisy[:,jj]).sum(0),a['counts'][index])
        assert (a['free_counts'][index]==2400).all() and np.array_equal((noisy!=clean).sum(0),a['corruption_counts'][index])
        duplicate=2400-len(np.unique(clean,axis=0));duplicates+=duplicate;generated+=receipt['generated_candidates']
        row=source['rows'][index]
        assert row['index']==index and row['condition']==family and row['within_bank_duplicate_instances']==duplicate
        assert row['generated_candidates']==receipt['generated_candidates'] and row['input_sha256']==e['sha256']
    assert duplicates==source['within_bank_duplicate_instances'] and generated==source['total_generated_candidates']
    assert source['synthetic_observation_instances']==s['synthetic_observation_instances']==307200
    assert source['new_official_source_rows']==s['new_official_source_rows']==source['new_solver_inputs']==s['new_solver_inputs']==0
    return graphs['truths']


def replay_candidates(counts):
    """Independent incremental degree sweep with pairwise dispersion identity."""
    pairs=[(left,right) for left in range(81) for right in range(left+1,81)]
    order=sorted(range(len(pairs)),key=lambda k:int(counts[k]));values=sorted(map(int,counts))
    median=(values[1619]+values[1620])/2;degree=[0]*81;candidates=[];cursor=0
    for cut in sorted(set(values)):
        if cut>median:break
        while cursor<len(order) and int(counts[order[cursor]])<=cut:
            left,right=pairs[order[cursor]];degree[left]+=1;degree[right]+=1;cursor+=1
        assert cursor>0
        dispersion=sum((degree[left]-degree[right])**2 for left in range(81) for right in range(left+1,81))
        candidates.append(dict(cut=cut,retained=cursor,degree_min=min(degree),degree_max=max(degree),
                               degree_sum=sum(degree),degree_square_sum=sum(d*d for d in degree),dispersion=dispersion))
    best=sorted(candidates,key=lambda row:(row['dispersion'],row['retained'],row['cut']))[0]
    return counts<=best['cut'],dict(n=81,median_count=float(median),candidates=candidates,selected=best,
                                   minimum_dispersion_ties=sum(row['dispersion']==best['dispersion'] for row in candidates))


def main():
    protocol=HERE/'varying_degree.json';p=json.loads(protocol.read_text());s=json.loads((PUBLIC/'summary.json').read_text())
    assert s['status']=='validated' and s['protocol_sha256']==sha(protocol)
    assert s['auditor_sha256']==sha(HERE/'audit_varying_degree_controls.py')
    
    skipped=[]
    for path,digest in (p['source_sha256']|p['input_sha256']).items():
        f=ROOT/path
        if f.exists():assert sha(f)==digest,path
        else:
            assert path.startswith(('artifacts/','upstream/')),path
            skipped.append(path)
    assert sha(PUBLIC/'models.npz')==s['models_sha256']
    with np.load(PUBLIC/'models.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    assert a['arms'].tolist()==p['arms']==['adamw','canonical','restricted','regularity']
    assert np.array_equal(a['groups'],np.repeat(np.arange(64),2))
    assert a['conditions'].tolist()==p['conditions']*64
    truths=verify_synthetic_sources(p,s,a)
    heatmaps=[np.sort(np.random.default_rng(p['heatmap_seed_base']+g).choice(81,18,replace=False)) for g in range(128)]
    assert np.array_equal(a['hot_cells_by_input'],np.array(heatmaps))
    ii,jj=np.triu_indices(81,1)
    records=json.loads((PUBLIC/'selection-records.json').read_text())
    assert sha(PUBLIC/'selection-records.json')==s['selection_records_sha256'] and records==s['selection_records'] and len(records)==128
    total_candidates=0
    assert a['supports'].shape==(128,4,3240) and a['supports'].dtype==bool
    for key in ['adamw_coefficients','canonical_coefficients','canonical_gradient']:
        assert a[key].shape==(128,3240) and a[key].dtype==np.float64 and np.isfinite(a[key]).all()
    assert a['counts'].shape==(128,3240) and a['counts'].dtype==np.int64 and ((a['counts']>=0)&(a['counts']<=2400)).all()
    for key in ['free_counts','corruption_counts']:
        assert a[key].shape==(128,81) and a[key].dtype==np.int64 and ((a[key]>=0)&(a[key]<=2400)).all()
    assert (a['corruption_counts']<=a['free_counts']).all() and (a['corruption_counts'].sum(1)==21600).all()
    assert len(s['rows'])==512 and len(s['fit_audits'])==len(s['diagnostics'])==128
    for index,(group,condition) in enumerate(zip(a['groups'],a['conditions'],strict=True)):
        truth=truths[index];ishot=np.isin(np.arange(81),heatmaps[index])
        categories={'hot_hot':ishot[ii]&ishot[jj],'hot_cold':ishot[ii]^ishot[jj],'cold_cold':~ishot[ii]&~ishot[jj]}
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
        regularity,record=replay_candidates(counts)
        actual=records[index]
        assert actual['group']==int(group) and actual['job_id']==job and actual['protocol_sha256']==sha(protocol)
        for key,value in record.items():assert actual[key]==value,(group,key)
        assert actual['models_sha256']==s['input_sha256'][f'artifacts/research/20261114/foureyes/varying_degree/fits/{job}/models.npz']
        receipt=s['selection_receipt']['rows'][index]
        assert receipt['group']==int(group) and receipt['job_id']==job
        assert receipt['record_sha256']==s['input_sha256'][f'artifacts/research/20261114/foureyes/varying_degree/regularity/{job}.json']
        total_candidates+=len(record['candidates'])
        assert np.array_equal(np.stack([adam,canonical,counts<=threshold,regularity]),a['supports'][index])
        g=a['canonical_gradient'][index];penalty=9/30000
        kkt=np.where(j>0,g+penalty,np.where(j<0,g-penalty,np.maximum(abs(g)-penalty,0)))
        assert abs(float(abs(kkt).max())-fit['numpy_kkt_linf'])<1e-15
        assert abs(fit['numpy_objective']-fit['numpy_loss']-penalty*float(abs(j).sum()))<1e-12
        assert fit['updates']==256 and fit['finite_budget_not_optimum_claim']
        assert d['true_count_min']==int(counts[truth].min()) and d['true_count_max']==int(counts[truth].max())
        assert d['nonpeer_count_min']==int(counts[~truth].min()) and d['nonpeer_count_max']==int(counts[~truth].max())
        margin=int(counts[~truth].min()-counts[truth].max())
        assert d['separation_margin']==margin and d['any_low_count_scalar_threshold_separates']==(margin>0)
        assert d['regularity_selected_cut']==record['selected']['cut'] and d['regularity_selected_dispersion']==record['selected']['dispersion']
        exact=bool(np.array_equal(regularity,truth))
        assert d['regularity_exact']==exact and d['no_separating_threshold']==(margin<=0)
        assert d['wrong_regularity_choice_despite_separability']==(margin>0 and not exact)
        assert d['separating_threshold_in_candidate_set']==any(int(counts[truth].max())<=row['cut']<int(counts[~truth].min()) for row in record['candidates'])
        for prefix,mask in [('hot',ishot),('cold',~ishot)]:
            assert d[prefix+'_free_exposures']==int(a['free_counts'][index,mask].sum())
            assert d[prefix+'_corruptions']==int(a['corruption_counts'][index,mask].sum())
        for arm,support in zip(a['arms'],a['supports'][index],strict=True):
            row=next(r for r in s['rows'] if r['job_id']==job and r['arm']==arm)
            missing=int((truth&~support).sum());extra=int((~truth&support).sum())
            assert row['group']==int(group) and row['condition']==condition and row['banks']==[int(group)]
            assert row['missing_true']==missing and row['extra']==extra and row['retained']==int(support.sum())
            assert row['exact_graph']==(missing==extra==0) and row['all_true_retained']==(missing==0)
            assert row['support_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
            assert row['eligible']==(meta['eligible'] if arm=='adamw' else cut is not None if arm=='canonical' else True)
            for name,mask in categories.items():
                expected=dict(missing_true=int((truth&mask&~support).sum()),extra=int((~truth&mask&support).sum()),true_pairs=int((truth&mask).sum()),nonpeer_pairs=int((~truth&mask).sum()))
                assert row['hot_categories'][name]==expected
    for grouped in s['grouped']:
        rows=[r for r in s['rows'] if r['condition']==grouped['condition'] and r['arm']==grouped['arm']];assert len(rows)==64
        for total,item in [('exact_graphs','exact_graph'),('eligible_groups','eligible'),('all_true_retaining_groups','all_true_retained'),('missing_true_total','missing_true'),('extra_total','extra')]:assert grouped[total]==sum(r[item] for r in rows)
        assert grouped['ineligible_groups']==64-grouped['eligible_groups']
    assert s['primary']['status']=='descriptive_controls_only'
    receipt=s['selection_receipt']
    assert receipt['status']=='complete' and receipt['protocol_sha256']==sha(protocol)
    assert receipt['selector_sha256']==sha(HERE/'regularity_gap.py') and 0<receipt['wall_seconds']<p['budget']['selection_wall_seconds']
    assert s['logical_graphs']==512 and s['groups']==64 and s['synthetic_banks']==128 and s['all_groups_retained']
    for phase,record in s['launch'].items():
        assert record['status']=='complete' and record['failure'] is None and record['returncodes']==[0,0,0,0]
        assert record['wall_seconds']<p['budget'][phase+'_phase_wall_seconds']
    print(json.dumps(dict(status='passed',graphs=512,primary=s['primary'],threshold_candidates=total_candidates,
                         excluded_raw_or_upstream_inputs=len(skipped),scope='Public full seeded synthetic source/graph/corruption/count regeneration, every threshold candidate, degree dispersion, tie breaks and graph diagnostics replay. Raw checkpoints and objective histories require separate local archive audit.')))


if __name__=='__main__':main()
