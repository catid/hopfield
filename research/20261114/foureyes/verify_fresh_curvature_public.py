"""Portable H4ak threshold and graph replay; raw fitting audit remains separate."""
from pathlib import Path
import hashlib
import io
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PUBLIC=ROOT/'results/research/foureyes/fresh_curvature/controls'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def npy_sha(array):
    stream=io.BytesIO();np.save(stream,array,allow_pickle=False)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def verify_source_ids(p,s):
    provenance=json.loads((PUBLIC/'source-provenance.json').read_text())
    ledger=json.loads((PUBLIC/'source-ledger.json').read_text());source=s['source_audit']
    assert provenance['status']=='exported_from_validated_archive'
    assert provenance['protocol_sha256']==s['protocol_sha256'] and provenance['summary_sha256']==sha(PUBLIC/'summary.json')
    assert provenance['publisher_sha256']==sha(HERE/'publish_fresh_curvature.py')
    assert provenance['source_ledger_sha256']==sha(PUBLIC/'source-ledger.json')==p['source_ledger_sha256']==source['source_ledger_sha256']
    assert provenance['source_identities_sha256']==sha(PUBLIC/'source-identities.npz') and provenance['raw_draws_sha256']==source['source_draws_sha256']
    with np.load(PUBLIC/'source-identities.npz',allow_pickle=False) as z:
        assert set(z.files)=={'source_rows','excluded_ids','motivating_ids'}
        draws=z['source_rows'].copy();excluded=z['excluded_ids'].copy();motivating=z['motivating_ids'].copy()
    # Reconstruct the declared exclusion union from already public predecessor exports.
    pieces=[]
    for rel,key in [('results/research/ripper/fresh_transfer_excluded_ids.npz','excluded'),('results/research/ripper/fresh_transfer_selected_sources.npz','all_source_row')]:
        assert sha(ROOT/rel)==ledger['exclusion_input_sha256'][rel]
        with np.load(ROOT/rel,allow_pickle=False) as z:pieces.append(np.unique(z[key]))
    assert len(np.union1d(*pieces))==2045471
    previous=ROOT/'results/research/foureyes/regularity_fresh'
    previous_receipt=json.loads((previous/'source-provenance.json').read_text())
    assert previous_receipt['source_identities_sha256']==sha(previous/'source-identities.npz')
    assert previous_receipt['raw_draws_sha256']==ledger['exclusion_input_sha256']['artifacts/research/20261114/foureyes/regularity_fresh/source-draws.npz']
    with np.load(previous/'source-identities.npz',allow_pickle=False) as z:
        assert np.array_equal(motivating,np.unique(z['source_rows']))
        prior_excluded=z['excluded_ids'].copy();pieces.extend([motivating,prior_excluded])
    assert npy_sha(prior_excluded)==ledger['exclusion_input_sha256']['artifacts/research/20261114/foureyes/regularity_fresh_design/excluded.npy']
    assert np.array_equal(np.unique(np.concatenate(pieces)),excluded)
    assert draws.shape==(64,2400) and all(a.dtype==np.int64 for a in [draws,excluded,motivating])
    assert np.array_equal(excluded,np.unique(excluded)) and np.array_equal(motivating,np.unique(motivating))
    assert ((excluded>=0)&(excluded<ledger['total_source_rows'])).all() and np.isin(motivating,excluded).all()
    eligible=np.setdiff1d(np.arange(ledger['total_source_rows'],dtype=np.int64),excluded,assume_unique=True)
    for name,array in [('eligible.npy',eligible),('excluded.npy',excluded),('motivating.npy',motivating)]:
        assert npy_sha(array)==ledger['array_sha256'][name]==p['input_sha256']['artifacts/research/20261114/foureyes/fresh_curvature_design/'+name]
    assert len(eligible)==ledger['eligible_rows']==source['eligible_rows']
    assert len(excluded)==ledger['excluded_rows']==source['prior_excluded_rows']
    assert len(motivating)==ledger['motivating_unique_rows']==source['motivating_excluded_rows']
    for bank,rows in enumerate(draws):
        assert len(np.unique(rows))==2400
        assert np.array_equal(rows,np.random.default_rng(p['bank_rng_base']+bank).choice(eligible,2400,replace=False))
    unique=np.unique(draws);assert not np.intersect1d(unique,excluded).size
    assert len(unique)==source['new_unique_source_rows']==s['new_unique_source_rows']==source['independent_csv_answers_and_clue_masks_verified']
    assert draws.size==source['new_source_draw_instances']==s['new_source_draw_instances']==153600
    assert draws.size-len(unique)==source['crossbank_repeated_instances']
    assert source['within_bank_repeated_instances']==source['declared_exclusion_overlap']==source['motivating_source_overlap']==0
    assert 0<source['timing']['wall_seconds']<p['budget']['audit_wall_seconds']
    assert 0<s['outcome_audit_timing']['wall_seconds']<p['budget']['audit_wall_seconds']


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
    protocol=HERE/'fresh_curvature.json';p=json.loads(protocol.read_text());s=json.loads((PUBLIC/'summary.json').read_text())
    assert s['status']=='validated' and s['protocol_sha256']==sha(protocol)
    assert s['auditor_sha256']==sha(HERE/'audit_fresh_curvature_controls.py')
    verify_source_ids(p,s)
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
    assert np.array_equal(a['groups'],np.arange(64))
    assert a['conditions'].tolist()==[p['condition']]*64
    heatmaps=[np.sort(np.random.default_rng(p['heatmap_seed_base']+g).choice(81,18,replace=False)) for g in range(64)]
    assert np.array_equal(a['hot_cells_by_group'],np.array(heatmaps))
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    records=json.loads((PUBLIC/'selection-records.json').read_text())
    assert sha(PUBLIC/'selection-records.json')==s['selection_records_sha256'] and records==s['selection_records'] and len(records)==64
    total_candidates=0
    assert a['supports'].shape==(64,4,3240) and a['supports'].dtype==bool
    for key in ['adamw_coefficients','canonical_coefficients','canonical_gradient']:
        assert a[key].shape==(64,3240) and a[key].dtype==np.float64 and np.isfinite(a[key]).all()
    assert a['counts'].shape==(64,3240) and a['counts'].dtype==np.int64 and ((a['counts']>=0)&(a['counts']<=2400)).all()
    for key in ['free_counts','corruption_counts']:
        assert a[key].shape==(64,81) and a[key].dtype==np.int64 and ((a[key]>=0)&(a[key]<=2400)).all()
    assert (a['corruption_counts']<=a['free_counts']).all() and (a['corruption_counts'].sum(1)==21600).all()
    assert len(s['rows'])==256 and len(s['fit_audits'])==len(s['diagnostics'])==64
    for index,(group,condition) in enumerate(zip(a['groups'],a['conditions'],strict=True)):
        ishot=np.isin(np.arange(81),heatmaps[index])
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
        assert actual['models_sha256']==s['input_sha256'][f'artifacts/research/20261114/foureyes/fresh_curvature/fits/{job}/models.npz']
        receipt=s['selection_receipt']['rows'][index]
        assert receipt['group']==int(group) and receipt['job_id']==job
        assert receipt['record_sha256']==s['input_sha256'][f'artifacts/research/20261114/foureyes/fresh_curvature/regularity/{job}.json']
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
    assert s['logical_graphs']==256 and s['groups']==64 and s['all_groups_retained']
    for phase,record in s['launch'].items():
        assert record['status']=='complete' and record['failure'] is None and record['returncodes']==[0,0,0,0]
        assert record['wall_seconds']<p['budget'][phase+'_phase_wall_seconds']
    print(json.dumps(dict(status='passed',graphs=256,primary=s['primary'],threshold_candidates=total_candidates,
                         excluded_raw_or_upstream_inputs=len(skipped),scope='Public seeded source IDs, frozen exclusion arrays, overlap, every threshold candidate, degree dispersion, tie breaks and graph diagnostics replay. Original exclusion manifests, CSV labels, raw corruptions, checkpoints and objective histories require separate local archive audit.')))


if __name__=='__main__':main()
