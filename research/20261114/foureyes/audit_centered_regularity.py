"""Independent integer centering and incremental degree-difference audit."""
import argparse,json,time
from pathlib import Path
import numpy as np
from centered_regularity_study import ROOT,HERE,PUBLIC,PARENT,PROTOCOL,CONDITIONS,COLUMNS,sha,save


def literal_center(counts,n=81):
    pairs=[(i,j) for i in range(n) for j in range(i+1,n)];values=list(map(int,counts));sums=[0]*n
    assert len(values)==len(pairs)
    for (i,j),c in zip(pairs,values):sums[i]+=c;sums[j]+=c
    residual=[(n-2)*c-sums[i]-sums[j] for (i,j),c in zip(pairs,values)];minimum=min(residual)
    return np.array([r-minimum for r in residual],np.int64),np.array(sums,np.int64),minimum


def incremental_selection(scores,n=81):
    """Maintain sum over unordered cell pairs of squared degree differences."""
    pairs=[(i,j) for i in range(n) for j in range(i+1,n)];values=list(map(int,scores));order=sorted(range(len(values)),key=lambda k:(values[k],k))
    ordered=sorted(values);m=len(values);median=ordered[m//2] if m%2 else (ordered[m//2-1]+ordered[m//2])/2
    degree=[0]*n;total=square=dispersion=cursor=0;rows=[]
    for cut in sorted(set(values)):
        if cut>median:break
        while cursor<len(order) and values[order[cursor]]<=cut:
            for vertex in pairs[order[cursor]]:
                old=degree[vertex]
                # Incrementing one degree changes each incident squared difference.
                dispersion+=2*n*old+n-1-2*total
                square+=2*old+1;degree[vertex]+=1;total+=1
            cursor+=1
        assert cursor>0
        rows.append([cut,cursor,min(degree),max(degree),total,square,dispersion])
    chosen=min(rows,key=lambda r:(r[6],r[1],r[0]));ties=sum(r[6]==chosen[6] for r in rows)
    return scores<=chosen[0],np.array(rows,np.int64),dict(zip(COLUMNS,chosen)),ties


def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def audit(export=False):
    start=time.monotonic();cpu=time.process_time();p=json.loads(PROTOCOL.read_text());skipped=[]
    for path,digest in (p['source_sha256']|p['input_sha256']).items():
        f=ROOT/path
        if f.exists():assert sha(f)==digest,path
        else:assert path.startswith('artifacts/');skipped.append(path)
    a=npz(PUBLIC/'models.npz');candidate=npz(PUBLIC/'candidates.npz');old=npz(PARENT/'models.npz');receipt=json.loads((PUBLIC/'selection.json').read_text())
    assert receipt['status']=='complete' and receipt['protocol_sha256']==sha(PROTOCOL)
    assert receipt['models_sha256']==sha(PUBLIC/'models.npz') and receipt['candidates_sha256']==sha(PUBLIC/'candidates.npz')
    assert receipt['wall_seconds']<p['budget']['selection_wall_seconds']
    assert a['scores'].shape==(128,3240) and a['scores'].dtype==np.int64 and (a['scores']>=0).all()
    assert a['row_sums'].shape==(128,81) and a['row_sums'].dtype==np.int64 and a['raw_min'].shape==(128,) and a['raw_min'].dtype==np.int64
    assert a['regularity_supports'].shape==a['top810_supports'].shape==(128,3240) and a['regularity_supports'].dtype==a['top810_supports'].dtype==bool
    assert a['groups'].tolist()==old['groups'].tolist()==np.repeat(np.arange(64),2).tolist()
    assert a['conditions'].tolist()==old['conditions'].tolist()==CONDITIONS*64
    assert candidate['columns'].tolist()==COLUMNS and candidate['rows'].dtype==np.int64 and candidate['rows'].shape[1]==7
    assert candidate['offsets'].shape==(129,) and candidate['offsets'].dtype==np.int64 and candidate['offsets'][0]==0 and candidate['offsets'][-1]==len(candidate['rows'])
    assert (np.diff(candidate['offsets'])>0).all() and len(receipt['records'])==128
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3))
    rows=[];diagnostics=[]
    for index,count in enumerate(old['counts']):
        assert count.dtype==np.int64 and ((count>=0)&(count<=2400)).all()
        scores,sums,minimum=literal_center(count)
        assert np.array_equal(scores,a['scores'][index]) and np.array_equal(sums,a['row_sums'][index]) and minimum==a['raw_min'][index]
        check=[0]*81
        for i,j,value in zip(ii,jj,scores):check[int(i)]+=int(value);check[int(j)]+=int(value)
        assert len(set(check))==1,'Centered weighted row sums must all coincide'
        regularity,table,chosen,ties=incremental_selection(scores)
        first,last=candidate['offsets'][index:index+2]
        assert np.array_equal(table,candidate['rows'][first:last])
        r=receipt['records'][index];assert r['index']==index and r['group']==index//2 and r['condition']==CONDITIONS[index%2]
        assert r['selected']==chosen and r['minimum_dispersion_ties']==ties and r['candidates']==len(table) and r['median_score']==float(np.median(scores))
        assert np.array_equal(regularity,a['regularity_supports'][index])
        order=sorted(range(3240),key=lambda k:(int(scores[k]),k));top=np.zeros(3240,bool);top[order[:810]]=True
        assert np.array_equal(top,a['top810_supports'][index])
        for arm,support in [('centered_regularity',regularity),('centered_top810',top)]:
            missing=int((truth&~support).sum());extra=int((~truth&support).sum())
            rows.append(dict(index=index,group=index//2,condition=CONDITIONS[index%2],arm=arm,exact=missing==extra==0,missing=missing,extra=extra,retained=int(support.sum())))
        raw_margin=int(count[~truth].min()-count[truth].max());margin=int(scores[~truth].min()-scores[truth].max());exact=bool(np.array_equal(regularity,truth))
        diagnostics.append(dict(index=index,group=index//2,condition=CONDITIONS[index%2],raw_separation_margin=raw_margin,centered_separation_margin=margin,
                                scalar_separable=margin>0,wrong_choice_despite_separability=margin>0 and not exact,
                                selected=chosen,minimum_dispersion_ties=ties,candidates=len(table),centered_weighted_row_sum=check[0],
                                candidate_contains_separating_cut=bool(any(int(scores[truth].max())<=row[0]<int(scores[~truth].min()) for row in table))))
        assert time.monotonic()-start<p['budget']['audit_wall_seconds'],'audit_budget'
    grouped=[];decisions={}
    for condition in CONDITIONS:
        for arm in ['centered_regularity','centered_top810']:
            part=[r for r in rows if r['condition']==condition and r['arm']==arm];assert len(part)==64
            g=dict(condition=condition,arm=arm,exact=sum(r['exact'] for r in part),missing=sum(r['missing'] for r in part),extra=sum(r['extra'] for r in part))
            grouped.append(g)
            if arm=='centered_regularity':decisions[condition]=dict(decision='supported' if g['exact']>=61 and g['missing']==0 else 'falsified',exact=g['exact'],missing=g['missing'],extra=g['extra'])
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(Path(__file__)),models_sha256=sha(PUBLIC/'models.npz'),
                candidates_sha256=sha(PUBLIC/'candidates.npz'),selection_sha256=sha(PUBLIC/'selection.json'),rows=rows,grouped=grouped,diagnostics=diagnostics,
                primary=decisions[CONDITIONS[1]],control=decisions[CONDITIONS[0]],threshold_candidates=len(candidate['rows']),
                new_source_rows=0,new_corruptions=0,new_solver_inputs=0,gpu_seconds=0,scope=p['source_scope'])
    timing=dict(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu)
    if export:
        assert not (PUBLIC/'summary.json').exists();save(PUBLIC/'summary.json',result);save(PUBLIC/'audit-timing.json',timing)
    else:
        assert result==json.loads((PUBLIC/'summary.json').read_text())
        runtime=json.loads((PUBLIC/'runtime.json').read_text());assert runtime['status']=='complete' and runtime['error'] is None and runtime['returncodes']==[0,0]
        assert runtime['protocol_sha256']==sha(PROTOCOL) and runtime['aggregate_cpu_seconds']<p['budget']['aggregate_cpu_seconds']
    print(json.dumps(dict(status='passed',primary=result['primary'],control=result['control'],grouped=grouped,threshold_candidates=result['threshold_candidates'],timing=timing)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--export',action='store_true');audit(ap.parse_args().export)
