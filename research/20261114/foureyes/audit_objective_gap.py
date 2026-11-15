"""Independent exact primal/dual feasibility and empirical true-graph gap."""
from pathlib import Path
import hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def certify(n,d,c,x,y,z):
    assert c.dtype==x.dtype==y.dtype==z.dtype==np.int64
    edges=[(a,b) for a in range(n) for b in range(a+1,n)]
    assert c.shape==x.shape==z.shape==(len(edges),) and y.shape==(n,)
    degree=[0]*n;primal=0;upper=0
    for e,(a,b) in enumerate(edges):
        v=int(x[e]);cost=int(c[e]);u=int(z[e]);assert v in [0,1,2] and u<=0
        reduced=2*cost-int(y[a])-int(y[b])-u;assert reduced>=0
        degree[a]+=v;degree[b]+=v;primal+=cost*v;upper+=u
    assert degree==[2*d]*n
    dual=d*sum(map(int,y))+upper;assert primal==dual
    return primal

def reconstruct(order,family):
    assert sorted(map(int,order))==list(range(81));a=np.zeros((81,81),bool)
    if family=='cliques9':
        for first in range(0,81,9):
            for left in order[first:first+9]:
                for right in order[first:first+9]:
                    if left!=right:a[left,right]=True
    else:
        k={'cycle81':1,'power2':2,'power3':3}[family]
        for i in range(81):
            for offset in range(1,k+1):a[order[i],order[(i+offset)%81]]=True;a[order[(i+offset)%81],order[i]]=True
    return a

def main():
    started=time.perf_counter();public=ROOT/'results/research/foureyes/objective_gap';parent=ROOT/'results/research/foureyes/prefix_budget/degree';p=json.loads((HERE/'objective-gap.json').read_text())
    for rel,h in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==h,rel
    old=json.loads((parent/'summary.json').read_text());summary=json.loads((public/'summary.json').read_text());rows=json.loads((public/'rows.json').read_text());assert len(rows)==len(old['rows'])==768 and summary['protocol_sha256']==sha(HERE/'objective-gap.json') and summary['rows_sha256']==sha(public/'rows.json')
    graphs={}
    for name in ['cycle_power','varying_degree']:
        with np.load(ROOT/f'results/research/foureyes/{name}/controls/graphs.npz',allow_pickle=False) as z:graphs[name]={k:z[k] for k in z.files}
    strict=cooptimal=success=mutations=0;groups={}
    for i,(r,o) in enumerate(zip(rows,old['rows'],strict=True)):
        family=['cycle81','power2','power3','cliques9'][i%12//3];n=[300,600,1200][i%3];group=i//12;name='varying_degree' if family in ['cycle81','cliques9'] else 'cycle_power';index=2*group+int(family in ['cliques9','power3']);a=reconstruct(graphs[name]['orders'][index],family);d=int(a[0].sum());assert (a.sum(0)==d).all();truth=np.array([a[left,right] for left in range(81) for right in range(left+1,81)],bool);assert np.array_equal(truth,graphs[name]['truths'][index]);assert o['selected_degree']==d
        with np.load(parent/f'j{i:03d}/models.npz',allow_pickle=False) as z:c,t,y,u=[z[key] for key in ['cost',f'd{d:02d}_primal_twice',f'd{d:02d}_node_dual_twice',f'd{d:02d}_upper_dual_twice']]
        opt=certify(81,d,c,t,y,u);actual=sum(2*int(c[e]) for e in range(3240) if truth[e]);gap=actual-opt;assert gap>=0
        for which in range(3):
            bad=[t.copy(),y.copy(),u.copy()];bad[which][0]+=1
            try:certify(81,d,c,*bad)
            except AssertionError:mutations+=1
            else:raise AssertionError('Altered certificate accepted')
        assert r==dict(index=i,group=group,family=family,rows=n,degree=d,parent_classification=o['classification'],optimum_twice=opt,truth_cost_twice=actual,gap_twice=gap,truth_objective_status='cooptimal' if gap==0 else 'strictly_suboptimal')
        exact=np.array_equal(t,truth.astype(np.int64)*2);assert exact==(o['classification']=='success');assert opt==o['objective_twice'][d]
        g=groups.setdefault((family,n),dict(family=family,rows=n,exact_graphs=0,failures=0,strictly_suboptimal_truth=0,cooptimal_truth=0,fractional_failures=0,maximum_gap_twice=0));g['maximum_gap_twice']=max(g['maximum_gap_twice'],gap)
        if exact:assert gap==0;success+=1;g['exact_graphs']+=1
        else:
            strict+=gap>0;cooptimal+=gap==0;g['failures']+=1;g['strictly_suboptimal_truth']+=int(gap>0);g['cooptimal_truth']+=int(gap==0);g['fractional_failures']+=int(o['classification']=='resolved_failure_fractional')
    assert success==374 and strict+cooptimal==394 and mutations==2304
    expected=[groups[(f,n)] for f in ['cycle81','power2','power3','cliques9'] for n in [300,600,1200]]
    for key,value in dict(denominator=394,strictly_suboptimal_truth=strict,cooptimal_truth=cooptimal,required_strict=355,decision='supported' if strict>=355 else 'falsified',groups=expected,successes_with_zero_gap=374,selected_certificates=768,new_LPs=0,new_labels=0,new_draws=0,new_GPU_seconds=0).items():assert summary[key]==value,key
    assert summary['status']=='complete' and summary['scientific_cpu_seconds']<60 and summary['scientific_wall_seconds']<90
    print(json.dumps(dict(status='passed',strict=strict,cooptimal=cooptimal,decision=summary['decision'],selected_certificates=768,mutations_rejected=mutations,wall_seconds=time.perf_counter()-started)))
if __name__=='__main__':main()
