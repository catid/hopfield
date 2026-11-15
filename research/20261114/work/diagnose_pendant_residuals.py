"""Post-H2ap source-only residual graph/label-energy diagnosis, no new search."""
import json
from collections import Counter
import numpy as np
import study as S
from verify_training_feasibility import energy,LD
from verify_graph_block_source import geometry,connected
from pendant_rescue_v2 import PP,DEST,PARENT


def components(t,b):
    h,w=b.shape;unseen=set(map(int,np.flatnonzero(b)));parts=[];endpoints=set(map(int,np.flatnonzero(t>=2)))
    def adj(u):
        r,c=divmod(u,w);return [v for v in ([u-w] if r else [])+([u+w] if r+1<h else [])+([u-1] if c else [])+([u+1] if c+1<w else []) if b.flat[v]]
    while unseen:
        root=min(unseen);unseen.remove(root);q=[root];part={root}
        while q:
            for v in adj(q.pop()):
                if v in unseen:unseen.remove(v);q.append(v);part.add(v)
        degrees=[len(adj(u)) for u in part];parts.append({'size':len(part),'edges':sum(degrees)//2,'cycle_rank':sum(degrees)//2-len(part)+1,'endpoints':sorted(part&endpoints),'degrees':dict(Counter(degrees)),'cells':sorted(part)})
    return parts


def main():
    p=json.loads(PP.read_text());audit=json.loads((DEST/'audit.json').read_text());assert audit['status']=='passed';rows=[]
    for item in audit['rows']:
        job=item['job'];bad=np.flatnonzero(~np.array(item['success'][4],bool))
        if not len(bad):continue
        with np.load(S.ROOT/p['datasets'][job['shard']]['path']) as z:t=z['tokens'][bad];labels=z['labels'][bad];indices=z['indices'][bad];distance=z['shortest_edges'][bad]
        with np.load(PARENT/'runs'/job['id']/'outputs.npz') as z:a=z['couplings'][bad].astype(LD);b=z['predictions'][3,bad]
        with np.load(S.ROOT/p['model_arrays'][str(job['seed'])]['path']) as z:theta=z['theta'].astype(LD);mu=z['mu_offsets'];ao=z['aoffs']
        gaps=energy(t,labels,a,theta,mu,ao)-energy(t,b,a,theta,mu,ao)
        for k in range(len(bad)):
            parts=components(t[k],b[k]);blocks,stats=geometry(t[k],b[k]);deltas=[]
            for cells in blocks:
                z=b[k:k+1].copy();z[0].flat[list(cells)]=False;d=float((energy(t[k:k+1],z,a[k:k+1],theta,mu,ao)-energy(t[k:k+1],b[k:k+1],a[k:k+1],theta,mu,ao))[0]);deltas.append({'size':len(cells),'delta':d,'connected_after':connected(t[k],z[0])})
            rows.append({'job':job['id'],'index':int(indices[k]),'seed':job['seed'],'selected_cells':int(b[k].sum()),'shortest_edges':int(distance[k]),'label_minus_output_energy':float(gaps[k]),'components':parts,'old_chain_candidates':deltas})
    out={'status':'complete','protocol_sha256':S.sha(PP),'audit_sha256':S.sha(DEST/'audit.json'),'source_sha256':S.sha(__file__),'model_cases':len(rows),'distinct_inputs':len(set(r['index'] for r in rows)),'lower_energy_labels':sum(r['label_minus_output_energy']<-1e-8 for r in rows),'label_energy_gap_min':min(r['label_minus_output_energy'] for r in rows),'label_energy_gap_max':max(r['label_minus_output_energy'] for r in rows),'component_count_histogram':dict(Counter(len(r['components']) for r in rows)),'rows':rows,'scope':'Posthoc source-only graph/literaloriginal-energy diagnosis with labels used only for analysis; no new searches, fits, orfreshinputs.'};S.save(DEST/'residual-diagnosis.json',out);print(json.dumps({k:v for k,v in out.items() if k!='rows'}))


if __name__=='__main__':main()
