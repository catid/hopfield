"""Posthoc H4ad isolation witnesses; not another frozen primary endpoint."""
from pathlib import Path
import json
import numpy as np
from heterogeneous_noise_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,sha,save


def main():
    s=json.loads((PUBLIC/'summary.json').read_text());assert s['status']=='validated'
    manifest=json.loads((OUT/'inputs.json').read_text())
    with np.load(PUBLIC/'models.npz',allow_pickle=False) as z:
        supports=z['supports'].copy();conditions=z['conditions'].copy();hot=z['hot_cells'].copy()
    pairs=[(i,j) for i in range(81) for j in range(i+1,81)]
    units=[[9*r+c for c in range(9)] for r in range(9)]+[[9*r+c for r in range(9)] for c in range(9)]+[[9*r+c for r in range(3*a,3*a+3) for c in range(3*b,3*b+3)] for a in range(3) for b in range(3)]
    rows=[]
    for group in range(16):
        index=2*group+1;assert conditions[index]=='hot18_weight8'
        support=supports[index,2];row=next(r for r in s['rows'] if r['group']==group and r['condition']=='hot18_weight8' and r['arm']=='restricted')
        isolated=[cell for cell in range(81) if all(not keep or cell not in pair for pair,keep in zip(pairs,support))]
        item=dict(group=group,exact_graph=row['exact_graph'],isolated_cells=isolated,witness=None)
        if not row['exact_graph']:
            assert isolated==hot.tolist()
            e=manifest['entries'][index];assert sha(ROOT/e['path'])==e['sha256']
            with np.load(ROOT/e['path'],allow_pickle=False) as z:clean=z['solutions'][0].copy()
            assert all(len(set(clean[u]))==9 for u in units)
            changed=int(hot[0]);invalid=clean.copy();invalid[changed]=(int(clean[changed])+1)%9
            energy=sum(bool(keep) and invalid[i]==invalid[j] for (i,j),keep in zip(pairs,support))
            broken=[k for k,u in enumerate(units) if len(set(invalid[u]))!=9]
            assert energy==0 and broken
            item['witness']=dict(clean=clean.tolist(),invalid=invalid.tolist(),changed_cell=changed,zero_energy=int(energy),invalid_units=broken,input_sha256=e['sha256'])
        else:assert not isolated
        rows.append(item)
    result=dict(status='validated',scope='Posthoc exact global certificate counterexamples for failed heterogeneous restricted-count graphs. No solver run, no claim about actual capped returns or a particular clue mask.',protocol_sha256=sha(PROTOCOL),summary_sha256=sha(PUBLIC/'summary.json'),models_sha256=sha(PUBLIC/'models.npz'),producer_sha256=sha(__file__),rows=rows,witnesses=sum(r['witness'] is not None for r in rows))
    save(PUBLIC/'isolation-witnesses.json',result)
    print(json.dumps(dict(status=result['status'],witnesses=result['witnesses'])))


if __name__=='__main__':main()
