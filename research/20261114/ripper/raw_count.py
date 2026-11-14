"""Exploratory raw-solution coverage; no augmentation or inference."""
import csv
import json
import platform
import numpy as np
from run import ROOT,HERE,save,sha,now

OUT=ROOT/'artifacts/research/20261114/ripper/raw'


def main():
    assert platform.node()=='ripper'
    p=json.loads((HERE/'protocol-raw-count.json').read_text());manifest=json.loads((OUT/'input_manifest.json').read_text())
    assert manifest['protocol_sha256']==sha(HERE/'protocol-raw-count.json')
    pairs=np.triu_indices(81,1);rows=[];supports=[]
    for e in manifest['inputs']:
        path=ROOT/e['path'];assert sha(path)==e['sha256'];z=np.load(path);sol=z['solutions']
        assert len(np.unique(sol,axis=0))==1000
        for n in p['sizes']:
            counts=(sol[:n,pairs[0]]==sol[:n,pairs[1]]).sum(0);support=counts==0
            d=OUT/f'counts/n{n}-s{e["seed"]}';d.mkdir(parents=True,exist_ok=False);np.savez_compressed(d/'support.npz',counts=counts,support=support)
            rows.append({'seed':e['seed'],'n':n,'retained_edges':int(support.sum()),'training_sha256':sha(path),'support_sha256':sha(d/'support.npz')});supports.append(support)
    cells=np.arange(81);r,c=cells//9,cells%9;peer=((r[:,None]==r[None])|(c[:,None]==c[None])|((r[:,None]//3==r[None]//3)&(c[:,None]//3==c[None]//3)))[pairs]
    for row,support in zip(rows,supports):row.update(exact_peer_graph=bool(np.array_equal(support,peer)),missing_true_edges=int(np.count_nonzero(peer&~support)),extra_edges=int(np.count_nonzero(~peer&support)))
    assert all(r['missing_true_edges']==0 for r in rows),'Invalid input or counter bug: true peers were excluded'
    grouped=[{'n':n,'exact_graphs':sum(r['exact_peer_graph'] for r in rows if r['n']==n),'extra_edges_per_seed':[r['extra_edges'] for r in rows if r['n']==n]} for n in p['sizes']]
    rec={'status':'validated_exploratory_structure_stage','utc':now(),'protocol_sha256':sha(HERE/'protocol-raw-count.json'),'wrapper_sha256':sha(__file__),'input_manifest_sha256':sha(OUT/'input_manifest.json'),'source_rows':manifest['source_rows'],'unique_complete_board_pool':manifest['unique_complete_boards'],'cross_seed_overlap':manifest['cross_seed_overlap'],'sizes':p['sizes'],'seeds':p['seeds'],'grouped':grouped,'rows':rows,'first_exact_studied_n':{s:next((r['n'] for r in rows if r['seed']==s and r['exact_peer_graph']),None) for s in p['seeds']},'new_gpu_searches':0,'interpretation':'Raw distinct officialtrainingcompleteboards, no augmentation. This stage identifies meaningful model-comparison sizes; no heldoutaccuracyclaim.','limitations':['Uniform draws from a deduplicateddataset pool are notuniform draws fromallSudokusolutions.','Nestedprefixes shareboards withinseed; crossseed overlaps are reported.','Graphoracle used onlyafter selection.','A future PLcomparison needsitsownfrozenprotocol.']}
    save(OUT/'audit.json',rec);public=ROOT/'results/research/ripper';save(public/'raw_count.json',rec)
    with (public/'raw_count_per_seed.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'grouped':grouped,'first_exact_studied_n':rec['first_exact_studied_n']}))


if __name__=='__main__':main()
