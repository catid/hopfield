"""No-fit equality control, with support frozen before geometry inspection."""
import json
import numpy as np
import torch
from run import ROOT,HERE,save,sha,now


def main():
    p=json.loads((HERE/'protocol-count.json').read_text());out=ROOT/'artifacts/research/20261114/ripper/count';out.mkdir(exist_ok=False)
    pair=np.triu_indices(81,1);rows=[];supports=[]
    for e in p['inputs']:
        src=ROOT/e['train_path'];gp=ROOT/e['learned_gap_path'];assert sha(src)==e['train_sha256'] and sha(gp)==e['learned_gap_sha256']
        sol=np.load(src)['solutions'];assert sol.shape==(e['k']*100,81)
        counts=(sol[:,pair[0]]==sol[:,pair[1]]).sum(0)
        support=counts==0
        # Freeze computed support on disk before reading the learned or oracle support.
        d=out/f'k{e["k"]}-s{e["seed"]}';d.mkdir();np.savez_compressed(d/'counts.npz',counts=counts,support=support)
        model=torch.load(gp,weights_only=True,map_location='cpu');learned=model['W'][:,0,0].numpy()>0
        rows.append({'k':e['k'],'seed':e['seed'],'training_rows':len(sol),'edges':int(support.sum()),'equal_learned_gap':bool(np.array_equal(support,learned)),'support_hamming_distance':int(np.count_nonzero(support!=learned)),'support_sha256':sha(d/'counts.npz'),'training_sha256':sha(src),'learned_gap_sha256':sha(gp)})
        supports.append(support)
    # Geometry diagnostic follows completed supportselection for every dataset.
    cells=np.arange(81);r,c=cells//9,cells%9
    peer=(r[:,None]==r[None])|(c[:,None]==c[None])|((r[:,None]//3==r[None]//3)&(c[:,None]//3==c[None]//3));truth=peer[pair]
    for row,support in zip(rows,supports):
        row.update(exact_peer_graph=bool(np.array_equal(support,truth)),missing_true_edges=int(np.count_nonzero(truth&~support)),extra_edges=int(np.count_nonzero(support&~truth)))
    allmatch=all(r['equal_learned_gap'] for r in rows)
    rec={'status':'validated','utc':now(),'protocol_sha256':sha(HERE/'protocol-count.json'),'wrapper_sha256':sha(__file__),'decision':'supported_for_all18_frozen_inputs' if allmatch else 'falsified_for_frozen_inputs','datasets':18,'equal_learned_gap':sum(r['equal_learned_gap'] for r in rows),'exact_peer_graph':sum(r['exact_peer_graph'] for r in rows),'all_count_graphs_identical':all(np.array_equal(supports[0],s) for s in supports),'rows':rows,'inference_performed':False,'interpretation':'If matches, optimizer training was not necessary to recover this graph from these augmented trainingdatasets. This does not establish that10unaugmentedpuzzles suffice. SpatialSudokuaugmentation is inherited taskknowledge.','limitations':['Finite frozen dataset audit, not universal recovery theorem.','Allgraphs identical would not create18independent structures.','Graph equality alone is not byte-identical inference when coupling scales differ.']}
    save(out/'audit.json',rec);save(ROOT/'results/research/ripper/count.json',rec)
    print(json.dumps(rec,indent=2))


if __name__=='__main__':main()
