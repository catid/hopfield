"""Frozen one-base count recovery, with orbit coverage audited afterward."""
import csv
import json
import numpy as np
from run import ROOT,HERE,save,sha,now


def main():
    p=json.loads((HERE/'protocol-onebase.json').read_text());base=ROOT/'artifacts/research/20261114/ripper';out=base/'onebase';out.mkdir(exist_ok=False)
    pair=np.triu_indices(81,1);rows=[];supports=[]
    for e in p['inputs']:
        src=ROOT/e['path'];assert sha(src)==e['sha256'];solutions=np.load(src)['solutions'][:100]
        for variants in p['variants_grid']:
            counts=(solutions[:variants+1,pair[0]]==solutions[:variants+1,pair[1]]).sum(0);support=counts==0
            path=out/f's{e["seed"]}-v{variants}.npz';np.savez_compressed(path,counts=counts,support=support)
            rows.append({'seed':e['seed'],'variants':variants,'training_rows':variants+1,'retained_edges':int(support.sum()),'support_sha256':sha(path)})
            supports.append(support)
    # Position/oracle information is consulted only after all edge selections are saved.
    a,b=pair;r1,r2=a//9,b//9;c1,c2=a%9,b%9
    same_r,same_c,same_band,same_stack=r1==r2,c1==c2,r1//3==r2//3,c1//3==c2//3
    same_box=same_band&same_stack;line=same_r|same_c;peer=line|same_box
    orbits={'peer_line_within_box':line&same_box,'peer_line_outside_box':line&~same_box,'peer_box_only':same_box&~line,'nonpeer_same_band_or_stack':~peer&(same_band|same_stack),'nonpeer_different_band_and_stack':~same_band&~same_stack}
    assert np.all(sum(m.astype(int) for m in orbits.values())==1)
    orbit_rows=[];reuse=[]
    for row,support in zip(rows,supports):
        row.update(exact_peer_graph=bool(np.array_equal(support,peer)),missing_true_edges=int(np.count_nonzero(peer&~support)),extra_edges=int(np.count_nonzero(~peer&support)))
        for name,mask in orbits.items():orbit_rows.append({'seed':row['seed'],'variants':row['variants'],'orbit':name,'pairs':int(mask.sum()),'retained_never_equal':int(np.count_nonzero(mask&support)),'observed_equal_at_least_once':int(np.count_nonzero(mask&~support))})
        if row['variants']==99:
            ref=np.load(base/f'count/k10-s{row["seed"]}/counts.npz')['support']
            assert np.array_equal(ref,support)
            d=base/f'count_search/jobs/k10-s{row["seed"]}-count_sparse';rec=json.loads((d/'receipt.json').read_text());assert rec['used_hbar']==1 and rec['statistics']['tau_train']==0 and rec['statistics']['ebar']==0
            reuse.append({'seed':row['seed'],'variants':99,'source_receipt':str((d/'receipt.json').relative_to(ROOT)),'source_receipt_sha256':sha(d/'receipt.json'),'exact_support_match':True,'same_inference_arguments':'Unit1,S/B determinedbysamegraph,tau0,ebar0,temp.4/.31,theta.45,C4,seed unchanged,full-all[40000,50000),cap262144','solved':rec['solved'],'n':10000,'wrong_stops':rec['wrong_stops'],'new_search':False})
    exact=sum(r['exact_peer_graph'] for r in rows if r['variants']==99)
    first={s:next((r['variants'] for r in rows if r['seed']==s and r['exact_peer_graph']),None) for s in p['seeds']}
    rec={'status':'validated','utc':now(),'protocol_sha256':sha(HERE/'protocol-onebase.json'),'wrapper_sha256':sha(__file__),'primary_exact_groups':exact,'primary_total_groups':6,'decision':'supported_for_all6_frozen_groups' if exact==6 else 'falsified_for_frozen_groups','first_exact_prespecified_prefix':first,'rows':rows,'orbit_coverage':orbit_rows,'inference_reuse':reuse,'new_gpu_searches':0,'interpretation':'When exact, onebaseplusstoredspatialaugmentations suffices fordirectcount recovery. This isaugmentation taskknowledge,notoptimizer learning oroneunaugmentedexample generalization.','limitations':['Sixfixedgroups; no universal probabilisticguarantee.','Reusedinference isnotnewindependent experimentalreplication.','Allmatchinggraphs identical; no independentstructuralmultiplicity.']}
    save(out/'audit.json',rec);public=ROOT/'results/research/ripper';save(public/'onebase.json',rec)
    for name,rs in [('onebase_prefixes.csv',rows),('onebase_orbits.csv',orbit_rows)]:
        with (public/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    summary=json.loads((public/'summary.json').read_text());summary['onebase_count']=rec;save(public/'summary.json',summary)
    print(json.dumps({'decision':rec['decision'],'first_exact_prefix':first,'extra_edges_by_prefix':{v:[r['extra_edges'] for r in rows if r['variants']==v] for v in p['variants_grid']}}))


if __name__=='__main__':main()
