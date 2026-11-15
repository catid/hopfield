"""Delete inherited finite classes using selected positive co-occurrences only.

This process never opens target truth caches/graphs, target inputs, capacities,
or the full N13 population. Its only new data are the sealed selected positives.
"""
import hashlib
import itertools
import os
import sys
from pathlib import Path
import numpy as np
from common import ROOT,SRC,OUT,read,save,sha,now
from enumerators import eligibility

def main():
    p=read(SRC/'protocol.json');assert not(OUT/'repair-seal.json').exists();source=read(OUT/'source-positive-seal.json');assert source['status']=='passed'and source['protocol_sha256']==sha(SRC/'protocol.json');rows=[];supports=[];support_arrays=[];witnesses=[]
    allowed={str((ROOT/cfg[key]).resolve())for cfg in p['models']for key in ['model_path','source_design_path']}
    allowed|={str((OUT/f'b{b}/{name}').resolve())for b in range(4)for name in ['positive-boards.npy','positive-population-indices.npy']}
    allowed|={str((OUT/f'b{b}/d{dose}/model.npz').resolve())for b in range(4)for dose in p['acquisition']['doses']}
    opened=set()
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2];reading=not isinstance(mode,str)or not any(c in mode for c in 'wax')
        if reading and (path.is_relative_to(ROOT/'artifacts')or path.is_relative_to(ROOT/'results')):
            assert str(path)in allowed, 'Repair forbidden data access: '+str(path)
            opened.add(str(path.relative_to(ROOT)))
    sys.addaudithook(guard)
    # Explicit model/design identities avoid a generic guard that might open target data.
    for bank,cfg in enumerate(p['models']):
        for key in ['model_path','source_design_path']:assert sha(ROOT/cfg[key])==p['source_data_sha256'][cfg[key]]
        base=dict(np.load(ROOT/cfg['model_path']));design=np.load(ROOT/cfg['source_design_path']);finite={tuple(x)for x in design['offsets']};folder=OUT/f'b{bank}';positives=np.load(folder/'positive-boards.npy');indices=np.load(folder/'positive-population-indices.npy');assert sha(folder/'positive-boards.npy')==source['banks'][bank]['boards_sha256']and sha(folder/'positive-population-indices.npy')==source['banks'][bank]['indices_sha256'];previous=base['F'];observed={}
        for dose in p['acquisition']['doses']:
            observed={}
            for position,board in enumerate(positives[:dose]):
                for a,b in itertools.combinations(np.flatnonzero(board),2):
                    offset=(int(b)//13-int(a)//13,int(b)%13-int(a)%13)
                    if offset in finite and offset not in observed:observed[offset]=dict(bank=bank,dose=dose,offset=list(offset),positive_position=position,population_index=int(indices[position]),source_cells=[int(a),int(b)])
            m={k:v.copy()for k,v in base.items()};removed=set()
            for a,b in itertools.combinations(range(225),2):
                offset=(b//15-a//15,b%15-a%15)
                if m['F'][a,b]and offset in observed:m['F'][a,b]=m['F'][b,a]=0;m['J'][a,b]=m['J'][b,a]=0;removed.add(offset)
            assert(m['F']<=previous).all()and(m['F']<=base['F']).all();previous=m['F'];proof=eligibility(m);dest=folder/f'd{dose}';dest.mkdir(exist_ok=True);np.savez_compressed(dest/'model.npz',**m);key=hashlib.sha256(m['F'].tobytes()).hexdigest();matches=[i for i,r in enumerate(supports)if r['support_sha256']==key]
            if matches:group=matches[0];assert np.array_equal(m['F'],support_arrays[group])
            else:group=len(supports);supports.append(dict(group=group,support_sha256=key,representative_bank=bank,representative_dose=dose,members=[]));support_arrays.append(m['F'].copy())
            supports[group]['members'].append(dict(bank=bank,dose=dose));local=[observed[o]for o in sorted(removed)];witnesses.extend(local);row=dict(bank=bank,dose=dose,support_group=group,removed_finite_offsets=[list(o)for o in sorted(removed)],removed_target_pairs=int((base['F'].sum()-m['F'].sum())//2),model_sha256=sha(dest/'model.npz'),source_base_model_sha256=sha(ROOT/cfg['model_path']),positive_boards_sha256=source['banks'][bank]['boards_sha256'],proof=proof,weights_and_unit_unchanged_except_deleted_edges=True,all_removals_have_positive_witness=True);save(dest/'repair.json',dict(row,witnesses=local));rows.append(row)
    save(OUT/'repair-seal.json',dict(status='complete_source_only_repair',utc=now(),rows=rows,supports=supports,witnesses=witnesses,all_four_banks_and_all_doses_retained=True,no_full_population_or_target_cache_graph_keys_or_scores_accessed=True,artifact_read_allowlist_enforced=True,artifact_read_paths=sorted(opened),protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
if __name__=='__main__':main()
