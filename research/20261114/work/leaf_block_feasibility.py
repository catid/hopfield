"""Freeze H2ag: all retained leaves plus complete leafless false-positive components."""
import argparse
from collections import deque
import json
import numpy as np
import study as S
from mine_leaf_negatives import PP as PARENT,DEST as SOURCE,TRAIN,PRIOR,protocol as parent_protocol

DEST=S.OUT/'leaf_block_feasibility'
PP=S.HERE/'leaf-block-feasibility-protocol.json'


def groups(mask):
    left=set(map(tuple,np.argwhere(mask)));result=[]
    while left:
        start=min(left);left.remove(start);queue=[start];part=[]
        while queue:
            r,c=queue.pop();part.append((r,c))
            for pos in [(r-1,c),(r,c+1),(r+1,c),(r,c-1)]:
                if pos in left:left.remove(pos);queue.append(pos)
        result.append(sorted(part))
    return result


def freeze():
    if PP.exists():raise FileExistsError(PP)
    p=parent_protocol();DEST.mkdir(exist_ok=True);selection=json.loads((SOURCE/'leaf-selection.json').read_text());verified=json.loads((SOURCE/'independent-source-verification.json').read_text());assert verified['status']=='passed';records=[]
    with np.load(TRAIN/'training-arrays.npz') as z:tokens=z['tokens'];labels=z['labels']
    for seed in range(3):
        record=selection['records'][seed];record['blocks']=[]
        with np.load(SOURCE/f'negatives-seed{seed}.npz') as z:indices=z['indices'];boards=z['boards']
        for item in record['wrong_boards_without_leaf']:
            j=item['negative_row'];i=item['training_index'];b=boards[j];assert int(indices[j])==i
            for cells in groups((tokens[i]==1)&b&~labels[i]):record['blocks'].append({'block_row':len(record['blocks']),'negative_row':j,'training_index':i,'cells':[list(map(int,q)) for q in cells],'size':len(cells),'normalization':'removed cells'})
        assert len(record['blocks'])==2 and sorted(x['size'] for x in record['blocks'])==[8,12]
        record['informative']=bool(record['wrong_boards']>0 and record['eligible_leaves']>0);records.append(record)
    assert sum(r['eligible_leaves'] for r in records)==1216 and sum(len(r['blocks']) for r in records)==6
    S.save(DEST/'move-selection.json',{'status':'frozen','records':records,'source_selection_sha256':S.sha(SOURCE/'leaf-selection.json'),'source_independent_verification_sha256':S.sha(SOURCE/'independent-source-verification.json'),'new_source_trajectories':0,'scope':'Retain all1216 H2af leaf rows. For each of six leafless model-boards, include its complete false-positive4-connected component; source labels only, no inference implementation or target use. All six component rows duplicate existing canonical-label ranking rows and are retained explicitly.'})
    p.update(id='source-leaf-and-component-removal-feasibility',hypothesis_id='H2ag',frozen_utc=S.utc(),hypothesis='All3 fixed source systems admit every old inequality, all1216 strict leaf removability rows and all6 normalized component-removal rows.')
    p['parent_leaf_protocol_sha256']=S.sha(PARENT);p['selection_sha256']=S.sha(DEST/'move-selection.json');p['constraints']['new']='All1216 H2af leaf rows unchanged. Additionally for every leafless board, each full false-positive4-connected component has removal energy<=-number of removed cells. Six component rows duplicate existing ranking rows; no deduplication.'
    p['decision']='Support only if all3 full systems admit independently verified primals. Exact checkable Farkas witness falsifies feasibility. Numerical infeasibility or timeout alone inconclusive. No new mining or target consumption; no claim current one-bit decoder can execute block moves.'
    p['source_files_sha256']={str(path.relative_to(S.ROOT)):S.sha(path) for path in [SOURCE/'leaf-selection.json',SOURCE/'independent-source-verification.json']+[SOURCE/f'negatives-seed{seed}.npz' for seed in range(3)]}
    p['source_sha256'].update({str((S.HERE/name).relative_to(S.ROOT)):S.sha(S.HERE/name) for name in ['leaf_block_feasibility.py','solve_leaf_block_feasibility.py']});S.save(PP,p);S.emit('leaf_block_protocol_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text());parent_protocol();assert p['parent_leaf_protocol_sha256']==S.sha(PARENT) and p['selection_sha256']==S.sha(DEST/'move-selection.json')
    for path,h in {**p['source_sha256'],**p['source_files_sha256']}.items():assert S.sha(S.ROOT/path)==h,path
    return p


if __name__=='__main__':freeze()
