"""Construct all three arms from inherited16positive banks before target access."""
import hashlib
import itertools
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from construct import construct,views
def valid(board,leaper):
    assert board.sum()==12
    for a,b in itertools.combinations(np.flatnonzero(board),2):
        dr,dc=abs(int(b)//12-int(a)//12),abs(int(b)%12-int(a)%12);assert dr and dc and dr!=dc and(dr,dc)not in(leaper,leaper[::-1])
def main():
    p=frozen();supports=[];arrays=[];rows=[]
    for cell in range(144):
        x=np.zeros(144,bool);x[cell]=True;r,c=divmod(cell,12);mapped={(r,c),(c,11-r),(11-r,11-c),(11-c,r),(r,11-c),(11-c,11-r),(11-r,c),(c,r)};assert{int(np.flatnonzero(v)[0])for v in views(x)}=={12*a+b for a,b in mapped}
    for index,cfg in enumerate(p['banks']):
        d=np.load(ROOT/cfg['design_path']);y=d['labels'];assert y.shape==(16,144);family=cfg['family'];leaper=tuple(p['families'][family]['leaper']);assert{(0,1),(1,-1),(1,0),(1,1)}<=set(map(tuple,d['lines']))
        for board in y:
            valid(board,leaper)
            for v in views(board):valid(v,leaper)
        baseline=np.load(ROOT/cfg['filtered_model_path'])['F'];models=[('published_filtered',baseline,dict(raw_labels=16,attempted_views=16,unique_views=16,identity_imported=True))]
        for augment,arm in[(False,'raw_unfiltered'),(True,'D4_unfiltered')]:
            F,receipt=construct(y,d['offsets'],d['lines'],augment);models.append((arm,F,receipt))
        assert(models[2][1]<=models[1][1]).all()
        for arm,F,receipt in models:
            digest=hashlib.sha256(F.tobytes()).hexdigest();same=[r['group']for r in supports if r['family']==family and r['support_sha256']==digest]
            if same:group=same[0];assert np.array_equal(F,arrays[group])
            else:
                assert len(arrays)<24,'Distinct support gate exceeded';group=len(arrays);arrays.append(F);dest=OUT/f'support{group}';dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'F.npz',F=F);supports.append(dict(group=group,family=family,support_sha256=digest,F_npz_sha256=sha(dest/'F.npz'),members=[]))
            folder=OUT/f'b{index}/{arm}';folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'model.npz',F=F,J=F.astype(np.float32),h=np.zeros(169,np.float32),q=np.array(13),unit=np.array(1.));save(folder/'construction.json',dict(index=index,family=family,arm=arm,group=group,**receipt,model_sha256=sha(folder/'model.npz')));member=dict(index=index,arm=arm);supports[group]['members'].append(member);rows.append(dict(member,family=family,bank=index%4,group=group,primary=arm=='D4_unfiltered',model_sha256=sha(folder/'model.npz')))
    save(OUT/'source-outcomes.json',dict(status='complete_source_only',utc=now(),rows=rows,supports=supports,source_D4_label_and_coordinate_gates_passed=True,all24logical_outcomes_retained=True,no_new_labels_or_BP_fits=True,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
if __name__=='__main__':main()
