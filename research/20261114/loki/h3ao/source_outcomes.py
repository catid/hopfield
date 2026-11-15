"""Seal every nested source control and primary fit before target diagnosis."""
import hashlib
import numpy as np
from common import SRC,OUT,frozen,read,save,sha,now
def main():
    frozen();supports=[];arrays=[];rows=[]
    for index in range(8):
        folder=OUT/f'b{index}';fit=read(folder/'model.json');learned=np.load(folder/'model.npz')
        for dose in(16,32,64):
            path=folder/f'd{dose}/count-model.npz';control=np.load(path);F=control['F'];digest=hashlib.sha256(F.tobytes()).hexdigest();matches=[r['group']for r in supports if r['family']==index//4 and r['support_sha256']==digest]
            if matches:group=matches[0];assert np.array_equal(F,arrays[group])
            else:group=len(arrays);arrays.append(F);dest=OUT/f'support{group}';dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'F.npz',F=F);supports.append(dict(group=group,family=index//4,support_sha256=digest,F_npz_sha256=sha(dest/'F.npz'),members=[]))
            member=dict(index=index,dose=dose,method='optimizer-free count/exposure/line control');supports[group]['members'].append(member);rows.append(dict(member,group=group,source_model_sha256=sha(path),role='matched_primary_control'if dose==32 else'descriptive'))
            if dose==32:
                assert np.array_equal(F,learned['F']);member=dict(index=index,dose=dose,method='BP AdamW positive-F');supports[group]['members'].append(member);rows.append(dict(member,group=group,source_model_sha256=sha(folder/'model.npz'),role='primary',source_eligible=fit['source_eligible'],native_source_delta=fit['native_source_delta'],count_support_identity=True))
    save(OUT/'source-outcomes.json',dict(status='complete_source_only',utc=now(),rows=rows,supports=supports,new_BP_fits=8,unfitted_descriptive_BP_doses=[16,64],all_source_control_models_retained=True,no_target_truth_labels_or_scores_read=True,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
if __name__=='__main__':main()
