"""Check pre-run correction lineage and literal pairwise synthetic references."""
from pathlib import Path
from fractions import Fraction
import ast,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    protocol=json.loads((HERE/'protocol-null-replay.json').read_text());root=HERE.parents[2]
    for p,d in protocol['code_sha256'].items():assert sha(root/p)==d
    old=(HERE/'null_replay_study.py').read_text();expected=old.replace('null_replay_study.py','null_replay_study_corrected.py').replace('audit_null_replay.py','audit_null_replay_corrected.py').replace('from audit_null_replay import','from audit_null_replay_corrected import').replace('protocol-null-replay.json','protocol-null-replay-corrected.json')
    before='cells=np.flatnonzero(masks[row]);view=np.ix_(cells,cells);joint[view]+=1;equal[view]+=(values[row,cells,None]==values[row,None,cells])'
    after='cells=np.flatnonzero(masks[row]);view=np.ix_(cells,cells);joint[view]+=1;labels=values[row,cells];equal[view]+=(labels[:,None]==labels[None,:])'
    assert expected.count(before)==1;expected=expected.replace(before,after)
    expected=expected.replace("inputs=[PARENT/'audit.json',HERE/'protocol-permuted-noise.json']","inputs=[PARENT/'audit.json',HERE/'protocol-permuted-noise.json',HERE/'protocol-null-replay.json',HERE/'null-replay-pre-run-correction.json',HERE/'null_replay_study.py',HERE/'audit_null_replay.py',HERE/'verify_null_replay_correction.py']")
    corrected=(HERE/'null_replay_study_corrected.py').read_text();assert expected==corrected and ast.dump(ast.parse(expected))==ast.dump(ast.parse(corrected))
    assert (HERE/'audit_null_replay_corrected.py').read_text()==(HERE/'audit_null_replay.py').read_text().replace('from null_replay_study import','from null_replay_study_corrected import')
    cells=np.array([x for x in range(81)if x%3==0][:25]);i,j=np.triu_indices(81,1);patterns=[np.arange(25),np.zeros(25,int),np.arange(25)%9];checked=0
    for labels in patterns:
        literal=np.array([[labels[a]==labels[b]for b in range(25)]for a in range(25)],bool)
        producer=labels[:,None]==labels[None,:];assert producer.shape==(25,25)and np.array_equal(producer,literal)
        joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint);view=np.ix_(cells,cells);joint[view]+=1;equal[view]+=producer
        mask=np.zeros(81,bool);mask[cells]=True;values=np.full(81,-1,int);values[cells]=labels
        auditor_joint=np.logical_and(mask[i],mask[j]);auditor_equal=auditor_joint&(values[i]==values[j])
        literal_joint=np.array([a in cells and b in cells for a,b in zip(i,j)])
        literal_equal=np.array([a in cells and b in cells and values[a]==values[b]for a,b in zip(i,j)])
        assert np.array_equal(joint[i,j],auditor_joint)and np.array_equal(equal[i,j],auditor_equal)
        assert np.array_equal(auditor_joint,literal_joint)and np.array_equal(auditor_equal,literal_equal);checked+=625
    from null_replay_policy import thresholds
    threshold_cases=0
    for q0 in [Fraction(42288940519,64000000000000),Fraction(9,1600),Fraction(311,25600)]:
        t=thresholds(q0,30);a=Fraction(1,9)/q0;b=Fraction(8,9)/(1-q0)
        for m in range(31):
            for count in range(m+1):assert(a**count*b**(m-count)>=648000)==(count>=t[m]);threshold_cases+=1
    result=dict(status='passed',original_protocol_sha256=sha(HERE/'protocol-null-replay.json'),literal_matrix_entries_checked=checked,synthetic_patterns=['all_distinct_arbitrary_symbols','all_same','mixed_nine_symbols'],exact_small_exposure_threshold_cases=threshold_cases,no_counterfactual_outcome_access=True)
    (HERE/'null-replay-pre-run-correction-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
