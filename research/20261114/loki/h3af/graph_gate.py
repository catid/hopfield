"""External post-compilation graph validation; never an input to compilation."""
import time
import numpy as np
from common import SRC,OUT,read,save,sha,now,frozen

def main():
    p=frozen();assert read(OUT/'compilation-gate.json')['status']=='complete_verified';rows=[]
    for bank,leaper in enumerate([[2,3],[2,5]]):
        wall=time.perf_counter();cpu=time.process_time();folder=OUT/f'b{bank}';m=np.load(folder/'model.npz');q=15;i,j=np.triu_indices(q*q,1);dr=j//q-i//q;dc=abs(j%q-i%q);a,b=leaper;truth=(dr==0)|(dc==0)|(dr==dc)|((dr==a)&(dc==b))|((dr==b)&(dc==a));actual=m['F'][i,j].astype(bool);false=int((actual&~truth).sum());missed=int((~actual&truth).sum());passed=false==missed==0;rows.append(dict(bank=bank,status='passed'if passed else'failed',false_edges=false,missed_edges=missed,true_forbidden_pairs=int(truth.sum()),wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,model_sha256=sha(folder/'model.npz')))
    passed=all(r['status']=='passed'for r in rows);save(OUT/'external-graph-gate.json',dict(status='passed'if passed else'failed',utc=now(),families=rows,external_geometry_only_used_after_all_compilation_seals=True,complete_zero_F_universe_equals_complete_true_target_universe=passed,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)));assert passed,'Graph mismatch: complete target universe unavailable, no cohort or GPU study'
if __name__=='__main__':main()
