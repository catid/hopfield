"""Audit paired complete compilation identity and conservatively charged costs."""
import csv
import shutil
import time
import numpy as np
from common import ROOT,SRC,OUT,read,save,sha,now,frozen
PUBLIC=ROOT/'results/research/loki/h3ai'
def main():
    start=time.perf_counter();p=frozen();g=read(OUT/'compilation-gate.json');raw=read(OUT/'compiler-result.json');assert g['status']==raw['status']=='complete_verified'and g['rows']==raw['rows']and g['aggregate_cpu_seconds']<=120 and g['wall_seconds']<=180 and not g['monitor_expired'];assert g['protocol_sha256']==sha(SRC/'protocol.json')and raw['source_sha256']==sha(SRC/'compiler.py');v=read(OUT/'verifier-validation.json');assert v['status']=='passed'and len(v['cases'])==44 and all(c['full_solution_set_parity']for c in v['cases']);assert v['binary_sha256']==sha(OUT/'reverse-literal')and v['cpp_source_sha256']==sha(SRC/'reverse_literal.cpp');shared=g['shared_prerequisite_wall_seconds'];assert abs(shared-(g['wall_seconds']-sum(a['compile_wall_seconds']for r in g['rows']for a in r['arms'])))<1e-10
    PUBLIC.mkdir(parents=True,exist_ok=True);files=['compilation-gate.json','compiler-result.json','verifier-validation.json','query-validation.json','campaign.log'];families=[];rows=[]
    for pair in g['rows']:
        bank=pair['bank'];assert pair['order']==p['order'][str(bank)]and [a['arm']for a in pair['arms']]==pair['order']and pair['full_cache_and_model_byte_identity'];local={}
        for comp in pair['arms']:
            arm=comp['arm'];folder=OUT/f'b{bank}/{arm}';assert comp==read(folder/'compilation.json')and comp['model_sha256']==sha(folder/'model.npz')and comp['cache_sha256']==sha(folder/'cache.npz')and comp['source_sha256']==sha(SRC/'compiler.py');cache=np.load(folder/'cache.npz');m=np.load(folder/'model.npz');columns=cache['columns'];boards=cache['boards'];assert comp['zero_F_solutions']==len(boards)and np.array_equal(columns,np.load(folder/'unverified-forward-columns.npy'))and np.array_equal(columns,np.fromfile(OUT/f'verifier-io/N15-b{bank}-{arm}.columns.bin',np.uint8).reshape(-1,15));cells=15*np.arange(15)[None,:]+columns;reconstructed=np.zeros_like(boards);reconstructed[np.arange(len(boards))[:,None],cells]=True;assert np.array_equal(reconstructed,boards)and np.array_equal(np.packbits(boards.T,axis=1,bitorder='little'),cache['index'])and len(np.unique(columns,axis=0))==len(columns)
            counts=np.zeros(len(cells),np.int64)
            for i in range(15):
                for j in range(i+1,15):counts+=m['F'][cells[:,i],cells[:,j]]
            assert not counts.any();prior=ROOT/f'results/research/loki/h3ag/b{bank}';assert sha(prior/'model.npz')==comp['model_sha256']and sha(prior/'cache.npz')==comp['cache_sha256']
            row=dict(bank=bank,family=pair['family'],method='BP AdamW '+arm+' certificate',arm=arm,complete_states=len(columns),compile_seconds=comp['compile_wall_seconds'],shared_cost_charged_seconds=shared,full_charged_seconds=shared+comp['compile_wall_seconds'],forward_seconds=comp['forward_seconds'],reverse_seconds=comp['reverse_seconds'],certificate_seconds=comp['certificate_seconds'],cache_sha256=comp['cache_sha256'],model_sha256=comp['model_sha256']);rows.append(row);local[arm]=row;files += [f'b{bank}/{arm}/{name}'for name in ['model.npz','cache.npz','compilation.json','transfer.json']]
        assert local['dense']['cache_sha256']==local['occupied']['cache_sha256']and local['dense']['model_sha256']==local['occupied']['model_sha256'];ratio=local['dense']['full_charged_seconds']/local['occupied']['full_charged_seconds'];families.append(dict(bank=bank,family=pair['family'],order=pair['order'],decision='supported'if ratio>=1.25 else'falsified',speed_ratio=ratio,required_speed_ratio=1.25,full_cache_and_model_byte_identity=True,post_run_identity_to_prior_H3ag=True,rows=[local['dense'],local['occupied']]))
    for rel in files:
        target=PUBLIC/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,target)
    with(PUBLIC/'compilation.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(status='validated',utc=now(),hypothesis='H3ai',decision='supported'if all(f['decision']=='supported'for f in families)else'falsified',families=families,rows=rows,aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],shared_prerequisite_wall_seconds=shared,GPU_seconds=0,paired_comparisons=2,compilations=4,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),all_full_cache_and_model_identities_verified=True,all_integer_certificates_independently_recomputed=True,all_forward_compiled_reverse_outputs_verified=True,source_validation_cases=44,scope=p['scope'],timing_interpretation=p['trials'],cost=p['cost'],public_sha256={rel:sha(PUBLIC/rel)for rel in files},post_run_audit_wall_seconds=time.perf_counter()-start)
    save(PUBLIC/'summary.json',summary);print(dict(decision=summary['decision'],ratios=[f['speed_ratio']for f in families],CPU=summary['aggregate_CPU_seconds'],wall=summary['wall_seconds']))
if __name__=='__main__':main()
