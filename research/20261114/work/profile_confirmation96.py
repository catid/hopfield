"""Bounded read-only timing profile, first sixteen frozen job identifiers."""
import json
import study as S

D=S.OUT/'loop80_confirmation'
p=json.loads((D/'protocol.json').read_text());jobs=sorted(p['jobs'],key=lambda j:j['id'])[:16];rows=[]
for job in jobs:
    path=D/'runs'/job['id']/'result.json';r=json.loads(path.read_text());assert r['status']=='complete' and r['job']==job and r['protocol_sha256']==S.sha(D/'protocol.json')
    rows.append({'job':job['id'],'receipt_sha256':S.sha(path),'job_seconds':r['job_seconds'],'archive_seconds':r['archive_seconds'],'hidden_seconds':r['hidden_seconds_shared'],'graph_seconds':sum(sum(b['block_cpu_graph_seconds']) for b in r['batch_timings']),'score_seconds':sum(sum(b['block_gpu_scoring_seconds']) for b in r['batch_timings']),'native_seconds':sum(sum(b['block_native_descent_seconds']) for b in r['batch_timings'])})
result={'status':'passed','selection':'First16jobs in sorted frozenjobid order; timing fields only, no outcome arrays read.','fresh_trajectories':0,'rows':rows,'totals':{k:sum(r[k] for r in rows) for k in ['job_seconds','archive_seconds','hidden_seconds','graph_seconds','score_seconds','native_seconds']},'source_sha256':S.sha(__file__),'scope':'Descriptive stage accounting for future bottleneck selection; not a comparative speed endpoint or new scientific evidence.'};S.save(D/'timing-profile.json',result);print(json.dumps(result['totals'],indent=2))
