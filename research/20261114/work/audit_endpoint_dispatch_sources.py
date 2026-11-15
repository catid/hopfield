"""Independent archived source gates/warmups, including dispatcher trajectories."""
from functools import lru_cache
import json
import time
import numpy as np
from audit_endpoint_dispatch_replays import sha,load,check_batch,MODES


def source_check(args,gate_only=False):
    root,directory,p,name,cohorts=args;started=time.perf_counter()
    source=next(s for s in p['sources'] if s['name']==name);source_dir=root/source['directory'];source_ph=sha(source_dir/'protocol.json')
    receipts={};folders={}
    for mode in ['LP96','BP_geometry_LP96']:
        folder=source_dir/'runs'/f'pair0-{mode}';assert sha(folder/'result.json')==source['cohort_receipts_sha256'][mode]
        receipt=json.loads((folder/'result.json').read_text());assert receipt['protocol_sha256']==source_ph
        receipts[mode]={(r['job_id'],r['batch']):r for r in receipt['rows']};folders[mode]=folder
    @lru_cache(maxsize=4)
    def archived(mode,job,batch):return load(folders[mode],receipts[mode][job,batch])
    tokens={}
    for shard in [0,15]:
        ds=source['datasets'][shard];assert sha(root/ds['path'])==ds['sha256']
        with np.load(root/ds['path'],allow_pickle=False) as z:tokens[shard]=z['tokens']
    cases=candidates=states=0;maximum=0.
    for cohort in [c for c in cohorts if c['kind'] in ['gate','warmup'] and c['source_name']==name]:
        actual_ph=p['compatibility_packet_sha256'] if cohort['kind']=='gate' else sha(directory/'protocol.json')
        assert cohort['protocol_sha256']==actual_ph and cohort['source_gate'] and cohort['mode'] in MODES
        folder=root/cohort['directory']
        for row in cohort['rows']:
            job,batch=row['job_id'],row['batch'];tt=tokens[row['shard']][batch*32:(batch+1)*32];actual=load(folder,row)
            lp,bp=archived('LP96',job,batch),archived('BP_geometry_LP96',job,batch)
            assert row['source_parity']['status']=='passed' and row['fresh_array_receipt'] is None
            _,_,_,_,counts=check_batch(folder,row,actual,lp,bp,folders['LP96'],receipts['LP96'][job,batch],tt,cohort['mode'],actual_ph,source_ph)
            c,s,e=counts;cases+=len(tt);candidates+=c;states+=s;maximum=max(maximum,e)
    assert cases==(1152 if gate_only else 1728)
    if gate_only:assert all(c['kind']=='gate' for c in cohorts)
    return {'source':name,'status':'passed','source_gate_and_warmup_model_copies_checked':cases,'candidate_comparisons':candidates,'active_trace_states':states,'maximum_candidate_score_error':maximum,'seconds':time.perf_counter()-started}
