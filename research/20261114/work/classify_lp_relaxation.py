"""Post-outcome exact all-open/endpoint identities for every H2z output."""
import json
import time
import numpy as np
import study as S
from lp_relaxation import PP,DEST,ARMS,protocol


def main():
    started=time.perf_counter();p=protocol();rows=[]
    for job in p['jobs']:
        run=DEST/'runs'/job['id'];receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256']
        with np.load(run/'outputs.npz') as x:pred=x['predictions']
        with np.load(DEST/f"shard{job['shard']:02d}.npz") as x:t=x['tokens'];indices=x['indices']
        for v,arm in enumerate(ARMS):
            dense=(pred[v]==(t>0)).all((1,2));endpoint=(pred[v]==(t>=2)).all((1,2))
            rows.append({'job':job['id'],'model_seed':job['seed'],'arm':arm,'n':256,'exact_allopen':int(dense.sum()),'exact_endpoints':int(endpoint.sum()),'allopen_mismatch_input_indices':indices[~dense].tolist() if arm=='lp_lambda4' else None,'source_outputs_sha256':receipt['outputs_sha256']})
    S.save(DEST/'output-identities.json',{'status':'passed','utc':S.utc(),'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'scope':'Post-outcome exact array equality against supplied-open and supplied-endpoint boards. No newinference, independentinputs, labels, orrescue.','model_input_states':36864,'rows':rows,'totals':{arm:{key:sum(r[key] for r in rows if r['arm']==arm) for key in ['n','exact_allopen','exact_endpoints']} for arm in ARMS},'seconds':time.perf_counter()-started})


if __name__=='__main__':main()
