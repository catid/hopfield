"""All-job measured timing/provenance supplement; no trajectories."""
import json
import numpy as np
import study as S
from confirmation96_study import PP,DEST,ARMS

p=json.loads(PP.read_text());a=json.loads((DEST/'audit.json').read_text());assert a['status']=='passed';rows=[]
for job in p['jobs']:
    path=DEST/'runs'/job['id']/'result.json';r=json.loads(path.read_text());assert r['status']=='complete' and r['job']==job and r['protocol_sha256']==S.sha(PP)
    rows.append({'job':job,'receipt_sha256':S.sha(path),**{k:r[k] for k in ['input_transfer_seconds','hidden_seconds_shared','diagnostic_seconds','archive_seconds','job_seconds','answer_seconds_by_arm','unique_answer_seconds_by_operation','cpu_return_seconds_by_arm']}})
answer=np.sum([r['answer_seconds_by_arm'] for r in rows],0);assert np.allclose(answer,[a['answer_seconds_by_arm'][arm] for arm in ARMS],rtol=0,atol=1e-10);totals={k:sum(r[k] for r in rows) for k in ['input_transfer_seconds','hidden_seconds_shared','diagnostic_seconds','archive_seconds','job_seconds']};S.save(DEST/'timing-accounting.json',{'status':'passed','protocol_sha256':S.sha(PP),'source_audit_sha256':S.sha(DEST/'audit.json'),'source_sha256':S.sha(__file__),'rows':rows,'totals':totals,'answer_seconds_by_arm':dict(zip(ARMS,answer.tolist())),'model_load_seconds_by_worker':[json.loads((DEST/f'eval-worker{g}.json').read_text())['model_load_seconds'] for g in [0,1]],'scope':'All48original measured receipts, completeprefix/ownCPUreturns and separately reported input/model/hidden/diagnostic/archive costs. No newtiming replay.'});print(json.dumps(totals,indent=2))
