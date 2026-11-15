"""CPU-only H2bo auditor compatibility on published H2bh/H2bn shard00 seed0."""
import json
from pathlib import Path
import signal
import sys
import tempfile
import time
import numpy as np
root=Path('/home/catid/hopfield')
sys.path.insert(0,str(root/'research/20261114/work'))
from audit_bp_first_fresh_states import job_check,sha
source=root/'artifacts/research/20261114/work/loop80_confirmation'
parent=root/'artifacts/research/20261114/work/bp_first_geometry_cascade'
dest=root/'artifacts/research/20261114/work/bp_first_fresh_confirmation'
started=time.perf_counter()
def timeout(signum,frame):raise TimeoutError('Source CPU archive compatibility180wall-second limit')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,180)
try:
 with tempfile.TemporaryDirectory(prefix='source-format-',dir=dest) as temporary:
  d=Path(temporary);p=json.loads((source/'protocol.json').read_text());job=p['jobs'][0]
  hashes=json.loads((parent/'protocol.json').read_text())['source_jobs'][0]['batch_sha256']
  (d/'protocol.json').write_text(json.dumps(p));ph=sha(d/'protocol.json')
  (d/'datasets.json').write_bytes((source/'datasets.json').read_bytes())
  for arm in ['pl','old_binary']:(d/f'model-seed0-{arm}.npz').write_bytes((source/f'model-seed0-{arm}.npz').read_bytes())
  for mode in ['LP96','BP_geometry_LP96']:
   original=parent/'runs'/f'pair0-{mode}';target=d/'runs'/f'pair0-{mode}';target.mkdir(parents=True)
   r=json.loads((original/'result.json').read_text());r['rows']=[row for row in r['rows'] if row['job_id']==job['id']];r['protocol_sha256']=ph
   for row in r['rows']:
    bi=row['batch'];file=source/'runs'/job['id']/f'batch{bi:02d}.npz'
    assert sha(file)==hashes[file.name] and sha(original/row['file'])==row['file_sha256']
    with np.load(original/row['file'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(file,allow_pickle=False) as z:b={k:z[k] for k in z.files}
    a['couplings']=b['couplings']
    if mode=='LP96':
     a['LP_minimum_flip_margin64']=b['minimum_flip_margin64'][1:];a['LP_energy64']=b['energy64'][1:]
    else:
     a['bp_minimum_flip_margin32']=b['native_minimum_flip_margin32'][0];a['bp_energy64']=b['energy64'][0]
    np.savez(target/row['file'],**a);row['file_sha256']=sha(target/row['file'])
    if mode=='LP96':
     for filename in row['trace_files']:
      assert sha(original/filename)==row['trace_sha256'][filename]
      with np.load(original/filename,allow_pickle=False) as z:trace={k:z[k] for k in z.files}
      payload=json.loads(trace['trace_utf8'].tobytes());payload.update(protocol_sha256=ph,offset=bi*32)
      trace['trace_utf8']=np.frombuffer(json.dumps(payload).encode(),np.uint8)
      np.savez(target/filename,**trace);row['trace_sha256'][filename]=sha(target/filename)
   (target/'result.json').write_text(json.dumps(r))
  print(json.dumps({'event':'source_format_fixture_ready','seconds':time.perf_counter()-started}),flush=True)
  result=job_check((root,d,job,p))
  out={'status':'passed','scope':'CPU-only compatibility of new direct auditor with published actual H2bn shard00 seed0 states and H2bh supplementary diagnostics; no inference or fresh scientific counts.','job':job,'all_final_states':result['all_final_states'],'candidate_literal_checks':result['candidate_literal_checks'],'maximum_margin_error':result['max_final_margin_error'],'maximum_energy_error':result['max_final_energy_error'],'maximum_candidate_error':result['max_literal_error'],'seconds':time.perf_counter()-started,'wall_limit_seconds':180,'fresh_inputs_drawn':0,'source_sha256':sha(root/'research/20261114/work/audit_bp_first_fresh_states.py'),'driver_sha256':sha(Path(__file__))}
  (dest/'source-format-compatibility.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
except BaseException as error:
 (dest/'source-format-compatibility-attempt2-failure.json').write_text(json.dumps({'status':'incomplete','error':repr(error),'seconds':time.perf_counter()-started,'fresh_inputs_drawn':0,'source_sha256':sha(root/'research/20261114/work/audit_bp_first_fresh_states.py')},indent=2)+'\n')
 raise
finally:signal.setitimer(signal.ITIMER_REAL,0)
