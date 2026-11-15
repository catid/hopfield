"""Compact audited outcome bits; no boards, checkpoints or hidden labels."""
from pathlib import Path
import hashlib
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[3];RAW=ROOT/'artifacts/research/20261114/foureyes/noise_solver';PUBLIC=ROOT/'results/research/foureyes/noise_solver'
s=json.loads((PUBLIC/'summary.json').read_text());jobs=json.loads((RAW/'queue.json').read_text());good=[];stopped=[];ids=[]
for job in jobs:
    path=RAW/job['job_id']/'moves-262144/boards.npz';digest=hashlib.sha256(path.read_bytes()).hexdigest();assert s['input_sha256'][str(path.relative_to(ROOT))]==digest
    z=np.load(path);ids.append(job['job_id']);good.append(z['valid']);stopped.append(z['stopped'])
out=PUBLIC/'audited-outcome-bits.npz';np.savez_compressed(out,job_ids=np.asarray(ids),valid=np.stack(good),stopped=np.stack(stopped),puzzle=np.arange(220000,230000))
receipt=dict(role='Compact audited scientific outcomes only;70executedmodels x10000puzzles. Mapping in summary retains1536logicalbank/armcells. No target solutions,boards orcheckpoints.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),summary_sha256=hashlib.sha256((PUBLIC/'summary.json').read_bytes()).hexdigest(),export_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
(PUBLIC/'outcome-export-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(out.stat().st_size)
