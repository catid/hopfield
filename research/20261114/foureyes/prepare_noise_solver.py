"""Map every frozen H4p N600 model to unique support/settings searches."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/20261114/foureyes/noise_solver'
SOURCE=ROOT/'artifacts/research/20261114/foureyes/chernoff_replication'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    protocol=HERE/'noise_solver.json';p=json.loads(protocol.read_text());assert not OUT.exists();OUT.mkdir(parents=True);(OUT/'models').mkdir()
    for name,expected in p['frozen_inputs'].items():assert sha(ROOT/name)==expected
    rows=list(csv.DictReader((ROOT/'results/research/foureyes/chernoff_replication/per_bank.csv').open()))
    manifest=json.loads((SOURCE/'inputs.json').read_text());jobs=[];mappings=[];models=[];seen={};evidence={}
    for entry in manifest['shards']:
        path=SOURCE/f'shard-{entry["shard"]:02d}/counts.npz';receipt=json.loads((path.parent/'receipt.json').read_text());assert sha(path)==receipt['output_sha256']['counts.npz'];evidence[str(path.relative_to(ROOT))]=sha(path)
        z=np.load(path)
        for k,bank in enumerate(entry['banks']):
            counts=z['n600'][k]
            thresholds={'gap':np.full(3240,z['t600'][k]),'restricted_gap':np.full(3240,z['restricted_t600'][k]),'chernoff9':z['law9_t600'][k]}
            for arm in p['arms']:
                support=counts<=thresholds[arm];digest=hashlib.sha256(support.tobytes()).hexdigest();requested=f'b{bank}-{arm}'
                audit=next(r for r in rows if int(r['bank'])==bank and r['n']=='600' and r['arm']==arm);assert digest==audit['support_sha256']
                if digest not in seen:
                    actual=f'm{len(seen):03d}';seen[digest]=actual;file=OUT/'models'/f'{actual}.npz'
                    np.savez_compressed(file,support=support,counts=counts,thresholds=thresholds[arm])
                    jobs.append(dict(job_id=actual,seed=p['search_seed'],arm=arm,bank=bank,n=600,threshold='frozen-vector',model_path=str(file.relative_to(ROOT)),model_sha256=sha(file),support_sha256=digest,status='pending'))
                mappings.append(dict(requested=requested,actual=seen[digest],bank=bank,arm=arm,seed=p['search_seed'],support_sha256=digest,reason='Identical support,zero biases,unit1,cohort,search seed and every search setting. Logical bank weights retained; mapped runs not independent.'))
                models.append(dict(bank=bank,arm=arm,support_sha256=digest,representative=seen[digest]))
    assert len(models)==1536 and len(jobs)==70
    result=dict(jobs=jobs,mappings=mappings,models=models,unique_searches=len(jobs),logical_cells=len(models),protocol_sha256=sha(protocol),source_sha256=sha(__file__),construction_inputs_sha256=evidence)
    (OUT/'prepared.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(unique=len(jobs),logical=len(models))))

if __name__=='__main__':main()
