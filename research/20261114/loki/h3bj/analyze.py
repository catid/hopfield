"""Export exact durable bytes and all paired cases, then audit independently."""
import csv
import gzip
import json
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3bj'


def main():
    frozen();PUBLIC.mkdir(parents=True,exist_ok=True)
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.name not in ['query-ledger.jsonl','predictions-bits.npy']:
            target=PUBLIC/path.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    raw=(OUT/'query-ledger.jsonl').read_bytes();(PUBLIC/'query-ledger.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0));save(PUBLIC/'ledger-export.json',dict(raw_sha256=sha(OUT/'query-ledger.jsonl'),raw_bytes=len(raw),gzip_sha256=sha(PUBLIC/'query-ledger.jsonl.gz'),gzip_bytes=(PUBLIC/'query-ledger.jsonl.gz').stat().st_size,compression='gzip mtime0; exact original bytes preserved'));outcome=json.loads(raw.splitlines()[-1])['data'];audit=verify(ROOT,False);g=read(PUBLIC/'capacity-gate.json');dependency=ROOT/'research/20261114/loki/h3bi/verify_public.py';s=dict(status='validated',utc=now(),hypothesis='H3bj',**outcome,aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,ledger_records=g['ledger_records'],ledger_sha256=g['ledger_sha256'],audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),verifier_dependency_sha256={str(dependency.relative_to(ROOT)):sha(dependency)},public_sha256={str(path.relative_to(PUBLIC)):sha(path) for path in sorted(PUBLIC.rglob('*')) if path.is_file() and path.name not in ['summary.json','README.md','plot-provenance.json'] and path.suffix not in ['png','svg','csv']});save(PUBLIC/'summary.json',s)
    with (PUBLIC/'truth-comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(s['truth_results'][0]));w.writeheader();w.writerows(s['truth_results'])
    print(dict(A=s['A_decision'],B=s['B_decision'],secondary=s['secondary_feasibility_decision'],hybrid=s['hybrid_exact'],baseline=s['baseline_exact'],audit=audit))


if __name__=='__main__':main()
