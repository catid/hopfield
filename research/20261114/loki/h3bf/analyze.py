"""Compress exact immutable ledger bytes and retain every hard-tail case."""
import csv
import gzip
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3bf'


def main():
    frozen();g=read(OUT/'capacity-gate.json');files=['capacity-gate.json','study-result.json','case-results.json','campaign.log','preparation.json','score-values.json','score-table.json','ledger-gate.json','synthetic-ledger.jsonl']+[str(path.relative_to(OUT)) for path in sorted((OUT/'legacy-format-gate').glob('*.json'))]
    for rel in files:
        target=PUBLIC/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,target)
    raw=(OUT/'query-ledger.jsonl').read_bytes();(PUBLIC/'query-ledger.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0));save(PUBLIC/'ledger-export.json',dict(raw_sha256=sha(OUT/'query-ledger.jsonl'),raw_bytes=len(raw),gzip_sha256=sha(PUBLIC/'query-ledger.jsonl.gz'),gzip_bytes=(PUBLIC/'query-ledger.jsonl.gz').stat().st_size,compression='gzip mtime0; exact raw bytes preserved, receipts not regenerated'));files+=['query-ledger.jsonl.gz','ledger-export.json'];audit=verify(ROOT,False);keys=['decision','truth_results','cases','identified_cases','membership_labels','ledger_records','ledger_sha256'];s=dict(status='validated',utc=now(),hypothesis='H3bf',**{k:g[k] for k in keys},aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={rel:sha(PUBLIC/rel) for rel in files});save(PUBLIC/'summary.json',s)
    with (PUBLIC/'truth-results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(s['truth_results'][0]));writer.writeheader();writer.writerows(s['truth_results'])
    print(dict(decision=s['decision'],truths_passing=sum(r['decision']=='supported' for r in s['truth_results']),identified=s['identified_cases'],labels=s['membership_labels'],audit=audit))


if __name__=='__main__':main()
