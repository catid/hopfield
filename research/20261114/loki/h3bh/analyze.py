"""Export exact new ledger bytes while referencing immutable parent evidence."""
import csv
import gzip
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3bh'


def main():
    frozen();g=read(OUT/'capacity-gate.json');files=['capacity-gate.json','study-result.json','case-results.json','campaign.log','preparation.json','score-values.json','score-table.json','continuation-checkpoints.json']
    for rel in files:
        target=PUBLIC/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,target)
    raw=(OUT/'query-ledger.jsonl').read_bytes();(PUBLIC/'query-ledger.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0));save(PUBLIC/'ledger-export.json',dict(raw_sha256=sha(OUT/'query-ledger.jsonl'),raw_bytes=len(raw),gzip_sha256=sha(PUBLIC/'query-ledger.jsonl.gz'),gzip_bytes=(PUBLIC/'query-ledger.jsonl.gz').stat().st_size,compression='gzip mtime0; exact raw bytes preserved, no receipt regeneration'));files+=['query-ledger.jsonl.gz','ledger-export.json'];audit=verify(ROOT,False);keys=['decision','secondary_decision','truth_results','cases','original_identified_cases','identified_cases','inherited_membership_labels','additional_membership_labels','cumulative_membership_labels','new_source_draws','ledger_records','ledger_sha256'];s=dict(status='validated',utc=now(),hypothesis='H3bh',**{k:g[k] for k in keys},aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={rel:sha(PUBLIC/rel) for rel in files});save(PUBLIC/'summary.json',s)
    with (PUBLIC/'truth-continuations.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(s['truth_results'][0]));writer.writeheader();writer.writerows(s['truth_results'])
    print(dict(primary=s['decision'],secondary=s['secondary_decision'],identified=s['identified_cases'],additional=s['additional_membership_labels'],audit=audit))


if __name__=='__main__':main()
