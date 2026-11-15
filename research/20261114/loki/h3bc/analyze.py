"""Export all source/query cases, empty models, failures and inherited ineligibles."""
import csv
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3bc'


def main():
    frozen();g=read(OUT/'capacity-gate.json');files=['capacity-gate.json','study-result.json','campaign.log','preparation.json','ordinary13.npy','predictions-bits.npy']+[str(path.relative_to(OUT)) for path in sorted((OUT/'cases').rglob('*')) if path.is_file()]
    for rel in files:
        target=PUBLIC/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,target)
    audit=verify(ROOT,False);s=dict(status='validated',utc=now(),hypothesis='H3bc',**{k:g[k] for k in ['decision','results','inherited_truth_grid','identified','membership_labels']},aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={rel:sha(PUBLIC/rel) for rel in files});save(PUBLIC/'summary.json',s)
    rows=[{k:v for k,v in r.items() if k!='steps'} for r in s['results']]
    with (PUBLIC/'label-results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(dict(decision=s['decision'],identified=s['identified'],membership_labels=s['membership_labels'],audit=audit))


if __name__=='__main__':main()
