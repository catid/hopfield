"""Publish every full-class truth case, source draw and sealed membership answer."""
import csv
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3be'


def main():
    frozen();g=read(OUT/'capacity-gate.json');files=['capacity-gate.json','study-result.json','campaign.log','preparation.json','runtime-parity.json','score-values.json','score-table.json']+[str(path.relative_to(OUT)) for path in sorted((OUT/'cases').rglob('*')) if path.is_file()]
    for rel in files:
        target=PUBLIC/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,target)
    audit=verify(ROOT,False);keys=['decision','strata','results','eligible_truths','ineligible_truths','identified','membership_labels'];s=dict(status='validated',utc=now(),hypothesis='H3be',**{k:g[k] for k in keys},aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={rel:sha(PUBLIC/rel) for rel in files});save(PUBLIC/'summary.json',s)
    rows=[{k:v for k,v in r.items() if k!='steps'} for r in s['results']];keys=list(dict.fromkeys(k for row in rows for k in row))
    with (PUBLIC/'all-truth-results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    print(dict(decision=s['decision'],identified=s['identified'],strata=s['strata'],membership_labels=s['membership_labels'],audit=audit))


if __name__=='__main__':main()
