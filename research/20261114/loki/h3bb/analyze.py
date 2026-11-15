"""Export every sealed active/control trajectory before independent audit."""
import csv
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify
PUBLIC=ROOT/'results/research/loki/h3bb'


def main():
    frozen();g=read(OUT/'capacity-gate.json');files=['capacity-gate.json','study-result.json','campaign.log','preparation.json','ordinary13.npy','predictions.npy']+[str(p.relative_to(OUT)) for p in sorted((OUT/'cases').rglob('*.json'))]
    for rel in files:
        target=PUBLIC/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,target)
    audit=verify(ROOT,False)
    s=dict(status='validated',utc=now(),hypothesis='H3bb',**{k:g[k] for k in ['decision','results','active_identified','random_identified','active_labels','random_labels','active_max_labels']},aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={rel:sha(PUBLIC/rel) for rel in files})
    save(PUBLIC/'summary.json',s);rows=[{k:v for k,v in row.items() if k!='steps'} for row in g['results']]
    with (PUBLIC/'label-results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(dict(decision=s['decision'],active=s['active_identified'],random=s['random_identified'],active_labels=s['active_labels'],random_labels=s['random_labels'],audit=audit))


if __name__=='__main__':main()
