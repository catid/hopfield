"""Export complete exact distributions, including tiny nonzero failure fractions."""
import csv
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify,exact_presentation
PUBLIC=ROOT/'results/research/loki/h3as'


def main():
    p=frozen();files=['capacity-gate.json','study-result.json','campaign.log']+[f'family{i}/{name}' for i in range(2) for name in ('population.npy','witness-masks.npy','frequency.npy','selection-patterns.json','exact-distribution.json')]
    for rel in files:
        path=PUBLIC/rel;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,path)
    audit=verify(ROOT,False);g=read(PUBLIC/'capacity-gate.json')
    s=dict(status='validated',utc=now(),hypothesis='H3as',**{k:g[k] for k in ('decision','rows','primary','graph_distinctness')},exact_presentation=exact_presentation(g['rows']),aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,lemma=p['lemma'],scope=p['evaluation'],audit=audit,protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={rel:sha(PUBLIC/rel) for rel in files})
    save(PUBLIC/'summary.json',s);rows=[dict(row,**{k:v for k,v in detail.items() if k not in ('family','dose')}) for row,detail in zip(s['rows'],s['exact_presentation'])]
    with(PUBLIC/'reliability.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(dict(decision=s['decision'],primary=s['primary'],audit=audit))


if __name__=='__main__':main()
