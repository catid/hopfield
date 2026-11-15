"""Losslessly compress every full score table and publish the fixed-grid audit."""
import csv
import gzip
import shutil
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from verify_public import verify,diagnostics
PUBLIC=ROOT/'results/research/loki/h3ax'


def main():
    frozen();g=read(OUT/'capacity-gate.json');files=['capacity-gate.json','study-result.json','candidate-normalizers.json','campaign.log']+[f'f{i}/{name}' for i in range(45) for name in ('source-population.npy','true-N13-F.npy')]+[f"f{r['family']}/b{r['bank']}/draw.npz" for r in g['rows']]
    for rel in files:
        path=PUBLIC/rel;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/rel,path)
    for row in g['rows']:
        rel=f"f{row['family']}/b{row['bank']}/scores.json";compressed=rel+'.gz';(PUBLIC/compressed).write_bytes(gzip.compress((OUT/rel).read_bytes(),mtime=0));files.append(compressed)
    save(PUBLIC/'failure-diagnostics.json',diagnostics(ROOT,PUBLIC));files.append('failure-diagnostics.json');audit=verify(ROOT,False);s=dict(status='validated',utc=now(),hypothesis='H3ax',**{k:g[k] for k in ('decision','rows','family_results','eligible_families','ineligible_families','new_positive_draws')},aggregate_CPU_seconds=g['aggregate_cpu_seconds'],wall_seconds=g['wall_seconds'],GPU_seconds=0,audit=audit,score_export='Deterministic gzip of exact original JSON bytes. Decompressed SHA256 must equal raw study row receipt; every1541candidate score retained perbank.',protocol_sha256=sha(SRC/'protocol.json'),audit_source_sha256=sha(__file__),verifier_source_sha256=sha(SRC/'verify_public.py'),public_sha256={rel:sha(PUBLIC/rel) for rel in files});save(PUBLIC/'summary.json',s)
    with(PUBLIC/'family-results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(s['family_results'][0]));writer.writeheader();writer.writerows(s['family_results'])
    print(dict(decision=s['decision'],eligible=s['eligible_families'],ineligible=s['ineligible_families'],banks=s['new_positive_draws'],audit=audit))


if __name__=='__main__':main()
