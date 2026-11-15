"""Public H1bi package hashes, all proofs and archived-primal comparison."""
import json
import numpy as np
from partial_degree_study import ROOT,PUBLIC,PARENT,sha,npz
from audit_partial_degree import public_audit


def main():
    manifest=ROOT/'results/research/ripper/partial_degree_public_manifest.json';m=json.loads(manifest.read_text())
    for path,digest in m['sha256'].items():assert sha(ROOT/path)==digest,path
    assert sha(PUBLIC/'summary.json')==m['raw_audit_sha256']
    public_audit();result=json.loads((PUBLIC/'summary.json').read_text());old=npz(PARENT);same=unique=0
    for row in result['rows']:
        arrays=npz(PUBLIC/f'j{row["index"]:03d}'/'models.npz');same+=int(np.array_equal(arrays['d20_primal_twice'],old[f's{row["seed"]}_{row["arm"]}_primal_twice']))
        v=row['curvature_twice'];unique+=int(v[19]>max(v[:19]+v[20:]))
    assert same==unique==64
    runtime=json.loads((PUBLIC/'runtime.json').read_text());assert runtime==result['runtime'] and runtime['status']=='validated' and runtime['wall_s']<180 and runtime['aggregate_cpu_s']<900
    assert len(runtime['commands'])==65 and all(c['returncode']==0 for c in runtime['commands'])
    print(json.dumps(dict(status='passed',manifest_paths=len(m['sha256'])+1,selected_primals_identical_to_archived=same,unique_degree20_curvature_maxima=unique,resource_budget_passed=True)))


if __name__=='__main__':main()
