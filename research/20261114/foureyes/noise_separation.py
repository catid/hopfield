"""Post-fit diagnostic: whether any equality-count threshold can recover the graph."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/research/20261114/foureyes/noise_boundary'
PUBLIC=ROOT/'results/research/foureyes/noise_boundary'


def main():
    manifest=json.loads((BASE/'prepared.json').read_text())
    ii,jj=np.triu_indices(81,1);row,col=np.arange(81)//9,np.arange(81)%9
    truth=(row[ii]==row[jj])|(col[ii]==col[jj])|((row[ii]//3==row[jj]//3)&(col[ii]//3==col[jj]//3))
    rows=[];evidence={}
    for model in manifest['models']:
        if model['corrupted_cells_per_board']!=9:continue
        path=ROOT/model['model_path']
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==model['model_sha256'];evidence[str(path.relative_to(ROOT))]=digest
        counts=np.load(path)['counts']
        lower=int(counts[truth].max());upper=int(counts[~truth].min())-1
        rows.append(dict(seed=model['seed'],n=model['n'],chosen_threshold=model['threshold'],
            smallest_exact_threshold=lower,largest_exact_threshold=upper,
            some_exact_count_threshold_exists=lower<=upper,
            selected_threshold_exact=lower<=model['threshold']<=upper))
    with (PUBLIC/'separation.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (PUBLIC/'separation-provenance.json').write_text(json.dumps(dict(
        role='Post-fit oracle diagnostic; true graph never enters estimator or threshold selection.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=evidence),indent=2)+'\n')


if __name__=='__main__':main()
