"""Deterministic protocol recreation; original unseeded TRM data arrays were not released."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
UP=ROOT.parents[1]/'upstream'
SRC=Path(os.environ['TRM_SRC'])
OUT=Path(os.environ['TRM_DATA'])/'sudoku-extreme-1k-aug-1000'
sys.path[:0]=[str(SRC/'dataset'),str(SRC)]
import build_sudoku_dataset as B

if __name__=='__main__':
    if (OUT/'build_provenance.json').exists():raise SystemExit('Finished dataset exists; no rebuild.')
    source=UP/'data/sudoku-extreme'
    def local_file(repo,filename,repo_type):
        assert repo=='sapientinc/sudoku-extreme' and filename in ['train.csv','test.csv'] and repo_type=='dataset'
        return str(source/filename)
    B.hf_hub_download=local_file
    # The released builder never seeds its global RNG. We explicitly pin a new run.
    np.random.seed(42)
    cfg=B.DataProcessConfig(output_dir=str(OUT),subsample_size=1000,num_aug=1000)
    begin=time.monotonic()
    B.convert_subset('train',cfg);B.convert_subset('test',cfg)
    rows=np.random.RandomState(42).choice(3831994,size=1000,replace=False)
    rec=dict(protocol='New deterministic recreation of original TRM data protocol; original unseeded row/augmentation arrays unavailable',
        seed=42,selected_train_rows=rows.tolist(),source_revision='58942f96baeb572ca3127e2a9e9c70f330783d6b',
        builder_sha256=hashlib.sha256(Path(B.__file__).read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_s=time.monotonic()-begin,
        arrays={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*.npy'))})
    (OUT/'build_provenance.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps({k:v for k,v in rec.items() if k not in ['selected_train_rows','arrays']}),flush=True)
