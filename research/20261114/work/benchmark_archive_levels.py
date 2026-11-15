"""Bounded CPU-only output-archive benchmark; no inference or target selection."""
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np


def encode(arrays,level):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=level,allowZip64=True) as archive:
        for key,value in arrays.items():
            with archive.open(key+'.npy','w',force_zip64=True) as target:np.lib.format.write_array(target,np.asanyarray(value),allow_pickle=False)
    return stream.getvalue()


def main():
    source=Path('artifacts/research/20261114/work/hard_negative_transfer/runs/shard00-s0/outputs.npz');target=Path('artifacts/research/20261114/work/archive_benchmark');target.mkdir(exist_ok=True)
    with np.load(source,allow_pickle=False) as z:arrays={key:z[key] for key in z.files}
    rows=[]
    for repeat,levels in enumerate([[1,6],[6,1]]):
        for level in levels:
            start=time.perf_counter();data=encode(arrays,level);seconds=time.perf_counter()-start
            with np.load(io.BytesIO(data),allow_pickle=False) as z:assert z.files==list(arrays) and all(z[key].dtype==value.dtype and np.array_equal(z[key],value) for key,value in arrays.items())
            rows.append({'repeat':repeat,'level':level,'seconds':seconds,'bytes':len(data),'exact_array_parity':True})
    medians={level:float(np.median([r['seconds'] for r in rows if r['level']==level])) for level in [1,6]}
    result={'status':'passed','source_archive':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows,'median_seconds_by_level':medians,'level6_over_level1_median':medians[6]/medians[1],'new_trajectories':0,'scope':'Retained H2ac outputs, CPU memory encode only; exact dtype/value/array-order equality, all arrays checked after decoding. Two alternating repeats, no inference or scientific selection. Disk transfer and archive decode excluded.'}
    (target/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
