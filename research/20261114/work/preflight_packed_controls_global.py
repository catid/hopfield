"""Bounded synthetic I/O check of source-grouped exact global-state comparison."""
from functools import lru_cache
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import study as S
import fast_npz as Z
from packed_controls_source_parity import Source


def main():
    started=time.perf_counter();dest=S.OUT/'bp_first_packed_controls';dest.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(2026111540);n=32;count=17;ids=np.arange(n*count)
    data={'couplings':rng.normal(size=(n,4,180,180)).astype(np.float32),'answer_prediction':rng.integers(2,size=(n,180,180),dtype=np.uint8).astype(bool),'answer_own_margin':rng.random(n)}
    for name in ['finite','stopped','native_cap','block_cap','accepted','hidden_good','clamps_ok','chosen_model','iterations']:data['answer_'+name]=np.arange(n)%2
    global_arrays={name[7:]:np.concatenate([value]*count) for name,value in data.items() if name.startswith('answer_')};global_arrays.update(global_indices=ids,model_seed=np.zeros(len(ids),np.int64));fields=set(global_arrays)-{'global_indices','model_seed'}
    with tempfile.TemporaryDirectory(prefix='global-synthetic-',dir=dest) as tmp:
        file=Path(tmp)/'source.npz';Z.save(file,**data);source=Source.__new__(Source);loads=[0]
        @lru_cache(maxsize=12)
        def arrays(seed,shard,batch,mode):
            loads[0]+=1
            with np.load(file,allow_pickle=False) as z:return {key:z[key] for key in z.files}
        source.arrays=arrays
        before=time.perf_counter()
        for name in fields:source.compare(0,ids,'LP96','answer_'+name,global_arrays[name],tolerance=1e-9 if name=='own_margin' else None)
        original_seconds=time.perf_counter()-before;original_loads=loads[0];arrays.cache_clear();loads[0]=0
        before=time.perf_counter();source.compare_global(0,ids,'LP96',global_arrays);grouped_seconds=time.perf_counter()-before;grouped_loads=loads[0]
        assert original_loads==count*len(fields) and grouped_loads==count
        rejected=0
        for field in ['prediction','own_margin','global_indices']:
            bad={**global_arrays,field:global_arrays[field].copy()}
            if field=='prediction':bad[field][33,0,0]=not bad[field][33,0,0]
            elif field=='own_margin':bad[field][33]+=.1
            else:bad[field][33]=12
            try:source.compare_global(0,ids,'LP96',bad)
            except AssertionError:rejected+=1
            else:raise AssertionError('Corrupted global output accepted')
        arrays.cache_clear()
    assert time.perf_counter()-started<60
    names=['preflight_packed_controls_global.py','packed_controls_source_parity.py','audit_packed_controls_cohort.py'];r={'status':'passed','synthetic_original_global_fields':len(fields),'source_batches':count,'original_source_archive_loads':original_loads,'grouped_source_archive_loads':grouped_loads,'original_seconds':original_seconds,'grouped_seconds':grouped_seconds,'rejected_corruptions':rejected,'source_outcomes_read':False,'gpu_executions':0,'seconds':time.perf_counter()-started,'source_sha256':{str((S.HERE/name).relative_to(S.ROOT)):S.sha(S.HERE/name) for name in names},'scope':'Bounded local synthetic source-I/O measurement, not GPU or scientific inference timing. Equality, nonfinite masks and1e-9 margin tolerance unchanged. H2bs frozen source remains untouched.'}
    S.save(dest/'global-audit-synthetic-gate.json',r);print(json.dumps(r,indent=2))


if __name__=='__main__':main()
