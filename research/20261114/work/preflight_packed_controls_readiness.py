"""H2bt test the actual unchanged readiness loader with synthetic runtime only."""
from contextlib import nullcontext
import json
from pathlib import Path
import time
import types
import study as S
import endpoint_dispatch_pipeline as P


def main():
    started=time.perf_counter();calls=[];models=[];old_imports=S.imports;old_model=S.model_for
    class Model:
        def __init__(self,seed):self.seed=seed;self.state=None;self.is_double=False
        def load_state_dict(self,state):self.state=state
        def double(self):self.is_double=True;return self
    def model(seed,*args):models.append(seed);return Model(seed)
    def load(path,**kwargs):calls.append(str(path));assert kwargs=={'map_location':'cuda','weights_only':True};return {'path':str(path)}
    runtime=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:None),load=load)
    p={'models':{str(seed):{arm:{'path':f'synthetic-seed{seed}-{arm}.pt','sha256':f'hash-{seed}-{arm}'} for arm in ['pl','old_binary']} for seed in range(3)}}
    try:
        S.imports=lambda:(runtime,None,None);S.model_for=model
        for mode in ['LP96','BP_geometry_LP96','endpoint_dispatch']:
            calls.clear();models.clear();native,doubles,receipts=P.load_models(p,mode);required=['lp'] if mode=='LP96' else ['bp','lp']
            assert len(calls)==len(models)==len(receipts)==3*len(required)
            assert all(set(native[seed])==set(doubles[seed])==set(required) for seed in range(3))
            assert {(r['seed'],r['model']) for r in receipts}=={(seed,label) for seed in range(3) for label in required}
            assert all(r['seconds']>=0 and r['checkpoint_sha256']==p['models'][str(r['seed'])]['old_binary' if r['model']=='lp' else 'pl']['sha256'] for r in receipts)
            for seed in range(3):
                for label in required:assert native[seed][label] is not doubles[seed][label] and not native[seed][label].is_double and doubles[seed][label].is_double and native[seed][label].state==doubles[seed][label].state
            if mode=='LP96':assert all('-old_binary.pt' in path for path in calls) and not any('-pl.pt' in path for path in calls)
    finally:S.imports=old_imports;S.model_for=old_model
    names=['preflight_packed_controls_readiness.py','endpoint_dispatch_pipeline.py','packed_controls_pipeline.py'];result={'status':'passed','actual_loader_branches_checked':3,'LP96_checkpoint_loads':3,'dual_model_checkpoint_loads_per_mode':6,'unused_BP_for_LP96':False,'float64_copies_distinct':True,'source_outcomes_read':False,'gpu_executions':0,'seconds':time.perf_counter()-started,'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in names}}
    S.save(S.OUT/'bp_first_packed_controls/readiness-synthetic-gate.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
