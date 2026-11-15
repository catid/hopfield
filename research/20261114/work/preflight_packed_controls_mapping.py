"""H2bt synthetic LP-only/full-BP/low-dispatch archive and transcript branches."""
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import study as S
from packed_controls_source_parity import Source,sha
from preflight_packed_dispatch_mapping import transcript


def main():
    started=time.perf_counter();n=32;ids=np.arange(32,64);tokens=np.ones((n,180,180),np.uint8);tokens[:,0,0]=2;tokens[:,0,2]=3;board=np.zeros_like(tokens,bool);board[:,0,:3]=True
    common={'couplings':np.ones((n,1),np.float32),'hidden_steps':np.full(n,2),'answer_prediction':board,'answer_own_margin':np.full(n,.1),'answer_chosen_model':np.ones(n,np.int8)}
    refs={'LP96':{**common,'LP_prediction':np.stack([board]*7)},'BP_geometry_LP96':{**common,'answer_chosen_model':np.zeros(n,np.int8),'bp_prediction':board,'bp_own_margin':np.full(n,.1),'bp_final_logits':np.ones((n,2),np.float32),'bp_minimum_flip_margin32':np.full(n,.1,np.float32),'bp_energy64':np.full(n,4.)},'endpoint_dispatch':{**common,'LP_prediction':np.stack([board]*7)}}
    source=Source.__new__(Source);source.arrays=lambda seed,shard,bi,mode:refs[mode]
    arms=['connectivity_veto','original_lexical16','unified16','unified48','unified64','unified96'];source.trace=lambda seed,shard,bi,ai:transcript(ids,arms[ai],'source',32)
    dest=S.OUT/'bp_first_packed_controls';dest.mkdir(parents=True,exist_ok=True);cases=rejected=0
    with tempfile.TemporaryDirectory(prefix='mapping-synthetic-',dir=dest) as tmp:
        folder=Path(tmp)
        for mode in ['LP96','BP_geometry_LP96','endpoint_dispatch']:
            actual={k:v.copy() for k,v in refs[mode].items()};actual['global_indices']=ids.copy();actual['geometry_valid']=np.ones(n,bool) if mode=='BP_geometry_LP96' else np.zeros(0,bool);actual['subset_indices']=np.zeros(0,np.int64) if mode!='endpoint_dispatch' else np.arange(n)
            if mode=='endpoint_dispatch':
                actual.update(endpoint_manhattan=np.full(n,2),dispatch_high=np.zeros(n,bool),bp_indices=np.zeros(0,np.int64),bp_prediction=board[:0],subset_prediction=board.copy(),subset_own_margin=np.full(n,.1))
            row={'seed':0,'indices':ids.tolist(),'file':mode+'.npz','trace_files':[],'trace_sha256':{},'answer_timing':{'mode':mode,'n':n,'actual_subset_inputs':n if mode=='endpoint_dispatch' else 0}}
            if mode=='endpoint_dispatch':row['answer_timing'].update(actual_BP_inputs=0,direct_LP_inputs=n,rescue_LP_inputs=0)
            if mode!='BP_geometry_LP96':
                for arm in arms:
                    name=mode+'-'+arm+'.npz';_,data=transcript(ids,arm,'actual',0);np.savez(folder/name,**data);row['trace_files'].append(name);row['trace_sha256'][name]=sha(folder/name)
            np.savez(folder/row['file'],**actual);row['file_sha256']=sha(folder/row['file']);result=source.check_batch(folder,row,'actual',tokens);assert result['model_inputs']==n and result['active_trace_states']==(0 if mode=='BP_geometry_LP96' else 192);cases+=1
            if mode=='LP96':broken={k:v for k,v in actual.items() if k!='LP_prediction'}
            elif mode=='BP_geometry_LP96':broken={k:v for k,v in actual.items() if k!='bp_final_logits'}
            else:broken={**actual,'dispatch_high':np.ones(n,bool)}
            np.savez(folder/row['file'],**broken);row['file_sha256']=sha(folder/row['file'])
            try:source.check_batch(folder,row,'actual',tokens)
            except AssertionError:rejected+=1
            else:raise AssertionError('Missing field or corrupted route accepted')
    names=['preflight_packed_controls_mapping.py','packed_controls_source_parity.py','packed_dispatch_source_parity.py','preflight_packed_dispatch_mapping.py'];r={'status':'passed','actual_mode_archive_branches':cases,'corrupted_archives_rejected':rejected,'LP_only_full_trajectory_states_checked':192,'low_dispatch_full_LP_trajectory_states_checked':192,'source_outcomes_read':False,'gpu_executions':0,'seconds':time.perf_counter()-started,'source_sha256':{str((S.HERE/name).relative_to(S.ROOT)):S.sha(S.HERE/name) for name in names}}
    S.save(dest/'mapping-synthetic-gate.json',r);print(json.dumps(r,indent=2))


if __name__=='__main__':main()
