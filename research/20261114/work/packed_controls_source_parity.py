"""H2bt source-mapped controls/candidate, including LP-only full trajectories."""
from pathlib import Path
import numpy as np
from packed_dispatch_source_parity import Source as Previous,sha
from audit_connectivity_screen import path_valid


class Source(Previous):
    def compare_global(self,seed,indices,mode,arrays):
        indices=np.asarray(indices);first=int(indices[0]);reference=self.arrays(seed,first//256,first%256//32,mode)
        expected={name[7:] for name in reference if name.startswith('answer_')}
        assert set(arrays)==expected|{'global_indices','model_seed'}
        assert np.array_equal(arrays['global_indices'],indices) and np.array_equal(arrays['model_seed'],np.full(len(indices),seed))
        # Every field has exactly the original equality/tolerance contract.
        # Group source reads once per original32-input batch to avoid reloading
        # full source couplings once for every tiny final scalar field.
        for _,_,positions,_ in self.groups(indices):
            ids=indices[positions]
            for name in expected:self.compare(seed,ids,mode,'answer_'+name,arrays[name][positions],tolerance=1e-9 if name=='own_margin' else None)

    def check_batch(self,folder,row,ph,tokens):
        folder=Path(folder);file=folder/row['file'];assert sha(file)==row['file_sha256']
        with np.load(file,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        ids=np.asarray(row['indices'],np.int64);seed=row['seed'];mode=row['answer_timing']['mode'];n=len(ids)
        assert mode in ['LP96','BP_geometry_LP96','endpoint_dispatch'] and np.array_equal(a['global_indices'],ids) and n==len(tokens)==row['answer_timing']['n']
        reference=self.arrays(seed,int(ids[0])//256,int(ids[0])%256//32,mode)
        assert {k for k in a if k.startswith('answer_')}=={k for k in reference if k.startswith('answer_')}
        for name in ['couplings','hidden_steps']+[k for k in a if k.startswith('answer_')]:self.compare(seed,ids,mode,name,a[name],tolerance=1e-9 if name=='answer_own_margin' else None)
        if mode=='endpoint_dispatch':
            distance=np.array([int(np.abs(np.argwhere(t==2)[0]-np.argwhere(t==3)[0]).sum()) for t in tokens],np.int64);high=distance>=182;bi=np.flatnonzero(high)
            assert np.array_equal(a['endpoint_manhattan'],distance) and np.array_equal(a['dispatch_high'],high) and np.array_equal(a['bp_indices'],bi)
        else:
            bi=np.arange(n) if mode=='BP_geometry_LP96' else np.zeros(0,np.int64)
            assert not any(k in a for k in ['endpoint_manhattan','dispatch_high','bp_indices'])
        if len(bi):
            assert {'bp_final_logits','bp_minimum_flip_margin32','bp_energy64'}<=a.keys()
            for name in [k for k in a if k.startswith('bp_') and k!='bp_indices']:
                self.compare(seed,ids[bi],'BP_geometry_LP96',name,a[name],tolerance=1e-6 if name=='bp_energy64' else 1e-9 if name=='bp_own_margin' else None)
        else:
            assert not any(k in a for k in ['bp_final_logits','bp_energy64','bp_minimum_flip_margin32'])
            if mode=='LP96':assert not any(k.startswith('bp_') for k in a)
        geometry=np.array([path_valid(t,b) for t,b in zip(tokens[bi],a['bp_prediction'])],bool) if len(bi) else np.zeros(0,bool)
        chosen=np.ones(n,np.int8);chosen[bi[geometry]]=0;li=np.flatnonzero(chosen)
        assert np.array_equal(a['geometry_valid'],geometry) and np.array_equal(a['answer_chosen_model'],chosen)
        assert np.array_equal(a['subset_indices'],np.zeros(0,np.int64) if mode=='LP96' else li)
        assert row['answer_timing']['actual_subset_inputs']==(0 if mode=='LP96' else len(li))
        if mode=='endpoint_dispatch':assert row['answer_timing']['actual_BP_inputs']==len(bi) and row['answer_timing']['direct_LP_inputs']==int((~high).sum()) and row['answer_timing']['rescue_LP_inputs']==int((~geometry).sum())
        lp_keys={k for k in a if k.startswith('LP_')}
        assert lp_keys==({k for k in self.arrays(seed,int(ids[0])//256,int(ids[0])%256//32,'LP96') if k.startswith('LP_')} if len(li) else set())
        for name in lp_keys:
            axis=0 if name in ['LP_binary_flip_count','LP_binary_cache_margin'] else 1
            self.compare(seed,ids[li],'LP96',name,a[name],axis=axis,tolerance=1e-6 if name=='LP_energy64' else 1e-9 if name in ['LP_binary_cache_margin','LP_minimum_flip_margin64'] else None)
        for name in [k for k in a if k.startswith('subset_') and k!='subset_indices']:
            if len(li):self.compare(seed,ids[li],'LP96',name.replace('subset_','answer_',1),a[name],tolerance=1e-9 if name=='subset_own_margin' else None)
            else:assert len(a[name])==0
        if mode=='LP96':assert not any(k.startswith('subset_') and k!='subset_indices' for k in a)
        candidates,states=self.check_traces(folder,row,{**a,'subset_indices':li},ph)
        return {'model_inputs':n,'candidate_score_comparisons':candidates,'active_trace_states':states}
