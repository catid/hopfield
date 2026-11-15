"""Prospective H2bs source-mapped final states and per-input LP transcripts."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_bp_first_geometry import trace_by_input,terminal_by_input
from audit_bp_first_fresh_states import numeric_error
from audit_connectivity_screen import path_valid


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Source:
    def __init__(self,root,bindings):
        self.root=Path(root);self.directory=self.root/bindings['directory'];self.protocol=bindings['protocol_sha256']
        # The first source-result access is guarded by the sealed full audit.
        assert sha(self.directory/'audit.json')==bindings['audit_sha256']
        audit=json.loads((self.directory/'audit.json').read_text());assert audit['status']=='passed' and audit['protocol_sha256']==self.protocol
        assert sha(self.directory/'sealed-replays.json')==bindings['seal_sha256']==audit['sealed_replays_sha256']
        self.rows={}
        for mode in ['LP96','BP_geometry_LP96','endpoint_dispatch']:
            file=self.directory/'runs'/f'block0-{mode}'/'result.json';assert sha(file)==bindings['cohort_receipts_sha256'][mode]
            receipt=json.loads(file.read_text());assert receipt['protocol_sha256']==self.protocol and receipt['model_inputs']==12288
            for row in receipt['rows']:self.rows[row['seed'],row['shard'],row['batch'],mode]=row

    @lru_cache(maxsize=12)
    def arrays(self,seed,shard,batch,mode):
        row=self.rows[seed,shard,batch,mode];path=self.directory/'runs'/f'block0-{mode}'/row['file'];assert sha(path)==row['file_sha256']
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

    @lru_cache(maxsize=24)
    def trace(self,seed,shard,batch,arm):
        row=self.rows[seed,shard,batch,'LP96'];name=row['trace_files'][arm];file=self.directory/'runs'/'block0-LP96'/name
        assert sha(file)==row['trace_sha256'][name]
        with np.load(file,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        payload=json.loads(arrays['trace_utf8'].tobytes());assert payload['protocol_sha256']==self.protocol and payload['offset']==batch*32
        return payload,arrays

    def groups(self,indices):
        for key in sorted(set(map(int,np.asarray(indices)//32))):
            positions=np.flatnonzero(np.asarray(indices)//32==key);yield key//8,key%8,positions,np.asarray(indices)[positions]%32

    def expected(self,seed,indices,mode,name,axis=0):
        result=None
        for shard,batch,positions,local in self.groups(indices):
            array=self.arrays(seed,shard,batch,mode)[name];part=np.take(array,local,axis=axis)
            if result is None:
                shape=list(array.shape);shape[axis]=len(indices);result=np.empty(shape,array.dtype)
            dest=[slice(None)]*result.ndim;dest[axis]=positions;result[tuple(dest)]=part
        return result

    def compare(self,seed,indices,mode,name,actual,axis=0,tolerance=None):
        expected=self.expected(seed,indices,mode,name,axis)
        if tolerance is None:assert np.array_equal(actual,expected,equal_nan=True),(name,seed,indices.tolist())
        else:numeric_error(actual,expected,tolerance)

    def check_traces(self,folder,row,actual,ph):
        ids=np.asarray(row['indices']);lpids=ids[actual['subset_indices']]
        if not len(lpids):assert not row['trace_files'];return 0,0
        arms=['connectivity_veto','original_lexical16','unified16','unified48','unified64','unified96'];candidates=states=0
        assert len(row['trace_files'])==6
        for ai,(arm,name) in enumerate(zip(arms,row['trace_files'])):
            file=folder/name;assert sha(file)==row['trace_sha256'][name]
            with np.load(file,allow_pickle=False) as z:a={k:z[k] for k in z.files}
            payload=json.loads(a['trace_utf8'].tobytes());assert payload['arm']==arm and payload['protocol_sha256']==ph and payload['offset']==0
            owners=lpids[a['valid_prefix_indices']];selected=set(map(int,lpids));aa=trace_by_input(payload,a,owners,selected)
            initial=terminal_by_input(payload['initial_terminal_checks'],lpids,selected);terminal=terminal_by_input(payload['final_terminal_checks'],owners,selected)
            expected_initial={};expected_terminal={};bb={};expected_valid=[];initial_array=np.empty(len(lpids),bool);expected_pre={k:{} for k in ['pre_terminal_stopped','pre_terminal_block_cap']}
            for shard,bi,positions,local in self.groups(lpids):
                bp,b=self.trace(row['seed'],shard,bi,ai);assert bp['arm']==arm
                base=shard*256+bi*32;source_all=base+np.arange(32);source_owners=base+b['valid_prefix_indices'];keep=set(map(int,lpids[positions]))
                initial_array[positions]=b['initial_terminal'][local]
                expected_initial.update(terminal_by_input(bp['initial_terminal_checks'],source_all,keep));expected_terminal.update(terminal_by_input(bp['final_terminal_checks'],source_owners,keep));bb.update(trace_by_input(bp,b,source_owners,keep))
                expected_valid.extend(int(i) for i in source_owners if int(i) in keep)
                for field in expected_pre:
                    if len(b[field]):expected_pre[field].update({int(i):bool(v) for i,v in zip(source_owners,b[field]) if int(i) in keep})
            assert np.array_equal(owners,np.array(sorted(expected_valid),np.int64)) and np.array_equal(a['initial_terminal'],initial_array)
            assert initial==expected_initial and terminal==expected_terminal and aa.keys()==bb.keys()
            for field,mapping in expected_pre.items():assert np.array_equal(a[field],np.array([mapping[int(i)] for i in owners],bool) if mapping else np.zeros(0,bool))
            for owner in aa:
                assert len(aa[owner])==len(bb[owner])
                for x,y in zip(aa[owner],bb[owner]):
                    assert x[:2]==y[:2] and x[3:]==y[3:] and len(x[2])==len(y[2]);states+=1
                    for c,d in zip(x[2],y[2]):assert c[0]==d[0] and c[2]==d[2];numeric_error(c[1],d[1],1e-8);candidates+=1
        return candidates,states

    def check_batch(self,folder,row,ph,tokens):
        folder=Path(folder);file=folder/row['file'];assert sha(file)==row['file_sha256']
        with np.load(file,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        ids=np.asarray(row['indices'],np.int64);seed=row['seed'];assert np.array_equal(a['global_indices'],ids) and len(ids)==len(tokens)==row['answer_timing']['n']
        distance=np.array([int(np.abs(np.argwhere(t==2)[0]-np.argwhere(t==3)[0]).sum()) for t in tokens],np.int64);high=distance>=182;bi=np.flatnonzero(high)
        assert np.array_equal(a['endpoint_manhattan'],distance) and np.array_equal(a['dispatch_high'],high) and np.array_equal(a['bp_indices'],bi)
        for name in ['couplings','hidden_steps']+[k for k in a if k.startswith('answer_')]:
            self.compare(seed,ids,'endpoint_dispatch',name,a[name],tolerance=1e-9 if name=='answer_own_margin' else None)
        if len(bi):
            for name in [k for k in a if k.startswith('bp_') and k!='bp_indices']:
                self.compare(seed,ids[bi],'BP_geometry_LP96',name,a[name],tolerance=1e-6 if name=='bp_energy64' else 1e-9 if name=='bp_own_margin' else None)
        else:assert not any(k in a for k in ['bp_final_logits','bp_energy64','bp_minimum_flip_margin32'])
        geometry=np.array([path_valid(t,b) for t,b in zip(tokens[bi],a['bp_prediction'])],bool);chosen=np.ones(len(ids),np.int8);chosen[bi[geometry]]=0;li=np.flatnonzero(chosen)
        assert np.array_equal(a['geometry_valid'],geometry) and np.array_equal(a['subset_indices'],li) and np.array_equal(a['answer_chosen_model'],chosen)
        assert row['answer_timing']['actual_BP_inputs']==len(bi) and row['answer_timing']['direct_LP_inputs']==int((~high).sum()) and row['answer_timing']['rescue_LP_inputs']==int((~geometry).sum())
        for name in [k for k in a if k.startswith('LP_')]:
            axis=0 if name in ['LP_binary_flip_count','LP_binary_cache_margin'] else 1
            self.compare(seed,ids[li],'LP96',name,a[name],axis=axis,tolerance=1e-6 if name=='LP_energy64' else 1e-9 if name in ['LP_binary_cache_margin','LP_minimum_flip_margin64'] else None)
        for name in [k for k in a if k.startswith('subset_') and k!='subset_indices']:
            if len(li):self.compare(seed,ids[li],'LP96',name.replace('subset_','answer_',1),a[name],tolerance=1e-9 if name=='subset_own_margin' else None)
            else:assert len(a[name])==0
        candidates,states=self.check_traces(folder,row,a,ph)
        return {'model_inputs':len(ids),'candidate_score_comparisons':candidates,'active_trace_states':states}
