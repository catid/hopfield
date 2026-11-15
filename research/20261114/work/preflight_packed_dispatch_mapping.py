"""H2bs synthetic source identity and normalized transcript mapping checks."""
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import study as S
from packed_dispatch_source_parity import Source,sha


def transcript(ids,arm,protocol,offset):
    n=len(ids);state=np.array([[[bool(i%2),False],[False,bool(i%3)]] for i in ids])
    payload={'protocol_sha256':protocol,'arm':arm,'offset':offset,'initial_terminal_checks':[{'local_input':j,'fixed':int(i)} for j,i in enumerate(ids)],'final_terminal_checks':[{'local_input':j,'terminal':int(i)} for j,i in enumerate(ids)],
      'trace':[{'round':1,'chosen':list(range(0,n,2)),'geometry':[{'local_input':j,'owner_tag':int(i)} for j,i in enumerate(ids)],'candidates':[{'local_input':j,'cells':[int(i),int(i)+1],'delta':float(i)/10} for j,i in enumerate(ids)],'native_results':[{'local_input':j,'stopped':True,'owner_tag':int(i)} for j,i in enumerate(ids)]}]}
    # Selection must be invariant to source batch position.
    payload['trace'][0]['chosen']=[j for j,i in enumerate(ids) if int(i)%2==0]
    arrays={'trace_utf8':np.frombuffer(json.dumps(payload).encode(),np.uint8),'valid_prefix_indices':np.arange(n),'initial_terminal':np.zeros(n,bool),'pre_terminal_stopped':np.ones(n,bool),'pre_terminal_block_cap':np.zeros(n,bool),'states_before':state[None]}
    return payload,arrays


def main():
    started=time.perf_counter();source=Source.__new__(Source);checked=0
    def arrays(seed,shard,batch,mode):
        ids=shard*256+batch*32+np.arange(32)
        return {'axis0':ids[:,None]+np.array([0,10000]),'axis1':ids[None,:]+np.array([0,10000,20000])[:,None]}
    source.arrays=arrays
    for ids in [np.array([1,3,31,32,34]),np.array([255,256,511,512]),np.array([4095,0,32,1]),np.arange(31,64)]:
        for axis,name in [(0,'axis0'),(1,'axis1')]:
            expected=ids[:,None]+np.array([0,10000]) if axis==0 else ids[None,:]+np.array([0,10000,20000])[:,None]
            assert np.array_equal(source.expected(2,ids,'LP96',name,axis),expected);source.compare(2,ids,'LP96',name,expected,axis);checked+=1
    arms=['connectivity_veto','original_lexical16','unified16','unified48','unified64','unified96']
    def trace(seed,shard,batch,ai):return transcript(shard*256+batch*32+np.arange(32),arms[ai],'source',batch*32)
    source.trace=trace;dest=S.OUT/'bp_first_packed_dispatch';dest.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mapping-synthetic-',dir=dest) as tmp:
        folder=Path(tmp);ids=np.array([1,3,31,32,34]);row={'seed':0,'indices':ids.tolist(),'trace_files':[],'trace_sha256':{}}
        for arm in arms:
            name=arm+'.npz';_,data=transcript(ids,arm,'actual',0);np.savez(folder/name,**data);row['trace_files'].append(name);row['trace_sha256'][name]=sha(folder/name)
        counts=source.check_traces(folder,row,{'subset_indices':np.arange(len(ids))},'actual');assert counts==(30,30);checked+=1
        first=folder/row['trace_files'][0]
        with np.load(first,allow_pickle=False) as z:mutated={k:z[k] for k in z.files}
        payload=json.loads(mutated['trace_utf8'].tobytes());payload['trace'][0]['candidates'][0]['delta']+=1.;mutated['trace_utf8']=np.frombuffer(json.dumps(payload).encode(),np.uint8);np.savez(first,**mutated);row['trace_sha256'][first.name]=sha(first)
        try:source.check_traces(folder,row,{'subset_indices':np.arange(len(ids))},'actual')
        except AssertionError:checked+=1
        else:raise AssertionError('Mutated source candidate score accepted')
        assert source.check_traces(folder,{'indices':[0],'trace_files':[]},{'subset_indices':np.zeros(0,np.int64)},'actual')==(0,0);checked+=1
    names=['packed_dispatch_source_parity.py','preflight_packed_dispatch_mapping.py','audit_bp_first_geometry.py','audit_bp_first_fresh_states.py']
    result={'status':'passed','synthetic_mapping_cases':checked,'normalized_candidate_comparisons':30,'normalized_active_states':30,'mutated_candidate_rejected':True,'source_outcomes_read':False,'gpu_executions':0,'seconds':time.perf_counter()-started,'source_sha256':{str((S.HERE/name).relative_to(S.ROOT)):S.sha(S.HERE/name) for name in names}}
    S.save(dest/'source-mapping-synthetic-gate.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
