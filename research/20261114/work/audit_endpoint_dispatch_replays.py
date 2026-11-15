"""Independent H2br token rule, actual subset identity, repeats and outcomes."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from audit_connectivity_screen import path_valid
from verify_hard_negative_feasibility import valid_shortest
from audit_endpoint_dispatch_states import numeric_error
from verify_endpoint_dispatch_trace import check_trace_pair
MODES=['LP96','BP_geometry_LP96','endpoint_dispatch']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    array=np.asarray(value);h=hashlib.sha256(array.dtype.str.encode());h.update(json.dumps(list(array.shape)).encode());h.update(np.ascontiguousarray(array).tobytes());return h.hexdigest()


def load(directory,row):
    file=directory/row['file'];assert sha(file)==row['file_sha256']
    with np.load(file,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def check_batch(folder,row,actual,lp,bp,control_folder,control_row,tokens,mode,actual_ph,source_ph):
    n=len(tokens);assert n==32 and mode in MODES
    geometry=np.array([path_valid(t,b) for t,b in zip(tokens,bp['bp_prediction'])],bool)
    distance=np.array([int(np.abs(np.argwhere(t==2)[0]-np.argwhere(t==3)[0]).sum()) for t in tokens],np.int64)
    high=distance>=182
    bi=np.arange(n) if mode=='BP_geometry_LP96' else np.flatnonzero(high) if mode=='endpoint_dispatch' else np.array([],np.int64)
    chosen=np.ones(n,np.int8)
    if mode!='LP96':chosen[bi[geometry[bi]]]=0
    ix=np.flatnonzero(chosen)
    assert np.array_equal(lp['couplings'],bp['couplings'],equal_nan=True)
    if 'couplings' in actual:assert np.array_equal(actual['couplings'],lp['couplings'],equal_nan=True)
    assert np.array_equal(actual['hidden_steps'],lp['hidden_steps']) and np.array_equal(actual['answer_hidden_good'],lp['answer_hidden_good'])
    assert np.array_equal(actual['answer_hidden_good'],bp['answer_hidden_good'])
    if mode!='LP96':
        assert np.array_equal(actual['geometry_valid'],geometry[bi]) and np.array_equal(actual['subset_indices'],ix)
        for name in ['prediction','iterations','stopped','finite','native_cap','block_cap']:assert np.array_equal(actual['bp_'+name],bp['bp_'+name][bi]),(mode,name)
        numeric_error(actual['bp_own_margin'],bp['bp_own_margin'][bi],1e-9)
        if len(bi):
            for name in ['bp_final_logits','bp_minimum_flip_margin32','bp_energy64']:numeric_error(actual[name],bp[name][bi],1e-6 if name=='bp_energy64' else 1e-9)
        else:assert not any(name in actual for name in ['bp_final_logits','bp_minimum_flip_margin32','bp_energy64'])
    if mode=='endpoint_dispatch':
        assert np.array_equal(actual['endpoint_manhattan'],distance) and np.array_equal(actual['dispatch_high'],high) and np.array_equal(actual['bp_indices'],bi)
        assert row['timing']['actual_BP_inputs']==len(bi) and row['timing']['direct_LP_inputs']==int((~high).sum()) and row['timing']['rescue_LP_inputs']==int((high&~geometry).sum())
    assert row['timing']['actual_subset_inputs']==(len(ix) if mode!='LP96' else 0)
    assert np.array_equal(actual['answer_chosen_model'],chosen)
    if len(ix):
        for name in ['prediction','iterations','stopped','finite','block_moves','post_block_sweeps','native_cap','block_cap','initial_simple_path']:assert np.array_equal(actual['LP_'+name],lp['LP_'+name][:,ix]),name
        assert np.array_equal(actual['LP_binary_flip_count'],lp['LP_binary_flip_count'][ix])
        numeric_error(actual['LP_binary_cache_margin'],lp['LP_binary_cache_margin'][ix],1e-9)
        numeric_error(actual['LP_minimum_flip_margin64'],lp['LP_minimum_flip_margin64'][:,ix],1e-9)
        numeric_error(actual['LP_energy64'],lp['LP_energy64'][:,ix],1e-6)
    else:assert not any(name.startswith('LP_') for name in actual)
    counts=check_trace_pair(folder,row,control_folder,control_row,ix,n,actual_ph,source_ph)
    for name in ['prediction','iterations','stopped','finite','own_margin','native_cap','block_cap']:
        expected=bp['bp_'+name].copy();expected[ix]=lp['answer_'+name][ix]
        if name=='own_margin':numeric_error(actual['answer_'+name],expected,1e-9)
        else:assert np.array_equal(actual['answer_'+name],expected),(mode,name)
        if mode!='LP96':
            subset=actual['subset_'+name]
            if name=='own_margin':numeric_error(subset,lp['answer_'+name][ix],1e-9)
            else:assert np.array_equal(subset,lp['answer_'+name][ix])
    clamps=((tokens==1)|(actual['answer_prediction']==(tokens>=2))).all((1,2))
    accepted=(lp['answer_hidden_good']&actual['answer_finite']&actual['answer_stopped']&~actual['answer_native_cap']&~actual['answer_block_cap']&clamps&np.isfinite(actual['answer_own_margin'])&(actual['answer_own_margin']>=-1e-6))
    assert np.array_equal(actual['answer_accepted'],accepted) and np.array_equal(actual['answer_clamps_ok'],clamps)
    return chosen,accepted,geometry,high,counts


def one_shard(args):
    root,directory,p,datasets,shard,cohorts=args;started=time.perf_counter();ds=datasets[shard];assert sha(root/ds['path'])==ds['sha256']
    with np.load(root/ds['path'],allow_pickle=False) as z:tokens,labels,indices,components=z['tokens'],z['labels'],z['indices'],z['mixture_component']
    assert np.array_equal(indices,np.arange(shard*256,(shard+1)*256)) and all(valid_shortest(t,b) for t,b in zip(tokens,labels))
    shortest=labels.sum((1,2));records={(r['pair'],r['mode']):r for r in cohorts if r['kind']=='timing'}
    ph=sha(directory/'protocol.json');outputs=[];cache={};candidate_scores=trace_states=repeated=0;maximum=0.
    for seed in range(3):
        job=f'shard{shard:02d}-s{seed}'
        for bi in range(8):
            tt=tokens[bi*32:(bi+1)*32];source={};reference_rows={}
            for mode in MODES:
                receipt=records[0,mode];row=next(r for r in receipt['rows'] if r['job_id']==job and r['batch']==bi);source[mode]=load(root/receipt['directory'],row);reference_rows[mode]=row
            lp,bp=source['LP96'],source['BP_geometry_LP96'];graphs=[]
            for prefix in ['bp','lp']:
                boards=bp['bp_prediction'] if prefix=='bp' else lp['answer_prediction'];values=[]
                for i,board in enumerate(boards):
                    key=(bi*32+i,np.packbits(board).tobytes())
                    if key not in cache:cache[key]=path_valid(tt[i],board) and int(board.sum())==int(shortest[bi*32+i])
                    values.append(cache[key])
                graphs.append(np.array(values,bool))
            bp_graph,lp_graph=graphs;bp_clamps=((tt==1)|(bp['bp_prediction']==(tt>=2))).all((1,2))
            bp_accept=lp['answer_hidden_good']&bp['bp_finite']&bp['bp_stopped']&bp_clamps&~bp['bp_native_cap']&~bp['bp_block_cap']&np.isfinite(bp['bp_own_margin'])&(bp['bp_own_margin']>=-1e-6)
            for block in range(6):
                for mode in MODES:
                    cohort=records[block,mode];row=next(r for r in cohort['rows'] if r['job_id']==job and r['batch']==bi);folder=root/cohort['directory'];actual=load(folder,row);reference=source[mode]
                    assert set(actual)-{'couplings'}==set(reference)-{'couplings'}
                    for name,value in actual.items():
                        if name!='couplings':assert np.array_equal(value,reference[name],equal_nan=True),(block,mode,name)
                    receipt=row['fresh_array_receipt'];assert row['source_parity'] is None and receipt['status']=='fresh_arrays_sealed_without_outcome_join' and receipt['labels_distances_source_outcomes_read'] is False
                    assert receipt['actual_couplings_sha256']==array_sha(lp['couplings']) and receipt['tokens_sha256']==array_sha(tt) and receipt['full_model_inputs']==32
                    chosen,accepted,geometry,high,counts=check_batch(folder,row,actual,lp,bp,root/records[0,'LP96']['directory'],reference_rows['LP96'],tt,mode,ph,ph)
                    assert receipt['actual_subset_inputs']==row['timing']['actual_subset_inputs']
                    c,states,error=counts;candidate_scores+=c;trace_states+=states;maximum=max(maximum,error);repeated+=32
                    if block==0:
                        graph=np.where(chosen==1,lp_graph,bp_graph)
                        for i in range(32):
                            extras={'mixture_component':int(components[bi*32+i]),'dispatch_high':int(high[i])}
                            outputs.append({'input_index':shard*256+bi*32+i,'model_seed':seed,'arm':{'LP96':'LP96','BP_geometry_LP96':'cascade','endpoint_dispatch':'dispatcher'}[mode],
                                'success':int(accepted[i] and graph[i]),'graph_valid_shortest':int(graph[i]),'numerically_accepted':int(accepted[i]),'chosen_model':int(chosen[i]),
                                'routed':int(mode!='LP96' and not geometry[i] and (mode=='BP_geometry_LP96' or high[i])),
                                'direct_LP':int(mode=='endpoint_dispatch' and not high[i]),'BP_executed':int(mode=='BP_geometry_LP96' or (mode=='endpoint_dispatch' and high[i])),
                                'hidden_good':int(lp['answer_hidden_good'][i]),'finite':int(actual['answer_finite'][i]),'stopped':int(actual['answer_stopped'][i]),'native_cap':int(actual['answer_native_cap'][i]),'block_cap':int(actual['answer_block_cap'][i]),
                                'BP_geometry_valid_longer':int(geometry[i] and not bp_graph[i]),**extras})
                            if mode=='BP_geometry_LP96':
                                outputs.append({'input_index':shard*256+bi*32+i,'model_seed':seed,'arm':'BP_AdamW','success':int(bp_accept[i] and bp_graph[i]),'graph_valid_shortest':int(bp_graph[i]),'numerically_accepted':int(bp_accept[i]),'chosen_model':0,'routed':0,'direct_LP':0,'BP_executed':1,
                                    'hidden_good':int(lp['answer_hidden_good'][i]),'finite':int(bp['bp_finite'][i]),'stopped':int(bp['bp_stopped'][i]),'native_cap':int(bp['bp_native_cap'][i]),'block_cap':int(bp['bp_block_cap'][i]),'BP_geometry_valid_longer':int(geometry[i] and not bp_graph[i]),**extras})
    assert len(outputs)==256*3*4 and repeated==256*3*3*6
    return {'shard':shard,'status':'passed','shortest_labels_checked':256,'measured_model_pipeline_copies':repeated,'candidate_score_comparisons':candidate_scores,'active_trace_states':trace_states,'maximum_candidate_score_error':maximum,'rows':outputs,'seconds':time.perf_counter()-started}
