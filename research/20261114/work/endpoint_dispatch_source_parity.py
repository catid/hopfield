"""H2br source gate against immutable H2bo/H2bp pair0 actual executions."""
import json
from pathlib import Path
import numpy as np
from bp_first_geometry_parity import sha,array_sha,terminal_records,trace_rows,compare_trace
from audit_bp_first_fresh_states import numeric_error
from audit_connectivity_screen import path_valid


def verify(result,tokens,source,job,bi,bound_receipts):
    source=Path(source);loaded={};rows={};directories={}
    for mode in ['LP96','BP_geometry_LP96']:
        directory=source/'runs'/f'pair0-{mode}';receipt=directory/'result.json'
        assert sha(receipt)==bound_receipts[mode]
        c=json.loads(receipt.read_text());row=next(r for r in c['rows'] if r['job_id']==job and r['batch']==bi)
        file=directory/row['file'];assert sha(file)==row['file_sha256']
        with np.load(file,allow_pickle=False) as z:loaded[mode]={k:z[k] for k in z.files}
        rows[mode]=row;directories[mode]=directory
    bp,lp=loaded['BP_geometry_LP96'],loaded['LP96'];a=result['audit'];mode=result['timing']['mode'];n=len(tokens)
    assert np.array_equal(a['couplings'],lp['couplings'],equal_nan=True) and np.array_equal(a['couplings'],bp['couplings'],equal_nan=True)
    assert np.array_equal(a['hidden_steps'],lp['hidden_steps']) and np.array_equal(a['hidden_steps'],bp['hidden_steps'])
    assert np.array_equal(result['answer']['hidden_good'],lp['answer_hidden_good']) and np.array_equal(result['answer']['hidden_good'],bp['answer_hidden_good'])
    if mode=='endpoint_dispatch':
        positions=np.array([[np.flatnonzero(t.ravel()==v)[0] for v in [2,3]] for t in tokens])
        distances=np.abs(positions[:,0]//180-positions[:,1]//180)+np.abs(positions[:,0]%180-positions[:,1]%180)
        bi_actual=np.flatnonzero(distances>=182)
        assert np.array_equal(a['endpoint_manhattan'],distances) and np.array_equal(a['dispatch_high'],distances>=182) and np.array_equal(a['bp_indices'],bi_actual)
    else:bi_actual=np.arange(n) if mode=='BP_geometry_LP96' else np.array([],np.int64)
    if len(bi_actual):
        for key in ['prediction','iterations','stopped','finite','native_cap','block_cap']:assert np.array_equal(result['bp_prefix'][key],bp['bp_'+key][bi_actual]),key
        for key in ['bp_final_logits','bp_minimum_flip_margin32','bp_energy64']:numeric_error(a[key],bp[key][bi_actual],1e-6 if key=='bp_energy64' else 1e-9)
        numeric_error(result['bp_prefix']['own_margin'],bp['bp_own_margin'][bi_actual],1e-9)
    if mode=='LP96':
        ix=np.arange(n);selected=np.ones(n,np.int8)
    else:
        geometry=np.array([path_valid(t,b) for t,b in zip(tokens[bi_actual],bp['bp_prediction'][bi_actual])],bool)
        assert np.array_equal(result['geometry_valid'],geometry)
        selected=np.ones(n,np.int8);selected[bi_actual[geometry]]=0;ix=np.flatnonzero(selected)
        assert np.array_equal(result['subset_indices'],ix)
    if len(ix):
        assert result['core'] is not None
        for name in ['prediction','iterations','stopped','finite','block_moves','post_block_sweeps','native_cap','block_cap','initial_simple_path']:
            assert np.array_equal(a['LP_'+name],lp['LP_'+name][:,ix]),name
        for name in ['LP_minimum_flip_margin64','LP_energy64']:numeric_error(a[name],lp[name][:,ix],1e-6 if name=='LP_energy64' else 1e-9)
        for name in ['LP_binary_flip_count','LP_binary_cache_margin']:numeric_error(a[name],lp[name][ix],1e-9)
        actual=result['answer'] if mode=='LP96' else result['lp_subset']
        numeric_error(actual['own_margin'],lp['answer_own_margin'][ix],1e-9)
    else:assert result['core'] is None
    for name in ['prediction','iterations','stopped','finite','native_cap','block_cap','own_margin']:
        expected=bp['bp_'+name].copy();expected[ix]=lp['answer_'+name][ix]
        if name=='own_margin':numeric_error(result['answer'][name],expected,1e-9)
        else:assert np.array_equal(result['answer'][name],expected),name
    assert np.array_equal(result['answer']['chosen_model'],selected)
    expected=bp['answer_accepted'].copy();expected[ix]=lp['answer_accepted'][ix]
    assert np.array_equal(result['answer']['accepted'],expected)
    totals={'candidate_comparisons':0,'active_trace_states':0,'maximum_candidate_score_error':0.}
    if len(ix):
        allowed=set(map(int,ix));core=result['core'];control=rows['LP96']
        for v,filename in enumerate(control['trace_files'],start=2):
            file=directories['LP96']/filename;assert sha(file)==control['trace_sha256'][filename]
            with np.load(file,allow_pickle=False) as z:old={k:z[k] for k in z.files}
            payload=json.loads(old['trace_utf8'].tobytes());block=core['blocks'][v];owners=ix[block['valid_prefix_indices']];original_owners=old['valid_prefix_indices']
            assert np.array_equal(owners,original_owners[np.isin(original_owners,ix)])
            assert np.array_equal(block['initial_terminal'],old['initial_terminal'][ix])
            assert terminal_records(block['initial_terminal_records'],ix,allowed)==terminal_records(payload['initial_terminal_checks'],np.arange(n),allowed)
            raw=block['raw'];trace=raw['trace'] if raw else [];states=raw['states_before'] if raw else np.zeros((0,0,180,180),bool)
            comparison=compare_trace(trace_rows(trace,states,owners,allowed),trace_rows(payload['trace'],old['states_before'],original_owners,allowed))
            for key in ['candidate_comparisons','active_trace_states']:totals[key]+=comparison[key]
            totals['maximum_candidate_score_error']=max(totals['maximum_candidate_score_error'],comparison['maximum_candidate_score_error'])
            assert terminal_records(raw.get('terminal_check_records',[]) if raw else [],owners,allowed)==terminal_records(payload['final_terminal_checks'],original_owners,allowed)
            positions=np.flatnonzero(np.isin(original_owners,ix))
            for name in ['pre_terminal_stopped','pre_terminal_block_cap']:
                actual=raw.get(name,np.zeros(0,bool)) if raw else np.zeros(0,bool);expected=old[name][positions] if len(old[name]) else np.zeros(0,bool);assert np.array_equal(actual,expected)
    return {'status':'passed','mode':mode,'full_model_inputs':n,'actual_BP_inputs':len(bi_actual),'actual_LP_inputs':len(ix),'actual_couplings_sha256':array_sha(a['couplings']),'reference_receipts':bound_receipts,**totals}
