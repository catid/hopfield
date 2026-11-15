"""H2an independent graph/flags and three separately decided endpoints."""
import csv
import json
import time
import numpy as np
import study as S
import audit as A
from audit_restart import check_path
from safe_block_transfer import PP,DEST,ARMS,protocol


def main():
    started=time.perf_counter();p=protocol();evidence={};rows=[];cases=[];saved={};skip_saved={};records=[];cache={};graph_checks=cache_hits=0;maximum_cache_error=0.
    def verify(path,expected=None):
        h=S.sha(path)
        if expected is not None:assert h==expected,path
        evidence[str(path.relative_to(S.ROOT))]=h
    verify(PP);inputs=json.loads((DEST/'independent-input-audit.json').read_text());assert inputs['status']=='passed' and inputs['inputs_checked']==4096 and inputs['protocol_sha256']==S.sha(PP);verify(DEST/'independent-input-audit.json');verify(DEST/'numerical-gate.json');manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['datasets']==inputs['datasets'];data={}
    for ds in manifest['datasets']:
        verify(S.ROOT/ds['path'],ds['sha256'])
        with np.load(S.ROOT/ds['path']) as x:data[ds['shard']]={k:x[k] for k in x.files}
    for job in p['jobs']:
        run=DEST/'runs'/job['id'];r=json.loads((run/'result.json').read_text());verify(run/'result.json');assert r['status']=='complete' and r['job']==job and r['protocol_sha256']==S.sha(PP);verify(run/'outputs.npz',r['outputs_sha256']);assert r['checkpoint_sha256_by_arm']==[p['models'][str(job['seed'])][arm]['sha256'] for arm in ARMS]
        with np.load(run/'outputs.npz') as x:out={k:x[k] for k in x.files if k!='couplings'}
        d=data[job['shard']];pred=out['predictions'];logits=out['native_final_logits'];assert pred.shape==(4,256,120,120) and logits.shape==(1,256,120,120)
        assert np.array_equal(pred[:1],logits>0) and np.array_equal(out['finite'][:1],np.isfinite(logits).all((2,3)));assert np.array_equal(out['stopped'][:1],out['finite'][:1]&(out['native_minimum_flip_margin32']>=-1e-6));assert np.all(out['iterations'][:1]%16==0) and np.all((out['iterations'][:1]>=16)&(out['iterations'][:1]<=8192)) and np.all(out['iterations'][:1][~out['stopped'][:1]]==8192)
        assert np.all((out['iterations'][1:2]>=1)&(out['iterations'][1:2]<=512)) and np.all(out['iterations'][1:2][~out['stopped'][1:2]]==512);assert np.all(out['binary_cache_margin'][out['stopped'][1:2]]>=-1e-8) and np.all(out['binary_flip_count']>=0);error=float(abs(out['binary_cache_margin']-out['minimum_flip_margin64'][1:2]).max());assert error<1e-10;maximum_cache_error=max(maximum_cache_error,error)
        assert np.array_equal(out['iterations'][2:],out['iterations'][1:2]+out['post_block_sweeps']);assert np.all((out['block_moves']>=0)&(out['block_moves']<=16));assert not np.any(out['stopped'][2:]&(out['block_native_cap']|out['block_cap']))
        for v in range(4):assert np.array_equal(pred[v][d['tokens']!=1],(d['tokens']>0)[d['tokens']!=1])
        for receipt in r['trace_archives']:
            path=run/receipt['path'];verify(path,receipt['sha256']);v=ARMS.index(receipt['arm']);lo=receipt['offset']
            with np.load(path) as x:ix=x['valid_prefix_indices'];states=x['states_before'];trace=json.loads(x['trace_utf8'].tobytes())
            assert trace['protocol_sha256']==S.sha(PP) and trace['arm']==receipt['arm'] and trace['offset']==lo;assert np.array_equal(ix,np.flatnonzero(out['stopped'][1,lo:lo+64]&out['finite'][1,lo:lo+64]));assert len(states)==len(trace['trace'])<=16
            if len(ix):assert np.array_equal(states[0],pred[1,lo:lo+64][ix])
        unique=np.sum([x['unique_operation_seconds'] for x in r['batch_timings']],0);charged=unique.copy();charged[2:]+=unique[1];assert np.array_equal(unique,r['unique_answer_seconds_by_operation']) and np.allclose(charged,r['answer_seconds_by_arm'],rtol=0,atol=1e-12)
        valid=np.zeros((4,256),bool);shortest=np.zeros_like(valid)
        for v in range(4):
            for i,(tok,board) in enumerate(zip(d['tokens'],pred[v])):
                key=(int(d['indices'][i]),np.packbits(board).tobytes())
                if key not in cache:cache[key]=check_path(tok,board);graph_checks+=1
                else:cache_hits+=1
                valid[v,i]=cache[key];shortest[v,i]=valid[v,i] and int(board.sum())==int(d['shortest_edges'][i])+1
        certified=(out['minimum_flip_margin64']>=-1e-6)&out['stopped']&out['finite'];success=shortest&certified&out['hidden_good'][None]&np.isfinite(out['energy64']);saved[(job['shard'],job['seed'])]=success;skip_saved[(job['shard'],job['seed'])]=np.where(valid[1],success[1],success[2])
        for v,arm in enumerate(ARMS):
            rows.append({'job':job['id'],'shard':job['shard'],'model_seed':job['seed'],'arm':arm,'n':256,'solved':int(success[v].sum()),'valid_paths':int(valid[v].sum()),'wrong_certified':int((certified[v]&~shortest[v]).sum()),'timeouts':int((~out['stopped'][v]&out['finite'][v]).sum()),'nonfinite':int((~out['finite'][v]).sum()),'hidden_failures':int((~out['hidden_good']).sum()),'iteration_unit':'updates' if v==0 else 'sweeps','mean_iterations':float(out['iterations'][v].mean()),'answer_seconds':r['answer_seconds_by_arm'][v],'unique_operation_seconds':unique[v],'block_moves':int(out['block_moves'][v-2].sum()) if v>=2 else 0,'native_cap':int(out['block_native_cap'][v-2].sum()) if v>=2 else 0,'block_cap':int(out['block_cap'][v-2].sum()) if v>=2 else 0,'minimum_flip_margin64':float(out['minimum_flip_margin64'][v].min())})
            for i,index in enumerate(d['indices']):cases.append({'input_index':int(index),'model_seed':job['seed'],'arm':arm,'success':int(success[v,i]),'valid_path':int(valid[v,i]),'shortest_path':int(shortest[v,i]),'iterations':int(out['iterations'][v,i]),'stopped':int(out['stopped'][v,i]),'certified64':int(certified[v,i]),'finite':int(out['finite'][v,i]),'minimum_flip_margin64':float(out['minimum_flip_margin64'][v,i]),'energy64':float(out['energy64'][v,i]),'selected_cells':int(pred[v,i].sum()),'shortest_cells':int(d['shortest_edges'][i])+1,'block_moves':int(out['block_moves'][v-2,i]) if v>=2 else 0})
        records.append(r);S.emit('safe_block_job_audited',job=job['id'])
    joint={arm:np.concatenate([np.stack([saved[(shard,s)][v] for s in range(3)]).all(0) for shard in range(16)]) for v,arm in enumerate(ARMS)}
    def paired(a,b):
        gain=int((a&~b).sum());loss=int((~a&b).sum());gl,gu=A.bounds(gain,4096,alpha=.025);ll,lu=A.bounds(loss,4096,alpha=.025)
        return {'gains':gain,'losses':loss,'observed_net_gain_pp':100*(gain-loss)/4096,'separately_one_sided95_net_lower_pp':100*(gl-lu),'separately_one_sided95_net_upper_pp':100*(gu-ll)}
    loss=int((joint['old_binary']&~joint['connectivity_veto']).sum());lo,hi=A.bounds(loss,4096,alpha=.05);regression={'regressions':loss,'n':4096,'one_sided95_lower':lo,'one_sided95_upper':hi,'maximum_probability':.001,'decision':'supported' if hi<=.001 else 'falsified' if lo>.001 else 'inconclusive'}
    opportunity=joint['old_blocks']&~joint['old_binary'];n=int(opportunity.sum());retained=int((opportunity&joint['connectivity_veto']).sum())
    if n:lo,hi=A.bounds(retained,n,alpha=.05);decision='supported' if lo>=.9 else 'falsified' if hi<.9 else 'inconclusive'
    else:lo=hi=None;decision='inconclusive'
    retention={'opportunities':n,'retained':retained,'one_sided95_lower':lo,'one_sided95_upper':hi,'minimum_conditional_probability':.9,'decision':decision};endpoints={'veto_regression_probability':regression,'retain_original_gains':retention};original_losses=joint['old_binary']&~joint['old_blocks'];secondary={'veto_vs_binary':paired(joint['connectivity_veto'],joint['old_binary']),'veto_vs_original_blocks':paired(joint['connectivity_veto'],joint['old_blocks']),'veto_vs_bp':paired(joint['connectivity_veto'],joint['pl']),'original_block_regressions':int(original_losses.sum()),'original_regressions_avoided':int((original_losses&joint['connectivity_veto']).sum()),'scope':'Descriptive paired comparisons; originalregression avoidance is separate fromthetwo primaryendpoints.'}
    skip_joint=np.concatenate([np.stack([skip_saved[(shard,seed)] for seed in range(3)]).all(0) for shard in range(16)]);skip={'joint_success':int(skip_joint.sum()),'gains':int((skip_joint&~joint['old_binary']).sum()),'losses':int((~skip_joint&joint['old_binary']).sum()),'new_searches':0,'scope':'Descriptive exactmapping: initialsimplepath→binary, otherwiseoriginalblocks; zero-lossconditionalidentityproof separate.'};perinput=[{'input_index':i,**{f'joint_{arm}_success':int(joint[arm][i]) for arm in ARMS},'descriptive_skip_simple_path_success':int(skip_joint[i])} for i in range(4096)]
    for name,values in [('runs.csv',rows),('per-model-input.csv',cases),('per-input.csv',perinput)]:
        with (DEST/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    workers={str(g):json.loads((DEST/f'eval-worker{g}.json').read_text()) for g in [0,1]};assert all(x['status']=='complete' and x['wall_seconds']<=300 for x in workers.values());result={'status':'validated','utc':S.utc(),'hypothesis':'H2an','primary_endpoints':endpoints,'joint_success':{k:int(v.sum()) for k,v in joint.items()},'skip_descriptive':skip,'secondary':secondary,'n_unique':4096,'evaluation_jobs':48,'new_scientific_model_maze_arm_evaluations':49152,'reused_prefixes':24576,'new_optimizer_updates':0,'rows':rows,'records':records,'independent_graph_checks':graph_checks,'exact_board_cache_hits':cache_hits,'maximum_cache_vs_upstream_float64_margin_error':maximum_cache_error,'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'evidence_sha256':evidence,'input_audit':inputs,'worker_receipts':workers,'audit_elapsed_seconds':time.perf_counter()-started,'scope':'Fresh factorial, fixed models/policy, independent input as inferential unit. Two separate primary endpoint decisions and descriptive skip comparison; no simultaneous95claim or overall conjunction.'};S.save(DEST/'audit.json',result);S.emit('safe_block_audit_complete',endpoints=endpoints,joint_success=result['joint_success'],skip_descriptive=skip)


if __name__=='__main__':main()
