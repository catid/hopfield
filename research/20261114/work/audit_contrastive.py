"""Independent fixed-endpoint ranking, graph outcomes and matched-fit audit."""
import csv
import json
import time
import numpy as np
import study as S
import audit as A
from audit_restart import check_path
from gate_contrastive import independent_energy
from contrastive_training import PP,DEST,ARMS,protocol


def main():
    started=time.perf_counter();torch,_,_=S.imports();p=protocol();evidence={};models={};fitrecords=[]
    def verify(path,expected=None):
        h=S.sha(path)
        if expected is not None:assert h==expected,path
        evidence[str(path.relative_to(S.ROOT))]=h
    verify(PP);verify(DEST/'training-manifest.json',p['training_manifest_sha256']);verify(DEST/'training-arrays.npz',p['training_manifest']['arrays_sha256'])
    for path,h in p['gates'].items():verify(S.ROOT/path,h)
    for seed in range(3):
        original=S.model_for(seed,120,'original','cpu');initial=original.state_dict();models[seed]=[]
        for arm in ARMS:
            model=S.model_for(seed,120,'original','cpu',torch.float64)
            if arm!='unchanged':
                run=DEST/f'fits/s{seed}-{arm}';r=json.loads((run/'result.json').read_text());verify(run/'result.json');assert r['status']=='complete' and r['protocol_sha256']==S.sha(PP) and r['updates']==400
                for filename,key in [('checkpoint.pt','checkpoint_sha256'),('optimizer.pt','optimizer_sha256'),('trace.json','trace_sha256')]:verify(run/filename,r[key])
                schedule=p['minibatch_schedules'][str(seed)];assert r['schedule_sha256']==schedule['sha256'];verify(S.ROOT/schedule['path'],schedule['sha256'])
                with np.load(S.ROOT/schedule['path']) as x:assert x['indices'].shape==(400,64) and np.all((x['indices']>=0)&(x['indices']<1000))
                state=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True);opt=torch.load(run/'optimizer.pt',map_location='cpu',weights_only=True)
                assert all(torch.equal(initial[k],state[k]) for k in initial if k not in ['T','b','kappa']);assert all(not torch.equal(initial[k],state[k]) for k in ['T','b','kappa']);assert all(int(v['step'])==400 for v in opt['state'].values())
                group=opt['param_groups'][0];assert group['lr']==.01 and group['weight_decay']==0. and group['betas']==(.9,.999) and group['eps']==1e-8 and group['foreach']==False
                trace=json.loads((run/'trace.json').read_text());assert len(trace)==400 and [x['step'] for x in trace]==list(range(1,401));assert all(np.isfinite(list(x.values())).all() for x in trace)
                model.load_state_dict(state);fitrecords.append(r|{'first_training_loss':trace[0],'last_training_loss':trace[-1]})
            models[seed].append(model)
    for seed in range(3):
        pair=[r for r in fitrecords if r['job']['seed']==seed];assert len(pair)==2 and pair[0]['schedule_sha256']==pair[1]['schedule_sha256'] and pair[0]['initial_checkpoint_sha256']==pair[1]['initial_checkpoint_sha256'] and pair[0]['readout_sha256']==pair[1]['readout_sha256']
    inputs=json.loads((DEST/'independent-input-audit.json').read_text());verify(DEST/'independent-input-audit.json');assert inputs['status']=='passed' and inputs['inputs_checked']==4096 and inputs['protocol_sha256']==S.sha(PP)
    manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['datasets']==inputs['datasets'];data={}
    for ds in manifest['datasets']:
        verify(S.ROOT/ds['path'],ds['sha256'])
        with np.load(S.ROOT/ds['path']) as x:data[ds['shard']]={k:x[k] for k in x.files}
    records=[];rows=[];cases=[];saved={};ranked={};cache={};maximum_rank_error=0.;graph_checks=cache_hits=0
    for job in p['jobs']:
        run=DEST/'runs'/job['id'];verify(run/'result.json');r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['job']==job and r['protocol_sha256']==S.sha(PP);verify(run/'outputs.npz',r['outputs_sha256'])
        with np.load(run/'outputs.npz') as x:out={k:x[k] for k in x.files}
        d=data[job['shard']];pred=out['predictions'];logits=out['final_logits'];assert pred.shape==logits.shape==(3,256,120,120);assert np.array_equal(pred,logits>0);assert np.array_equal(out['finite'],np.isfinite(logits).all((2,3)))
        assert np.array_equal(out['stopped'],out['finite']&(out['minimum_flip_margin']>=-1e-6));assert np.all(out['answer_steps']%16==0) and np.all((out['answer_steps']>=16)&(out['answer_steps']<=8192))
        assert np.all(out['answer_steps'][~out['stopped']]==8192)
        gaps=[]
        with torch.no_grad():
            for lo in range(0,256,64):
                t=torch.tensor(d['tokens'][lo:lo+64].astype(np.int64));y=torch.tensor(d['labels'][lo:lo+64],dtype=torch.float64);a=torch.tensor(out['couplings'][lo:lo+64],dtype=torch.float64);zero=torch.zeros_like(y);count=(y*(t==1)).sum((1,2)).clamp_min(1)
                gap=np.stack([((independent_energy(torch,m,t,y,a)-independent_energy(torch,m,t,zero,a))/count).numpy() for m in models[job['seed']]]);gaps.append(gap)
        gaps=np.concatenate(gaps,axis=1);error=float(np.max(abs(gaps-out['normalized_rank_gap'])));maximum_rank_error=max(error,maximum_rank_error);assert error<1e-8,(job,error)
        ranking=np.isfinite(gaps)&(gaps<-1e-6);assert np.array_equal(ranking,np.isfinite(out['normalized_rank_gap'])&(out['normalized_rank_gap']<-1e-6))
        valid=np.zeros((3,256),bool);shortest=np.zeros_like(valid)
        for v in range(3):
            for i,(tok,board) in enumerate(zip(d['tokens'],pred[v])):
                key=(int(d['indices'][i]),np.packbits(board).tobytes())
                if key not in cache:cache[key]=check_path(tok,board);graph_checks+=1
                else:cache_hits+=1
                valid[v,i]=cache[key];shortest[v,i]=valid[v,i] and int(board.sum())==int(d['shortest_edges'][i])+1
        success=shortest&out['stopped']&out['finite']&out['hidden_good'][None]&np.isfinite(out['energy']);saved[(job['shard'],job['seed'])]=success;ranked[(job['shard'],job['seed'])]=ranking
        for v,arm in enumerate(ARMS):
            endpoint=(pred[v].sum((1,2))==2)&((pred[v]&(d['tokens']>=2)).sum((1,2))==2)
            rows.append({'job':job['id'],'shard':job['shard'],'model_seed':job['seed'],'arm':arm,'n':256,'solved':int(success[v].sum()),'valid_paths':int(valid[v].sum()),'label_ranked_below_endpoints':int(ranking[v].sum()),'mean_normalized_rank_gap':float(gaps[v].mean()),'endpoint_only_outputs':int(endpoint.sum()),'wrong_certified':int((out['stopped'][v]&~shortest[v]).sum()),'timeouts':int((~out['stopped'][v]&out['finite'][v]).sum()),'nonfinite':int((~out['finite'][v]).sum()),'hidden_failures':int((~out['hidden_good']).sum()),'mean_answer_steps':float(out['answer_steps'][v].mean()),'answer_seconds':r['answer_seconds_by_arm'][v]})
            for i,index in enumerate(d['indices']):cases.append({'input_index':int(index),'model_seed':job['seed'],'arm':arm,'success':int(success[v,i]),'valid_path':int(valid[v,i]),'shortest_path':int(shortest[v,i]),'label_ranked_below_endpoints':int(ranking[v,i]),'normalized_label_minus_endpoint_energy':float(gaps[v,i]),'answer_steps':int(out['answer_steps'][v,i]),'certified':int(out['stopped'][v,i]),'finite':int(out['finite'][v,i]),'minimum_flip_margin':float(out['minimum_flip_margin'][v,i]),'selected_cells':int(pred[v,i].sum()),'shortest_cells':int(d['shortest_edges'][i])+1,'endpoint_only':int(endpoint[i])})
        records.append(r);S.emit('contrastive_job_audited',job=job['id'],max_rank_error=error)
    joint={arm:np.concatenate([np.stack([saved[(shard,s)][v] for s in range(3)]).all(0) for shard in range(16)]) for v,arm in enumerate(ARMS)};rankjoint={arm:np.concatenate([np.stack([ranked[(shard,s)][v] for s in range(3)]).all(0) for shard in range(16)]) for v,arm in enumerate(ARMS)}
    def paired(a,b):
        gain=int((a&~b).sum());loss=int((~a&b).sum());gl,gu=A.bounds(gain,4096,alpha=.025);ll,lu=A.bounds(loss,4096,alpha=.025)
        return {'gains':gain,'losses':loss,'observed_net_gain_pp':100*(gain-loss)/4096,'separately_one_sided95_net_lower_pp':100*(gl-lu),'separately_one_sided95_net_upper_pp':100*(gu-ll)}
    ranking=paired(rankjoint['contrastive'],rankjoint['pl']);ranking.update(target_pp=90.,decision='supported' if ranking['separately_one_sided95_net_lower_pp']>=90 else 'falsified' if ranking['separately_one_sided95_net_upper_pp']<90 else 'inconclusive')
    accuracy=paired(joint['contrastive'],joint['pl']);ll,lu=A.bounds(accuracy['losses'],4096,alpha=.05);accuracy.update(loss_one_sided95_lower_pp=100*ll,loss_one_sided95_upper_pp=100*lu,noninferiority_margin_pp=.1,decision='supported' if 100*lu<=.1 else 'falsified' if 100*ll>.1 else 'inconclusive')
    overall='supported' if ranking['decision']==accuracy['decision']=='supported' else 'falsified' if 'falsified' in [ranking['decision'],accuracy['decision']] else 'inconclusive'
    perinput=[{'input_index':i,**{f'joint_{arm}_success':int(joint[arm][i]) for arm in ARMS},**{f'joint_{arm}_label_ranked_below_endpoints':int(rankjoint[arm][i]) for arm in ARMS}} for i in range(4096)]
    for name,values in [('runs.csv',rows),('per-model-input.csv',cases),('per-input.csv',perinput)]:
        with (DEST/name).open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    result={'status':'validated','utc':S.utc(),'hypothesis':'H2v','ranking_primary':ranking,'accuracy_primary':accuracy,'overall_decision':overall,'joint_success':{k:int(v.sum()) for k,v in joint.items()},'joint_label_ranked_below_endpoints':{k:int(v.sum()) for k,v in rankjoint.items()},'comparisons_vs_unchanged':{arm:paired(joint[arm],joint['unchanged']) for arm in ['pl','contrastive']},'n_unique':4096,'evaluation_jobs':48,'fit_jobs':6,'new_scientific_model_maze_evaluations':36864,'training_minibatch_presentations_separate':153600,'train_updates_separate':2400,'rows':rows,'records':records,'fits':fitrecords,'independent_graph_checks':graph_checks,'exact_board_cache_hits':cache_hits,'all_rankings_independently_recomputed':36864,'maximum_independent_normalized_rank_gap_error':maximum_rank_error,'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'evidence_sha256':evidence,'input_audit':inputs,'worker_receipts':{f'{kind}{gpu}':json.loads((DEST/f'{kind}-worker{gpu}.json').read_text()) for kind in ['train','eval'] for gpu in [0,1]},'audit_seconds':time.perf_counter()-started,'scope':'Answer-only BPAdamW continuedPL versus fixed0.1endpointcontrastive addition. Original1000trainingarrays, three fixed trained-model starts, fresh120x120loop20cohort. Label use is explicitly trainingloss and offlineevaluation. Nativeinference sees only graph tokens/frozenreadout, never labels. Pairwiseenergy ranking does not certify globaloptimality.'}
    assert all(x['status']=='complete' for x in result['worker_receipts'].values());S.save(DEST/'audit.json',result);S.emit('contrastive_audit_complete',ranking=ranking,accuracy=accuracy,overall=overall,joint_success=result['joint_success'],joint_ranking=result['joint_label_ranked_below_endpoints'],seconds=result['audit_seconds'])


if __name__=='__main__':main()
