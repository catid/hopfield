"""Independent H2z graph outcomes, control identities and paired statistics."""
import csv
import json
import time
import numpy as np
import study as S
import audit as A
from audit_restart import check_path
from lp_relaxation import PP,DEST,ARMS,protocol


def main():
    started=time.perf_counter();p=protocol();evidence={};rows=[];cases=[];saved={};records=[];cache={};graph_checks=cache_hits=0
    def verify(path,expected=None):
        h=S.sha(path)
        if expected is not None:assert h==expected,path
        evidence[str(path.relative_to(S.ROOT))]=h
    verify(PP)
    for path,h in p['gates'].items():verify(S.ROOT/path,h)
    for group in p['models'].values():
        assert group['lp_native']==group['lp_lambda4']
        for item in group.values():verify(S.ROOT/item['path'],item['sha256'])
    inputs=json.loads((DEST/'independent-input-audit.json').read_text());assert inputs['status']=='passed' and inputs['inputs_checked']==4096 and inputs['protocol_sha256']==S.sha(PP);verify(DEST/'independent-input-audit.json')
    manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['datasets']==inputs['datasets'];data={}
    for ds in manifest['datasets']:
        verify(S.ROOT/ds['path'],ds['sha256'])
        with np.load(S.ROOT/ds['path']) as x:data[ds['shard']]={k:x[k] for k in x.files}
    for job in p['jobs']:
        run=DEST/'runs'/job['id'];wait_start=time.perf_counter()
        while not (run/'result.json').exists():
            if time.perf_counter()-wait_start>900:raise TimeoutError('Completedresult not available in900s')
            time.sleep(.2)
        r=json.loads((run/'result.json').read_text());verify(run/'result.json');assert r['status']=='complete' and r['job']==job and r['protocol_sha256']==S.sha(PP);verify(run/'outputs.npz',r['outputs_sha256'])
        assert r['checkpoint_sha256_by_arm']==[p['models'][str(job['seed'])][arm]['sha256'] for arm in ARMS]
        with np.load(run/'outputs.npz') as x:out={k:x[k] for k in x.files if k!='couplings'}
        d=data[job['shard']];pred=out['predictions'];logits=out['final_logits'];assert pred.shape==logits.shape==(3,256,120,120)
        assert np.array_equal(pred,logits>0) and np.array_equal(out['finite'],np.isfinite(logits).all((2,3)))
        assert np.array_equal(out['stopped'],out['finite']&(out['minimum_flip_margin']>=-1e-6))
        assert np.all(out['answer_steps']%16==0) and np.all((out['answer_steps']>=16)&(out['answer_steps']<=8192)) and np.all(out['answer_steps'][~out['stopped']]==8192)
        valid=np.zeros((3,256),bool);shortest=np.zeros_like(valid)
        for v in range(3):
            for i,(tok,board) in enumerate(zip(d['tokens'],pred[v])):
                key=(int(d['indices'][i]),np.packbits(board).tobytes())
                if key not in cache:cache[key]=check_path(tok,board);graph_checks+=1
                else:cache_hits+=1
                valid[v,i]=cache[key];shortest[v,i]=valid[v,i] and int(board.sum())==int(d['shortest_edges'][i])+1
        success=shortest&out['stopped']&out['finite']&out['hidden_good'][None]&np.isfinite(out['energy']);saved[(job['shard'],job['seed'])]=success
        for v,arm in enumerate(ARMS):
            ep=(pred[v].sum((1,2))==2)&((pred[v]&(d['tokens']>=2)).sum((1,2))==2)
            rows.append({'job':job['id'],'shard':job['shard'],'model_seed':job['seed'],'arm':arm,'n':256,'solved':int(success[v].sum()),'valid_paths':int(valid[v].sum()),'wrong_certified':int((out['stopped'][v]&~shortest[v]).sum()),'timeouts':int((~out['stopped'][v]&out['finite'][v]).sum()),'nonfinite':int((~out['finite'][v]).sum()),'hidden_failures':int((~out['hidden_good']).sum()),'endpoint_only_outputs':int(ep.sum()),'mean_answer_steps':float(out['answer_steps'][v].mean()),'answer_seconds':r['answer_seconds_by_arm'][v],'mean_original_field_residual':float(out['original_field_residual'][v].mean()),'mean_modified_field_residual':float(out['modified_field_residual'][v].mean()),'mean_near_half_free':float(out['near_half_free'][v].mean()),'mean_fractional_free':float(out['fractional_free'][v].mean())})
            for i,index in enumerate(d['indices']):cases.append({'input_index':int(index),'model_seed':job['seed'],'arm':arm,'success':int(success[v,i]),'valid_path':int(valid[v,i]),'shortest_path':int(shortest[v,i]),'answer_steps':int(out['answer_steps'][v,i]),'certified':int(out['stopped'][v,i]),'finite':int(out['finite'][v,i]),'minimum_flip_margin':float(out['minimum_flip_margin'][v,i]),'original_field_residual':float(out['original_field_residual'][v,i]),'modified_field_residual':float(out['modified_field_residual'][v,i]),'near_half_free':int(out['near_half_free'][v,i]),'fractional_free':int(out['fractional_free'][v,i]),'selected_cells':int(pred[v,i].sum()),'shortest_cells':int(d['shortest_edges'][i])+1,'endpoint_only':int(ep[i])})
        records.append(r);S.emit('lp_relaxation_job_audited',job=job['id'])
    joint={arm:np.concatenate([np.stack([saved[(shard,s)][v] for s in range(3)]).all(0) for shard in range(16)]) for v,arm in enumerate(ARMS)}
    def paired(a,b):
        gain=int((a&~b).sum());loss=int((~a&b).sum());gl,gu=A.bounds(gain,4096,alpha=.025);ll,lu=A.bounds(loss,4096,alpha=.025)
        return {'gains':gain,'losses':loss,'observed_net_gain_pp':100*(gain-loss)/4096,'separately_one_sided95_net_lower_pp':100*(gl-lu),'separately_one_sided95_net_upper_pp':100*(gu-ll)}
    gain=paired(joint['lp_lambda4'],joint['lp_native']);gain.update(target_pp=90.,decision='supported' if gain['separately_one_sided95_net_lower_pp']>=90 else 'falsified' if gain['separately_one_sided95_net_upper_pp']<90 else 'inconclusive')
    ni=paired(joint['lp_lambda4'],joint['pl']);ll,lu=A.bounds(ni['losses'],4096,alpha=.05);ni.update(loss_one_sided95_lower_pp=100*ll,loss_one_sided95_upper_pp=100*lu,margin_pp=.1,decision='supported' if 100*lu<=.1 else 'falsified' if 100*ll>.1 else 'inconclusive')
    overall='supported' if gain['decision']==ni['decision']=='supported' else 'falsified' if 'falsified' in [gain['decision'],ni['decision']] else 'inconclusive'
    perinput=[{'input_index':i,**{f'joint_{arm}_success':int(joint[arm][i]) for arm in ARMS}} for i in range(4096)]
    for name,values in [('runs.csv',rows),('per-model-input.csv',cases),('per-input.csv',perinput)]:
        with (DEST/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    result={'status':'validated','utc':S.utc(),'hypothesis':'H2z','gain_primary':gain,'noninferiority_primary':ni,'overall_decision':overall,'joint_success':{k:int(v.sum()) for k,v in joint.items()},'n_unique':4096,'evaluation_jobs':48,'new_scientific_model_maze_evaluations':36864,'new_optimizer_updates':0,'source_gate_trajectories':384,'rows':rows,'records':records,'independent_graph_checks':graph_checks,'exact_board_cache_hits':cache_hits,'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'evidence_sha256':evidence,'input_audit':inputs,'worker_receipts':{str(g):json.loads((DEST/f'eval-worker{g}.json').read_text()) for g in [0,1]},'audit_seconds':time.perf_counter()-started,'scope':'Fixedtheory-chosenbinary-preservinglambda4 onfrozen norm-matchedLPmodels. FreshnativeLP andBPPLcontrols; no graphfallback, targettuning, capextension orrescue. Model/objectivecomparison, not equalcomputeoptimizerbenchmark. Fixedmodel repeats not independentinputs.'}
    assert all(x['status']=='complete' for x in result['worker_receipts'].values());S.save(DEST/'audit.json',result);S.emit('lp_relaxation_audit_complete',gain=gain,noninferiority=ni,overall=overall,joint_success=result['joint_success'],seconds=result['audit_seconds'])


if __name__=='__main__':main()
