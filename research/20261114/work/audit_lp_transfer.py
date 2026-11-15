"""Independent H2y scale/state, bothrankings, graphoutcomes and pairedendpoints."""
import csv
import json
import time
import numpy as np
import study as S
import audit as A
from audit_restart import check_path
from gate_contrastive import independent_energy
from lp_transfer import PP,DEST,ARMS,protocol


def main():
    started=time.perf_counter();torch,_,_=S.imports();p=protocol();evidence={};models={};modelchecks=[]
    def verify(path,expected=None):
        h=S.sha(path)
        if expected is not None:assert h==expected,path
        evidence[str(path.relative_to(S.ROOT))]=h
    verify(PP)
    for path,h in p['gates'].items():verify(S.ROOT/path,h)
    for seed in range(3):
        original=S.model_for(seed,120,'original','cpu');initial=original.state_dict();models[seed]=[]
        castrow=p['cast_receipt']['records'][seed];theta=np.load(S.OUT/f'training_min_norm/seed{seed}/candidate-theta.npy');original_vector=np.concatenate([initial['T'].numpy().ravel(),initial['b'].numpy(),initial['kappa'].numpy()]);scale=float(abs(original_vector).max().astype(np.float64)/abs(theta).max());assert scale==castrow['positive_scale'] and scale>0;expected=np.asarray(theta*scale,np.float32)
        for arm in ARMS:
            item=p['models'][str(seed)][arm];verify(S.ROOT/item['path'],item['sha256']);state=torch.load(S.ROOT/item['path'],map_location='cpu',weights_only=True);assert all(torch.equal(initial[k],state[k]) for k in initial if k not in ['T','b','kappa']);model=S.model_for(seed,120,'original','cpu',torch.float64);model.load_state_dict(state);models[seed].append(model)
            if arm=='lp_scaled':
                actual=np.concatenate([state['T'].numpy().ravel(),state['b'].numpy(),state['kappa'].numpy()]);assert np.array_equal(actual,expected) and float(abs(actual).max())==float(abs(original_vector).max())
        modelchecks.append({'seed':seed,'scale_exact':True,'float32_cast_exact':True,'all_hidden_tensors_unchanged':True,'original_and_lp_linf':float(abs(original_vector).max())})
    inputs=json.loads((DEST/'independent-input-audit.json').read_text());verify(DEST/'independent-input-audit.json');assert inputs['status']=='passed' and inputs['inputs_checked']==4096 and inputs['protocol_sha256']==S.sha(PP)
    manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['datasets']==inputs['datasets'];data={}
    for ds in manifest['datasets']:
        verify(S.ROOT/ds['path'],ds['sha256'])
        with np.load(S.ROOT/ds['path']) as x:data[ds['shard']]={k:x[k] for k in x.files}
    records=[];rows=[];cases=[];saved={};ranked={};endpoint_rank={};dense_rank={};cache={};maximum_error={'endpoint':0.,'dense':0.};graph_checks=cache_hits=0
    for job in p['jobs']:
        run=DEST/'runs'/job['id'];wait_start=time.perf_counter()
        while not (run/'result.json').exists():
            if time.perf_counter()-wait_start>300:raise TimeoutError('Completedresult transfer not available in300s')
            time.sleep(.2)
        verify(run/'result.json');r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['job']==job and r['protocol_sha256']==S.sha(PP);verify(run/'outputs.npz',r['outputs_sha256']);assert r['checkpoint_sha256_by_arm']==[p['models'][str(job['seed'])][arm]['sha256'] for arm in ARMS]
        with np.load(run/'outputs.npz') as x:out={k:x[k] for k in x.files}
        d=data[job['shard']];pred=out['predictions'];logits=out['final_logits'];assert pred.shape==logits.shape==(3,256,120,120);assert np.array_equal(pred,logits>0);assert np.array_equal(out['finite'],np.isfinite(logits).all((2,3)));assert np.array_equal(out['stopped'],out['finite']&(out['minimum_flip_margin']>=-1e-6));assert np.all(out['answer_steps']%16==0) and np.all((out['answer_steps']>=16)&(out['answer_steps']<=8192));assert np.all(out['answer_steps'][~out['stopped']]==8192)
        endpoints=[];denses=[]
        with torch.no_grad():
            for lo in range(0,256,64):
                t=torch.tensor(d['tokens'][lo:lo+64].astype(np.int64));y=torch.tensor(d['labels'][lo:lo+64],dtype=torch.float64);a=torch.tensor(out['couplings'][lo:lo+64],dtype=torch.float64);zero=torch.zeros_like(y);ones=torch.ones_like(y);on=(y*(t==1)).sum((1,2)).clamp_min(1);off=((1-y)*(t==1)).sum((1,2)).clamp_min(1);ep=[];dn=[]
                for model in models[job['seed']]:
                    energy=independent_energy(torch,model,t,y,a);ep.append(((energy-independent_energy(torch,model,t,zero,a))/on).numpy());dn.append(((energy-independent_energy(torch,model,t,ones,a))/off).numpy())
                endpoints.append(np.stack(ep));denses.append(np.stack(dn))
        endpoint=np.concatenate(endpoints,axis=1);dense=np.concatenate(denses,axis=1)
        for key,value,stored in [('endpoint',endpoint,'normalized_endpoint_gap'),('dense',dense,'normalized_dense_gap')]:
            error=float(np.max(abs(value-out[stored])));maximum_error[key]=max(maximum_error[key],error);assert error<1e-8,(job,key,error)
        e_rank=np.isfinite(endpoint)&(endpoint<-1e-6);d_rank=np.isfinite(dense)&(dense<-1e-6);ranking=e_rank&d_rank;assert np.array_equal(ranking,(out['normalized_endpoint_gap']<-1e-6)&(out['normalized_dense_gap']<-1e-6))
        valid=np.zeros((3,256),bool);shortest=np.zeros_like(valid)
        for v in range(3):
            for i,(tok,board) in enumerate(zip(d['tokens'],pred[v])):
                key=(int(d['indices'][i]),np.packbits(board).tobytes())
                if key not in cache:cache[key]=check_path(tok,board);graph_checks+=1
                else:cache_hits+=1
                valid[v,i]=cache[key];shortest[v,i]=valid[v,i] and int(board.sum())==int(d['shortest_edges'][i])+1
        success=shortest&out['stopped']&out['finite']&out['hidden_good'][None]&np.isfinite(out['energy']);key=(job['shard'],job['seed']);saved[key]=success;ranked[key]=ranking;endpoint_rank[key]=e_rank;dense_rank[key]=d_rank
        for v,arm in enumerate(ARMS):
            epboard=(pred[v].sum((1,2))==2)&((pred[v]&(d['tokens']>=2)).sum((1,2))==2)
            rows.append({'job':job['id'],'shard':job['shard'],'model_seed':job['seed'],'arm':arm,'n':256,'solved':int(success[v].sum()),'valid_paths':int(valid[v].sum()),'label_below_both':int(ranking[v].sum()),'label_below_endpoints':int(e_rank[v].sum()),'label_below_allopen':int(d_rank[v].sum()),'mean_normalized_endpoint_gap':float(endpoint[v].mean()),'mean_normalized_dense_gap':float(dense[v].mean()),'endpoint_only_outputs':int(epboard.sum()),'wrong_certified':int((out['stopped'][v]&~shortest[v]).sum()),'timeouts':int((~out['stopped'][v]&out['finite'][v]).sum()),'nonfinite':int((~out['finite'][v]).sum()),'hidden_failures':int((~out['hidden_good']).sum()),'mean_answer_steps':float(out['answer_steps'][v].mean()),'answer_seconds':r['answer_seconds_by_arm'][v]})
            for i,index in enumerate(d['indices']):cases.append({'input_index':int(index),'model_seed':job['seed'],'arm':arm,'success':int(success[v,i]),'valid_path':int(valid[v,i]),'shortest_path':int(shortest[v,i]),'label_below_both':int(ranking[v,i]),'label_below_endpoints':int(e_rank[v,i]),'label_below_allopen':int(d_rank[v,i]),'normalized_endpoint_gap':float(endpoint[v,i]),'normalized_dense_gap':float(dense[v,i]),'answer_steps':int(out['answer_steps'][v,i]),'certified':int(out['stopped'][v,i]),'finite':int(out['finite'][v,i]),'minimum_flip_margin':float(out['minimum_flip_margin'][v,i]),'selected_cells':int(pred[v,i].sum()),'shortest_cells':int(d['shortest_edges'][i])+1,'endpoint_only':int(epboard[i])})
        records.append(r);S.emit('lp_transfer_job_audited',job=job['id'])
    def joint(values):return {arm:np.concatenate([np.stack([values[(shard,s)][v] for s in range(3)]).all(0) for shard in range(16)]) for v,arm in enumerate(ARMS)}
    solved=joint(saved);rankjoint=joint(ranked);ejoint=joint(endpoint_rank);djoint=joint(dense_rank)
    def paired(a,b):
        gain=int((a&~b).sum());loss=int((~a&b).sum());gl,gu=A.bounds(gain,4096,alpha=.025);ll,lu=A.bounds(loss,4096,alpha=.025)
        return {'gains':gain,'losses':loss,'observed_net_gain_pp':100*(gain-loss)/4096,'separately_one_sided95_net_lower_pp':100*(gl-lu),'separately_one_sided95_net_upper_pp':100*(gu-ll)}
    ranking=paired(rankjoint['lp_scaled'],rankjoint['pl']);ranking.update(target_pp=90.,decision='supported' if ranking['separately_one_sided95_net_lower_pp']>=90 else 'falsified' if ranking['separately_one_sided95_net_upper_pp']<90 else 'inconclusive');accuracy=paired(solved['lp_scaled'],solved['pl']);ll,lu=A.bounds(accuracy['losses'],4096,alpha=.05);accuracy.update(loss_one_sided95_lower_pp=100*ll,loss_one_sided95_upper_pp=100*lu,noninferiority_margin_pp=.1,decision='supported' if 100*lu<=.1 else 'falsified' if 100*ll>.1 else 'inconclusive');overall='supported' if ranking['decision']==accuracy['decision']=='supported' else 'falsified' if 'falsified' in [ranking['decision'],accuracy['decision']] else 'inconclusive'
    perinput=[{'input_index':i,**{f'joint_{arm}_success':int(solved[arm][i]) for arm in ARMS},**{f'joint_{arm}_label_below_both':int(rankjoint[arm][i]) for arm in ARMS}} for i in range(4096)]
    for name,values in [('runs.csv',rows),('per-model-input.csv',cases),('per-input.csv',perinput)]:
        with (DEST/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    result={'status':'validated','utc':S.utc(),'hypothesis':'H2y','ranking_primary':ranking,'accuracy_primary':accuracy,'overall_decision':overall,'joint_success':{k:int(v.sum()) for k,v in solved.items()},'joint_label_below_both':{k:int(v.sum()) for k,v in rankjoint.items()},'joint_label_below_endpoints':{k:int(v.sum()) for k,v in ejoint.items()},'joint_label_below_allopen':{k:int(v.sum()) for k,v in djoint.items()},'comparisons_vs_unchanged':{arm:paired(solved[arm],solved['unchanged']) for arm in ['pl','lp_scaled']},'n_unique':4096,'evaluation_jobs':48,'new_scientific_model_maze_evaluations':36864,'new_optimizer_updates':0,'rows':rows,'records':records,'model_state_checks':modelchecks,'independent_graph_checks':graph_checks,'exact_board_cache_hits':cache_hits,'all_ranking_pairs_independently_recomputed':73728,'maximum_independent_normalized_gap_error':maximum_error,'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'evidence_sha256':evidence,'input_audit':inputs,'worker_receipts':{str(g):json.loads((DEST/f'eval-worker{g}.json').read_text()) for g in [0,1]},'audit_seconds':time.perf_counter()-started,'scope':'Fixedpositive norm-matching of verifiedH2xLPprimals, onefloat32cast; sourcepositiveconstraints verified, independentfresh120cohort. Nativeinference only. Compared to frozenH2vBPAdamWPL and immutableoriginal. Differentobjectives/computebudgets, not optimizerfairness or globalenergy certificate.'}
    assert all(x['status']=='complete' for x in result['worker_receipts'].values());S.save(DEST/'audit.json',result);S.emit('lp_transfer_audit_complete',ranking=ranking,accuracy=accuracy,overall=overall,joint_success=result['joint_success'],joint_ranking=result['joint_label_below_both'],seconds=result['audit_seconds'])


if __name__=='__main__':main()
