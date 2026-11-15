"""Independent H2ae graph/flag/iteration audit and paired primary endpoints."""
import csv
import json
import time
import numpy as np
import study as S
import audit as A
from audit_restart import check_path
from second_negative_transfer import PP,DEST,ARMS,protocol


def main():
    started=time.perf_counter();p=protocol();evidence={};rows=[];cases=[];saved={};records=[];cache={};graph_checks=cache_hits=0;maximum_cache_error=0.
    def verify(path,expected=None):
        h=S.sha(path)
        if expected is not None:assert h==expected,path
        evidence[str(path.relative_to(S.ROOT))]=h
    verify(PP)
    for path,h in p['gates'].items():verify(S.ROOT/path,h)
    for group in p['models'].values():
        for item in group.values():verify(S.ROOT/item['path'],item['sha256'])
    inputs=json.loads((DEST/'independent-input-audit.json').read_text());assert inputs['status']=='passed' and inputs['inputs_checked']==4096 and inputs['protocol_sha256']==S.sha(PP);verify(DEST/'independent-input-audit.json')
    manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['datasets']==inputs['datasets'];data={}
    for ds in manifest['datasets']:
        verify(S.ROOT/ds['path'],ds['sha256'])
        with np.load(S.ROOT/ds['path']) as x:data[ds['shard']]={k:x[k] for k in x.files}
    for job in p['jobs']:
        run=DEST/'runs'/job['id'];wait_start=time.perf_counter()
        while not (run/'result.json').exists():
            if time.perf_counter()-wait_start>900:raise TimeoutError('Result not available in900s')
            time.sleep(.2)
        r=json.loads((run/'result.json').read_text());verify(run/'result.json');assert r['status']=='complete' and r['job']==job and r['protocol_sha256']==S.sha(PP);verify(run/'outputs.npz',r['outputs_sha256'])
        assert r['checkpoint_sha256_by_arm']==[p['models'][str(job['seed'])][arm]['sha256'] for arm in ARMS]
        with np.load(run/'outputs.npz') as x:out={k:x[k] for k in x.files if k!='couplings'}
        d=data[job['shard']];pred=out['predictions'];logits=out['native_final_logits'];assert pred.shape==(3,256,120,120) and logits.shape==(1,256,120,120)
        assert np.array_equal(pred[:1],logits>0) and np.array_equal(out['finite'][:1],np.isfinite(logits).all((2,3)))
        assert np.array_equal(out['stopped'][:1],out['finite'][:1]&(out['native_minimum_flip_margin32']>=-1e-6))
        assert np.all(out['iterations'][:1]%16==0) and np.all((out['iterations'][:1]>=16)&(out['iterations'][:1]<=8192)) and np.all(out['iterations'][:1][~out['stopped'][:1]]==8192)
        assert np.all((out['iterations'][1:]>=1)&(out['iterations'][1:]<=512)) and np.all(out['iterations'][1:][~out['stopped'][1:]]==512)
        assert np.all(out['binary_cache_margin'][out['stopped'][1:]]>=-1e-8) and np.all(out['binary_flip_count']>=0)
        error=float(abs(out['binary_cache_margin']-out['minimum_flip_margin64'][1:]).max());assert error<1e-10;maximum_cache_error=max(maximum_cache_error,error)
        for v in [1,2]:assert np.array_equal(pred[v][d['tokens']!=1],(d['tokens']>0)[d['tokens']!=1])
        valid=np.zeros((3,256),bool);shortest=np.zeros_like(valid)
        for v in range(3):
            for i,(tok,board) in enumerate(zip(d['tokens'],pred[v])):
                key=(int(d['indices'][i]),np.packbits(board).tobytes())
                if key not in cache:cache[key]=check_path(tok,board);graph_checks+=1
                else:cache_hits+=1
                valid[v,i]=cache[key];shortest[v,i]=valid[v,i] and int(board.sum())==int(d['shortest_edges'][i])+1
        certified=(out['minimum_flip_margin64']>=-1e-6)&out['stopped']&out['finite'];success=shortest&certified&out['hidden_good'][None]&np.isfinite(out['energy64']);saved[(job['shard'],job['seed'])]=success
        for v,arm in enumerate(ARMS):
            ep=(pred[v]==(d['tokens']>=2)).all((1,2));dense=(pred[v]==(d['tokens']>0)).all((1,2));unit='sweeps' if v>0 else 'updates'
            rows.append({'job':job['id'],'shard':job['shard'],'model_seed':job['seed'],'arm':arm,'n':256,'solved':int(success[v].sum()),'valid_paths':int(valid[v].sum()),'wrong_certified':int((certified[v]&~shortest[v]).sum()),'timeouts':int((~out['stopped'][v]&out['finite'][v]).sum()),'nonfinite':int((~out['finite'][v]).sum()),'hidden_failures':int((~out['hidden_good']).sum()),'endpoint_only_outputs':int(ep.sum()),'allopen_outputs':int(dense.sum()),'iteration_unit':unit,'mean_iterations':float(out['iterations'][v].mean()),'answer_seconds':r['answer_seconds_by_arm'][v],'binary_cache_setup_seconds_included':r['binary_cache_setup_seconds_included'][v],'minimum_flip_margin64':float(out['minimum_flip_margin64'][v].min())})
            for i,index in enumerate(d['indices']):cases.append({'input_index':int(index),'model_seed':job['seed'],'arm':arm,'success':int(success[v,i]),'valid_path':int(valid[v,i]),'shortest_path':int(shortest[v,i]),'iterations':int(out['iterations'][v,i]),'iteration_unit':unit,'stopped':int(out['stopped'][v,i]),'certified64':int(certified[v,i]),'finite':int(out['finite'][v,i]),'minimum_flip_margin64':float(out['minimum_flip_margin64'][v,i]),'energy64':float(out['energy64'][v,i]),'selected_cells':int(pred[v,i].sum()),'shortest_cells':int(d['shortest_edges'][i])+1,'endpoint_only':int(ep[i]),'allopen':int(dense[i]),'binary_flip_count':int(out['binary_flip_count'][v-1,i]) if v>0 else 0})
        records.append(r);S.emit('second_negative_transfer_job_audited',job=job['id'])
    joint={arm:np.concatenate([np.stack([saved[(shard,s)][v] for s in range(3)]).all(0) for shard in range(16)]) for v,arm in enumerate(ARMS)}
    def paired(a,b):
        gain=int((a&~b).sum());loss=int((~a&b).sum());gl,gu=A.bounds(gain,4096,alpha=.025);ll,lu=A.bounds(loss,4096,alpha=.025)
        return {'gains':gain,'losses':loss,'observed_net_gain_pp':100*(gain-loss)/4096,'separately_one_sided95_net_lower_pp':100*(gl-lu),'separately_one_sided95_net_upper_pp':100*(gu-ll)}
    gain=paired(joint['second_negative_binary'],joint['hard_negative_binary']);gain.update(target_pp=1.,decision='supported' if gain['separately_one_sided95_net_lower_pp']>=1 else 'falsified' if gain['separately_one_sided95_net_upper_pp']<1 else 'inconclusive')
    ni=paired(joint['second_negative_binary'],joint['pl']);ll,lu=A.bounds(ni['losses'],4096,alpha=.05);ni.update(loss_one_sided95_lower_pp=100*ll,loss_one_sided95_upper_pp=100*lu,margin_pp=.1,decision='supported' if 100*lu<=.1 else 'falsified' if 100*ll>.1 else 'inconclusive');overall='supported' if gain['decision']==ni['decision']=='supported' else 'falsified' if 'falsified' in [gain['decision'],ni['decision']] else 'inconclusive'
    perinput=[{'input_index':i,**{f'joint_{arm}_success':int(joint[arm][i]) for arm in ARMS}} for i in range(4096)]
    for name,values in [('runs.csv',rows),('per-model-input.csv',cases),('per-input.csv',perinput)]:
        with (DEST/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    result={'status':'validated','utc':S.utc(),'hypothesis':'H2ae','gain_primary':gain,'noninferiority_primary':ni,'overall_decision':overall,'joint_success':{k:int(v.sum()) for k,v in joint.items()},'n_unique':4096,'evaluation_jobs':48,'new_scientific_model_maze_evaluations':36864,'new_optimizer_updates':0,'rows':rows,'records':records,'independent_graph_checks':graph_checks,'exact_board_cache_hits':cache_hits,'maximum_cache_vs_upstream_float64_margin_error':maximum_cache_error,'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'evidence_sha256':evidence,'input_audit':inputs,'worker_receipts':{str(g):json.loads((DEST/f'eval-worker{g}.json').read_text()) for g in [0,1]},'audit_elapsed_seconds_including_waits':time.perf_counter()-started,'scope':'Fixed H2ad additional source unit-margin coefficient fit transferred with unchanged float64 four-color binary decoder; H2ac binary and BP PL native controls on fresh inputs. Same original-energy form, no graph repair, target fitting or rescue.'}
    assert all(x['status']=='complete' for x in result['worker_receipts'].values());S.save(DEST/'audit.json',result);S.emit('second_negative_transfer_audit_complete',gain=gain,noninferiority=ni,overall=overall,joint_success=result['joint_success'])


if __name__=='__main__':main()
