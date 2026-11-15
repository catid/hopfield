"""H2br charged token dispatch with actual BP/LP subsets; controls unchanged."""
import time
import numpy as np
import study as S
import larger_relaxation as L
from factorial_transfer import hidden
from bp_first_lp96_core import solve as lp96
from bp_first_geometry_policy import routes
from bp_first_fresh_batch import run as unchanged
from endpoint_dispatch_policy import partition,lp_indices,assemble


def run(mode,host_tokens,models,doubles,watchdog=None,after_answer=None):
    if mode in ['LP96','BP_geometry_LP96']:
        return unchanged(mode,host_tokens,models,doubles,watchdog,after_answer)
    if mode!='endpoint_dispatch':raise ValueError('Unknown arm')
    torch,_,_=S.imports();torch.cuda.synchronize();answer_started=time.perf_counter();components={}
    def start():torch.cuda.synchronize();return time.perf_counter()
    def finish(name,began):torch.cuda.synchronize();components[name]=time.perf_counter()-began
    began=start();distance,high,bi,low=partition(host_tokens);finish('endpoint_extraction_and_dispatch',began)
    began=start();t=torch.tensor(host_tokens.astype(np.int64),device='cuda');finish('input_transfer',began)
    began=start();a,hs,good=hidden(models['bp'],t);hidden_good=good.cpu().numpy();finish('hidden_readout_and_flags',began)
    began=start();bix=torch.tensor(bi,device=t.device);bt,ba=t[bix],a[bix];finish('BP_subset_gather',began)
    def cpu_final(out,caps,model,tt,aa,prefix):
        began=start();cpu={name:(out[name].bool() if name=='prediction' else out[name]).cpu().numpy() for name in ['prediction','iterations','stopped','finite']}
        cpu['native_cap']=caps['native_cap'].copy();cpu['block_cap']=caps['block_cap'].copy();finish(prefix+'_answer_return',began)
        began=start();cpu['own_margin']=model.flip_delta(tt,out['prediction'].double(),aa.double()).masked_fill(tt!=1,float('inf')).flatten(1).amin(1).cpu().numpy();finish(prefix+'_original_margin_and_return',began)
        return cpu
    bp_native=None
    if len(bi):
        began=start();bp_native=L.solve(models['bp'],bt,ba,torch.full_like(ba[:,0],3.),8192,0.);finish('BP_native_solve',began)
        out={k:bp_native[i] for i,k in enumerate(['prediction','iterations','stopped','finite'])}
        began=start();caps={'native_cap':(~bp_native[2]&bp_native[3]).cpu().numpy(),'block_cap':np.zeros(len(bi),bool)};finish('BP_cap_flags',began)
        bp=cpu_final(out,caps,doubles['bp'],bt,ba,'BP')
    else:
        began=start();bp={'prediction':np.zeros((0,180,180),bool),'iterations':np.zeros(0,np.int64),'stopped':np.zeros(0,bool),'finite':np.zeros(0,bool),'own_margin':np.zeros(0,np.float64),'native_cap':np.zeros(0,bool),'block_cap':np.zeros(0,bool)};finish('empty_BP_receipt_arrays',began)
        components.update(BP_native_solve=0.,BP_cap_flags=0.,BP_answer_return=0.,BP_original_margin_and_return=0.)
    began=start();geometry,_=routes(host_tokens[bi],bp['prediction']);li=lp_indices(len(t),bi,geometry);finish('geometry_screen_and_indices',began)
    began=start();lix=torch.tensor(li,device=t.device);lt,la=t[lix],a[lix];finish('LP_subset_gather',began)
    core=None
    if len(li):
        began=start();core=lp96(doubles['lp'],lt,la,watchdog);finish('LP96_complete_prefix',began)
        lp=cpu_final(core['outputs'][7],core['blocks'][7],doubles['lp'],lt,la,'LP96')
    else:
        components.update(LP96_complete_prefix=0.,LP96_answer_return=0.,LP96_original_margin_and_return=0.)
        began=start();lp={name:value[:0].copy() for name,value in bp.items()};finish('empty_subset_receipt_arrays',began)
    began=start();final=assemble(len(t),bp,lp,bi,li);finish('subset_scatter_and_provenance',began)
    began=start();clamps=((host_tokens==1)|(final['prediction']==(host_tokens>=2))).all((1,2))
    final['accepted']=(hidden_good&final['finite']&final['stopped']&clamps&~final['native_cap']&~final['block_cap']&np.isfinite(final['own_margin'])&(final['own_margin']>=-1e-6))
    final['hidden_good']=hidden_good;final['clamps_ok']=clamps;finish('final_numerical_acceptance',began)
    torch.cuda.synchronize();answer_seconds=time.perf_counter()-answer_started;summed=sum(components.values());assert summed<=answer_seconds+1e-9
    barrier_wait=after_answer() if after_answer is not None else 0.;diagnostic_started=time.perf_counter()
    audit={'couplings':a.cpu().numpy(),'hidden_steps':hs.cpu().numpy(),'endpoint_manhattan':distance,'dispatch_high':high,'bp_indices':bi}
    if bp_native is not None:
        audit['bp_final_logits']=bp_native[4].cpu().numpy()
        audit['bp_minimum_flip_margin32']=models['bp'].flip_delta(bt,bp_native[0],ba).masked_fill(bt!=1,float('inf')).flatten(1).amin(1).cpu().numpy()
        audit['bp_energy64']=doubles['bp'].energy(bt,bp_native[0].double(),ba.double()).cpu().numpy()
    if core is not None:
        for name in ['prediction','iterations','stopped','finite']:audit['LP_'+name]=np.stack([core['outputs'][v][name].cpu().numpy() for v in range(1,8)])
        for name in ['block_moves','post_block_sweeps','native_cap','block_cap','initial_simple_path']:audit['LP_'+name]=np.stack([core['blocks'][v][name] for v in range(2,8)])
        audit['LP_binary_flip_count']=core['prefix']['flip_count'].cpu().numpy();audit['LP_binary_cache_margin']=core['prefix']['minimum_flip_margin'].cpu().numpy()
        margins,energies=[],[]
        for v in range(1,8):
            board=core['outputs'][v]['prediction'].double();margins.append(doubles['lp'].flip_delta(lt,board,la.double()).masked_fill(lt!=1,float('inf')).flatten(1).amin(1).cpu().numpy());energies.append(doubles['lp'].energy(lt,board,la.double()).cpu().numpy())
        audit['LP_minimum_flip_margin64']=np.stack(margins);audit['LP_energy64']=np.stack(energies)
    torch.cuda.synchronize();diagnostic_seconds=time.perf_counter()-diagnostic_started
    return {'answer':final,'bp_prefix':bp,'lp_subset':lp,'geometry_valid':geometry,'subset_indices':li,'core':core,'audit':audit,
        'timing':{'mode':mode,'answer_seconds':answer_seconds,'components':components,'unattributed_in_segment_seconds':answer_seconds-summed,
            'answer_started_monotonic':answer_started,'answer_finished_monotonic':answer_started+answer_seconds,'post_answer_barrier_wait_seconds':barrier_wait,'diagnostic_started_monotonic':diagnostic_started,'diagnostic_seconds':diagnostic_seconds,
            'actual_subset_inputs':len(li),'actual_BP_inputs':len(bi),'direct_LP_inputs':len(low),'rescue_LP_inputs':int((~geometry).sum()),'n':len(t),'all_required_operations_inside_answer_segment':True}}
