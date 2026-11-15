"""H2bk unchanged eight-state prefix plus charged numerical-only BP fallback."""
import time
import numpy as np
import study as S
import confirmation96_batch as Q
import larger_relaxation as L
from certificate_fallback_policy import accepted,chosen_model

HYBRID_KEYS=['prediction','iterations','stopped','finite','lp_accepted','chosen_model',
             'source_lp_margin','output_own_margin','output_lp_margin','accepted_main','accepted_dual']


def solve(models,doubles,t,a,hidden_good,rotation=0,watchdog=None):
    torch,_,_=S.imports();r=Q.solve(models,doubles,t,a,rotation,watchdog);out=r['outputs'];block=r['blocks'];n=len(t);costs={}
    def begin():torch.cuda.synchronize();return time.perf_counter()
    def end(name,start):torch.cuda.synchronize();costs[name]=time.perf_counter()-start
    before=begin();lp_margin=doubles[1].flip_delta(t,out[7]['prediction'].double(),a.double()).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy();hg=hidden_good.cpu().numpy();end('lp_routing_certificate',before)
    before=begin();cpu=out[7]['cpu_output'];lp_accept=accepted(hg,cpu['finite'],cpu['stopped'],block[7]['native_cap'],block[7]['block_cap'],lp_margin);indices=np.flatnonzero(~lp_accept);selected=chosen_model(lp_accept);end('routing_selection',before)
    subset={'indices':indices,'prediction':np.zeros((0,*t.shape[1:]),bool),'iterations':np.zeros(0,np.int64),
            'stopped':np.zeros(0,bool),'finite':np.zeros(0,bool),'final_logits':np.zeros((0,*t.shape[1:]),np.float32),
            'own_margin':np.zeros(0),'lp_margin':np.zeros(0),'own_energy':np.zeros(0),'lp_energy':np.zeros(0),
            'minimum_margin32':np.zeros(0,np.float32)}
    for name in ['fallback_gather','bp_subset_solve','bp_subset_cpu_return','bp_own_certificate','lp_cross_certificate']:costs[name]=0.
    if len(indices):
        before=begin();ix=torch.tensor(indices,device=t.device);tt=t[ix];aa=a[ix];u=torch.full_like(aa[:,0],3.);end('fallback_gather',before)
        before=begin();z=L.solve(models[0],tt,aa,u,8192,0.);end('bp_subset_solve',before)
        before=begin();subset.update(prediction=z[0].bool().cpu().numpy(),iterations=z[1].cpu().numpy(),stopped=z[2].cpu().numpy(),finite=z[3].cpu().numpy());end('bp_subset_cpu_return',before)
        before=begin();subset['own_margin']=doubles[0].flip_delta(tt,z[0].double(),aa.double()).masked_fill(tt!=1,float('inf')).flatten(1).amin(1).cpu().numpy();end('bp_own_certificate',before)
    before=begin();hybrid={k:cpu[k].copy() for k in ['prediction','iterations','stopped','finite']};own_margin=lp_margin.copy();cross_margin=lp_margin.copy()
    if len(indices):
        for k in ['prediction','iterations','stopped','finite']:hybrid[k][indices]=subset[k]
        own_margin[indices]=subset['own_margin']
    main=hg&hybrid['finite']&hybrid['stopped']&np.isfinite(own_margin)&(own_margin>=-1e-6)
    # The un-routed side also carries its cap conditions through lp_accept.
    main[lp_accept]=True
    hybrid.update(lp_accepted=lp_accept,chosen_model=selected,source_lp_margin=lp_margin,output_own_margin=own_margin,accepted_main=main)
    end('hybrid_mapping_and_flags',before)
    if len(indices):
        before=begin();subset['lp_margin']=doubles[1].flip_delta(tt,z[0].double(),aa.double()).masked_fill(tt!=1,float('inf')).flatten(1).amin(1).cpu().numpy();end('lp_cross_certificate',before)
    before=begin()
    if len(indices):cross_margin[indices]=subset['lp_margin']
    hybrid['output_lp_margin']=cross_margin;hybrid['accepted_dual']=main&np.isfinite(cross_margin)&(cross_margin>=-1e-6);end('strict_certificate_flags',before)
    # Extra logits and energies are archival diagnostics, outside charged answer time.
    before=begin()
    if len(indices):
        subset['final_logits']=z[4].cpu().numpy()
        subset['own_energy']=doubles[0].energy(tt,z[0].double(),aa.double()).cpu().numpy()
        subset['lp_energy']=doubles[1].energy(tt,z[0].double(),aa.double()).cpu().numpy()
        subset['minimum_margin32']=models[0].flip_delta(tt,z[0],aa).masked_fill(tt!=1,float('inf')).flatten(1).amin(1).cpu().numpy()
    torch.cuda.synchronize();diagnostic_seconds=time.perf_counter()-before
    main_added=sum(v for k,v in costs.items() if k not in ['lp_cross_certificate','strict_certificate_flags']);strict_added=sum(costs.values())
    control={name:out[0]['cpu_output'][name][indices] for name in ['prediction','iterations','stopped','finite']}
    parity={name:bool(np.array_equal(subset[name],control[name])) for name in control}
    if len(indices):parity['final_logits']=bool(np.array_equal(subset['final_logits'],out[0]['final_logits'][torch.tensor(indices,device=t.device)].cpu().numpy()))
    else:parity['final_logits']=True
    r.update(hybrid=hybrid,subset=subset,hybrid_timing={'operations':costs,'main_added_seconds':main_added,
        'strict_added_seconds':strict_added,'main_charged_seconds':r['timing']['charged_answer_seconds'][7]+main_added,
        'strict_charged_seconds':r['timing']['charged_answer_seconds'][7]+strict_added,
        'actual_subset_inputs':len(indices),'subset_vs_full_control_parity':parity,
        'subset_diagnostic_seconds':diagnostic_seconds})
    if watchdog:watchdog()
    return r
