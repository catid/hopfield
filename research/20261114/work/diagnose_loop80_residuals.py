"""Read-only cap geometry/progress and BP cross-objective certificates; no trajectories."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
import time
import numpy as np
import study as S
from verify_large_cycle_geometry import geometry
from verify_cycle_cut_geometry import cycle_rank
from verify_training_feasibility import flip,LD
from local_factor_delta import deltas

SOURCE=S.OUT/'loop80_transfer';DEST=S.OUT/'loop80_residual_diagnosis'


def check_job(args):
    job,audit_row,ds=args;started=time.perf_counter();run=SOURCE/'runs'/job['id'];receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256']
    with np.load(run/'outputs.npz') as z:
        cap=z['block_cap'][3];assert np.array_equal(cap,~np.array(audit_row['success'][5],bool));ix=np.flatnonzero(cap);base={'job':job['id'],'source_outputs_sha256':receipt['outputs_sha256'],'caps':len(ix),'cases':[]}
        if not len(ix):return base
        boards=z['predictions'][:,ix];a=z['couplings'][ix].astype(LD);margins=z['minimum_flip_margin64'][5,ix];native=z['block_native_cap'][3,ix];finite=z['finite'][5,ix];moves=z['block_moves'][:,ix];veto_lineage=z['block_cap'][0,ix]
    assert finite.all() and not native.any() and (margins>=-1e-8).all();assert S.sha(S.ROOT/ds['path'])==ds['sha256']
    with np.load(S.ROOT/ds['path']) as z:t=z['tokens'][ix]
    model=SOURCE/f'model-seed{job["seed"]}-old_binary.npz';ex=json.loads((SOURCE/'model-extraction.json').read_text());ref=next(r for r in ex['records'] if r['seed']==job['seed'] and r['arm']=='old_binary');assert S.sha(model)==ref['sha256']
    with np.load(model) as z:theta=z['theta'].astype(LD);mu=z['mu_offsets'];ao=z['aoffs']
    bp_margins=np.where(t==1,flip(t,boards[0],a,theta,mu,ao),np.inf).reshape(len(ix),-1).min(1);progress={int(i):[] for i in ix}
    for item in receipt['trace_archives']:
        arm=['pl','old_binary','connectivity_veto','original_lexical16','unified16','unified48'].index(item['arm']);lo=item['offset'];selected=[int(i) for i in ix if lo<=i<lo+32]
        if arm<3 or not selected:continue
        path=run/item['path'];assert S.sha(path)==item['sha256']
        with np.load(path) as z:local=z['valid_prefix_indices'];payload=json.loads(z['trace_utf8'].tobytes())
        for index in selected:
            hit=np.flatnonzero(local==index-lo)
            if not len(hit):continue
            owner=int(hit[0]);ranks=[];sizes=[];costs=[];sweeps=[]
            for record in payload['trace']:
                g=next((g for g in record['geometry'] if g['local_input']==owner),None)
                if g is None:continue
                ranks.append(g['cycle_rank']);chosen=[record['candidates'][j] for j in record['chosen'] if record['candidates'][j]['local_input']==owner];sizes.extend(len(c['cells']) for c in chosen);costs.extend(c['delta'] for c in chosen);sweeps.extend(n['sweeps'] for n in record.get('native_results',[]) if n['local_input']==owner)
            progress[index].append({'stage':item['arm'],'cycle_rank_before_round':ranks,'chosen_chain_sizes':sizes,'chosen_original_costs':costs,'native_sweeps':sweeps})
    for j,index in enumerate(ix):
        c256,g256=geometry(t[j],boards[5,j],256);c512,g512=geometry(t[j],boards[5,j],512);initial=cycle_rank(boards[2,j]);final=cycle_rank(boards[5,j]);total=int(moves[1:,j].sum());assert sum(len(r['chosen_chain_sizes']) for r in progress[int(index)])==total;assert initial-final>=total
        union=sorted(set(c256)|set(c512));values=deltas(t[j:j+1],boards[5,j:j+1],a[j:j+1],theta,mu,ao,[(0,c) for c in union]);costs={c:float(v) for c,v in zip(union,values)};bp_success=bool(audit_row['success'][0][int(index)]);case={'input_index':job['shard']*256+int(index),'local_input':int(index),'model_seed':job['seed'],'veto_cap_lineage':bool(veto_lineage[j]),'original_lp_margin':float(margins[j]),'total_cycle_moves':total,'cycle_rank_before_cycles':initial,'cycle_rank_at_cap':final,'native_extra_cycle_removals':initial-final-total,'candidates256':len(c256),'candidates512':len(c512),'candidate_families_equal':c256==c512,'maximum_candidate_size256':max(map(len,c256),default=0),'maximum_candidate_size512':max(map(len,c512),default=0),'oversize_components256':g256['oversize_components'],'oversize_components512':g512['oversize_components'],'lexical_cost256':costs[c256[0]] if c256 else None,'minimum_cost256':min((costs[c] for c in c256),default=None),'bp_success_under_bp_certificate':bp_success,'bp_margin_under_lp_objective':float(bp_margins[j]),'bp_has_lp_margin_certificate':bool(bp_margins[j]>=-1e-6),'progress':progress[int(index)]};base['cases'].append(case)
    base['seconds']=time.perf_counter()-started;return base


def main():
    started=time.perf_counter();DEST.mkdir(exist_ok=True);partial=DEST/'partial-rows';partial.mkdir(exist_ok=True);p=json.loads((SOURCE/'protocol.json').read_text());a=json.loads((SOURCE/'audit.json').read_text());assert a['status']=='passed';byjob={r['job']['id']:r for r in a['rows']};data=json.loads((SOURCE/'datasets.json').read_text());rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(check_job,(j,byjob[j['id']],data['datasets'][j['shard']])) for j in p['jobs']]):
            row=future.result();rows.append(row);path=partial/f'{row["job"]}.json';tmp=path.with_name(path.name+f'.tmp-{os.getpid()}');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(path);assert time.perf_counter()-started<120
    cases=[c for r in sorted(rows,key=lambda r:r['job']) for c in r['cases']];assert len(cases)==148;bp_correct=[c for c in cases if c['bp_success_under_bp_certificate']];summary={'status':'passed','source_protocol_sha256':S.sha(SOURCE/'protocol.json'),'source_audit_sha256':S.sha(SOURCE/'audit.json'),'source_sha256':S.sha(__file__),'source_cases':148,'source_joint_inputs':len({c['input_index'] for c in cases}),'cap_flags_exactly_match_all_lp_failures':True,'cycle_move_counts':sorted(set(c['total_cycle_moves'] for c in cases)),'final_cycle_rank_range':[min(c['cycle_rank_at_cap'] for c in cases),max(c['cycle_rank_at_cap'] for c in cases)],'final_cycle_rank_histogram':{str(k):sum(c['cycle_rank_at_cap']==k for c in cases) for k in sorted({c['cycle_rank_at_cap'] for c in cases})},'native_extra_cycle_removals':sum(c['native_extra_cycle_removals'] for c in cases),'all_have_candidate256':all(c['candidates256']>0 for c in cases),'candidate_families_equal_cases':sum(c['candidate_families_equal'] for c in cases),'maximum_candidate_size256':max(c['maximum_candidate_size256'] for c in cases),'maximum_candidate_size512':max(c['maximum_candidate_size512'] for c in cases),'bp_correct_on_lp_caps':len(bp_correct),'bp_correct_also_lp_margin_certified':sum(c['bp_has_lp_margin_certificate'] for c in bp_correct),'minimum_lp_margin_on_bp_correct_cap_answers':min(c['bp_margin_under_lp_objective'] for c in bp_correct),'cases':cases,'processes':8,'seconds':time.perf_counter()-started,'new_trajectories':0,'changed_boards_or_flags':0,'scope':'Read-only source diagnosis. Current candidate-family equality does not establish future equality after more deletions. Cycle-rank progress and BP cross-objective margins are measured on archived states only; no claimed rescue beyond existing audited unions.'};assert summary['seconds']<120;S.save(DEST/'geometry-progress.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
