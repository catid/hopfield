"""Portable H2bp scalar, exact-binomial, full-cost and final-state verification."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from audit_connectivity_screen import path_valid
from verify_hard_negative_feasibility import valid_shortest
from verify_training_feasibility import flip,LD


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cp(k,n,alpha):
    """Independent binomial-tail inversion; no SciPy or production statistics."""
    coefficients=[math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1) for i in range(n+1)]
    def tail(p,indices):
        if p==0:return float(0 in indices)
        if p==1:return float(n in indices)
        a,b=math.log(p),math.log1p(-p)
        return math.fsum(math.exp(coefficients[i]+i*a+(n-i)*b) for i in indices)
    def invert(indices,increasing):
        lo,hi=0.,1.
        for _ in range(70):
            mid=(lo+hi)/2;value=tail(mid,indices)
            if (value<alpha)==increasing:lo=mid
            else:hi=mid
        return (lo+hi)/2
    return (0. if k==0 else invert(range(k,n+1),True),1. if k==n else invert(range(k+1),False))


def close(actual,expected,tolerance):
    a,b=np.asarray(actual),np.asarray(expected)
    assert a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b))
    assert np.array_equal(np.isposinf(a),np.isposinf(b)) and np.array_equal(np.isneginf(a),np.isneginf(b))
    finite=np.isfinite(a);error=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
    assert error<tolerance
    return error


def costs(directory,ph,audit):
    marker_bytes=json.loads((directory/'barrier-markers.json').read_text())['markers'];values=[];events=0
    for pair in range(4):
        sums={}
        for mode in ['LP96','BP_geometry_LP96']:
            cohort=json.loads((directory/'runs'/f'pair{pair}-{mode}'/'result.json').read_text())
            assert cohort['model_inputs']==12288 and len(cohort['rows'])==384
            assert {(r['seed'],r['shard'],r['batch']) for r in cohort['rows']}=={(s,h,b) for s in range(3) for h in range(16) for b in range(8)}
            ready=cohort['model_readiness']
            assert {(r['seed'],r['model']) for r in ready}=={(s,m) for s in range(3) for m in (['lp'] if mode=='LP96' else ['bp','lp'])}
            assert abs(sum(r['seconds'] for r in ready)+cohort['readiness_orchestration_seconds']-cohort['model_readiness_seconds'])<1e-8
            assert cohort['readiness_orchestration_seconds']>=0
            barrier=cohort['host_barrier'];base=Path(barrier['directory']).relative_to('artifacts/research/20261114/work/bp_first_endpoint_shift');times={}
            for e in barrier['events']:
                times[e['stage']]=[]
                for side,gpu in [('own',barrier['gpu']),('peer',1-barrier['gpu'])]:
                    raw=marker_bytes[str(base/e[side+'_marker'])];assert hashlib.sha256(raw.encode()).hexdigest()==e[side+'_sha256']
                    marker=json.loads(raw);worker=json.loads((directory/f'timing-worker{gpu}.json').read_text())
                    assert marker['protocol_sha256']==ph and marker['stage']==e['stage'] and marker['gpu']==gpu and marker['pid']==worker['pid']
                    times[e['stage']].append(marker['arrived_monotonic'])
                assert 0<=e['wait_seconds']<=60.01;events+=1
            assert cohort['readiness_started_monotonic']>=max(times['readiness-begin'])
            assert cohort['readiness_finished_monotonic']<=times['readiness-done'][0]
            for i,row in enumerate(cohort['rows']):
                t=row['timing'];c=t['components'];required={'input_transfer','hidden_readout_and_flags','LP96_complete_prefix','LP96_answer_return','LP96_original_margin_and_return','final_numerical_acceptance'}
                required|={'model_provenance'} if mode=='LP96' else {'BP_native_solve','BP_cap_flags','BP_answer_return','BP_original_margin_and_return','geometry_screen_and_indices','LP_subset_gather','subset_scatter_and_provenance'}
                assert required<=c.keys() and all(np.isfinite(v) and v>=0 for v in c.values())
                assert abs(sum(c.values())+t['unattributed_in_segment_seconds']-t['answer_seconds'])<1e-8
                assert t['unattributed_in_segment_seconds']>=0 and row['archive_tasks_at_answer_start']==0
                assert t['answer_started_monotonic']>=max(times[f'batch{i:04d}-begin'])
                assert t['answer_finished_monotonic']<=times[f'batch{i:04d}-answer-done'][0]
                assert t['diagnostic_started_monotonic']>=max(times[f'batch{i:04d}-answer-done'])
            total=cohort['model_readiness_seconds']+sum(r['timing']['answer_seconds'] for r in cohort['rows'])
            assert abs(total-cohort['complete_answer_seconds'])<1e-8;sums[mode]=total
        values.append(Fraction(1)-Fraction(sums['BP_geometry_LP96'])/Fraction(sums['LP96']))
    ordered=sorted(values);median=(ordered[1]+ordered[2])/2
    assert str(median)==audit['cost_endpoint']['exact_median_fraction']
    assert audit['cost_endpoint']['decision']==('supported' if median>=Fraction(1,10) else 'falsified')
    return events


def verify(directory):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());a=json.loads((directory/'audit.json').read_text());seal=json.loads((directory/'sealed-replays.json').read_text());ph=sha(directory/'protocol.json')
    assert a['status']=='passed' and seal['status']=='complete' and a['protocol_sha256']==seal['protocol_sha256']==ph
    assert a['sealed_replays_sha256']==sha(directory/'sealed-replays.json') and not seal['labels_or_shortestness_outcomes_joined']
    for name,digest in {**seal['cohort_and_worker_receipt_sha256'],**seal['fresh_input_receipt_sha256']}.items():assert sha(directory/name)==digest
    assert a['primary_model_maze_arm_evaluations']==36864 and a['planned_jobs']==48 and a['fresh_independent_inputs']==4096 and a['accuracy_pair']==0
    with (directory/'per-model-input.csv').open() as stream:rows=list(csv.DictReader(stream))
    with (directory/'per-input.csv').open() as stream:joint_rows=list(csv.DictReader(stream))
    assert len(rows)==36864 and len(joint_rows)==4096
    assert sha(directory/'per-model-input.csv')==a['per_model_rows_sha256'] and sha(directory/'per-input.csv')==a['per_input_rows_sha256']
    bycase={(int(r['input_index']),int(r['model_seed']),r['arm']):r for r in rows};assert len(bycase)==36864
    arms=['BP_AdamW','LP96','cascade'];joint={arm:[] for arm in arms}
    for i in range(4096):
        assert int(joint_rows[i]['input_index'])==i
        for arm in arms:
            success=all(int(bycase[i,s,arm]['success']) for s in range(3));joint[arm].append(success)
            assert int(joint_rows[i][arm])==success
    for arm in arms:
        assert sum(joint[arm])==a['joint_success'][arm]
        selected=[r for r in rows if r['arm']==arm]
        assert sum(int(r['success']) for r in selected)==a['model_success'][arm]
        for row in selected:
            assert int(row['success'])==(int(row['graph_valid_shortest'])&int(row['numerically_accepted']))
            assert int(row['numerically_accepted'])<=int(row['hidden_good']) and int(row['numerically_accepted'])<=int(row['finite']) and int(row['numerically_accepted'])<=int(row['stopped'])
            if row['numerically_accepted']=='1':assert row['native_cap']==row['block_cap']=='0'
    successes=sum(joint['cascade']);gains=sum(c and not l for c,l in zip(joint['cascade'],joint['LP96']));losses=sum(l and not c for c,l in zip(joint['cascade'],joint['LP96']))
    lo,hi=cp(successes,4096,.05);gl,gu=cp(gains,4096,.025);ll,lu=cp(losses,4096,.025)
    absolute_decision='supported' if lo>=.999 else 'falsified' if hi<.999 else 'inconclusive'
    ni_decision='supported' if gl-lu>=-.001 else 'falsified' if gu-ll<-.001 else 'inconclusive'
    absolute=a['primary_endpoints']['absolute_joint_success_999'];ni=a['primary_endpoints']['paired_LP96_noninferiority']
    assert absolute['successes']==successes and absolute['n']==ni['n']==4096
    assert ni['gains']==gains and ni['losses']==losses
    close([lo,hi],[absolute['one_sided95_success_lower_pp']/100,absolute['one_sided95_success_upper_pp']/100],1e-10)
    close([gl-lu,gu-ll],[ni['separately_one_sided95_net_lower_pp']/100,ni['separately_one_sided95_net_upper_pp']/100],1e-10)
    assert a['primary_endpoints']['absolute_joint_success_999']['decision']==absolute_decision
    assert a['primary_endpoints']['paired_LP96_noninferiority']['decision']==ni_decision
    inputs=json.loads((directory/'independent-input-audit.json').read_text())
    distances=np.asarray(inputs['distances']).reshape(-1)
    assert distances.shape==(4096,) and np.all(distances>=1)
    groups=np.searchsorted(np.array([64,128,256]),distances,side='left')
    strata=json.loads((directory/'distance-strata.json').read_text())
    assert strata['status']=='complete' and strata['protocol_sha256']==ph and strata['audit_sha256']==sha(directory/'audit.json')
    names=['1..64','65..128','129..256','>=257']
    for group,record in enumerate(strata['bins']):
        assert record['shortest_edge_bin']==names[group]
        selected_indices=set(map(int,np.flatnonzero(groups==group)))
        assert record['empty']==(len(selected_indices)==0)
        for arm in arms:
            selected=[r for r in rows if r['arm']==arm and int(r['input_index']) in selected_indices]
            expected={'maze_denominator':len(selected_indices),'joint_successes':sum(joint[arm][i] for i in selected_indices),
                'model_input_denominator':len(selected),'model_input_successes':sum(int(r['success']) for r in selected),
                'numerical_rejections':sum(r['numerically_accepted']=='0' for r in selected),
                'routed':sum(int(r['routed']) for r in selected),'BP_geometry_valid_longer':sum(int(r['BP_geometry_valid_longer']) for r in selected)}
            assert record['arms'][arm]==expected
    assert len(strata['bins'])==4
    barrier_events=costs(directory,ph,a)
    d=directory/'state-fixtures';manifest=json.loads((d/'manifest.json').read_text())
    assert manifest['protocol_sha256']==ph and manifest['audit_sha256']==sha(directory/'audit.json')
    for name,digest in manifest['parameters_sha256'].items():
        assert sha(d/name)==digest and p['source_evidence_sha256']['artifacts/research/20261114/work/bp_first_endpoint_shift/'+name]==digest
    fixed={(h*256,s) for h in range(16) for s in range(3)}
    routed={(int(r['input_index']),int(r['model_seed'])) for r in rows if r['arm']=='cascade' and r['routed']=='1'}
    errors={(int(r['input_index']),int(r['model_seed'])) for r in rows if r['success']=='0'}
    longer={(int(r['input_index']),int(r['model_seed'])) for r in rows if r['BP_geometry_valid_longer']=='1'}
    assert {(r['input_index'],r['model_seed']) for r in manifest['records']}==fixed|routed|errors|longer
    assert len(routed)==manifest['complete_routed_cases'] and len(errors)==manifest['complete_any_arm_failure_cases'] and len(longer)==manifest['complete_BP_valid_longer_cases']
    timed_receipts={}
    for pair in range(4):
        for mode in ['LP96','BP_geometry_LP96']:
            cohort=json.loads((directory/'runs'/f'pair{pair}-{mode}'/'result.json').read_text())
            timed_receipts[pair,mode]={r['file']:r for r in cohort['rows']}
    checked=0;maximum=0.;labels=set()
    for record in manifest['records']:
        index,seed=record['input_index'],record['model_seed'];file=d/record['file'];assert sha(file)==record['sha256']
        with np.load(file,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        assert v['indices'].tolist()==[index]
        endpoints=v['sampled_endpoints'][0];old=v['original_endpoints'][0]
        assert endpoints[0]!=endpoints[1] and old[0]!=old[1]
        assert v['tokens'][0].flat[endpoints[0]]==2 and v['tokens'][0].flat[endpoints[1]]==3
        rng=np.random.default_rng(np.random.SeedSequence([p['cohort']['base_seed'],180,80,index,1]))
        assert np.array_equal(endpoints,rng.choice(np.flatnonzero(v['tokens'][0].ravel()>0),size=2,replace=False))
        assert all(v['tokens'][0].flat[i]==(2 if i==endpoints[0] else 3 if i==endpoints[1] else 1) for i in old)
        for prefix,mode in [('control','LP96'),('candidate','BP_geometry_LP96')]:
            provenance=json.loads(v[prefix+'_provenance_utf8'].tobytes());assert len(provenance)==4
            for pair,item in enumerate(provenance):
                file=Path(item['batch_file']);assert item['pair']==pair and file.parent.name==f'pair{pair}-{mode}'
                assert timed_receipts[pair,mode][file.name]['file_sha256']==item['batch_sha256']
        token_hash=hashlib.sha256(v['tokens'].tobytes()).hexdigest()
        if token_hash not in labels:
            assert valid_shortest(v['tokens'][0],v['labels'][0]);labels.add(token_hash)
        params=[]
        for arm in ['pl','old_binary']:
            with np.load(d/f'model-seed{seed}-{arm}.npz',allow_pickle=False) as z:params.append((z['theta'].astype(LD),z['mu_offsets'],z['aoffs']))
        cached={}
        for pair in range(4):
            geometry=path_valid(v['tokens'][0],v['candidate_bp_prediction'][pair])
            assert record['actually_routed']==(not geometry)
            for arm,prefix,model in [('BP_AdamW','candidate_bp_',0),('LP96','control_answer_',1),('cascade','candidate_answer_',int(not geometry))]:
                row=bycase[index,seed,arm];board=v[prefix+'prediction'][pair];key=(model,np.packbits(board).tobytes())
                if key not in cached:
                    delta=flip(v['tokens'],board[None],v['couplings'].astype(LD),*params[model]);margin=np.where(v['tokens']==1,delta,np.inf).min()
                    graph=path_valid(v['tokens'][0],board) and int(board.sum())==int(v['shortest_edges'][0])+1
                    cached[key]=(margin,graph)
                margin,graph=cached[key];maximum=max(maximum,close(margin,v[prefix+'own_margin'][pair],1e-9))
                for name in ['finite','stopped','native_cap','block_cap']:assert int(v[prefix+name][pair])==int(row[name])
                hidden=v['candidate_answer_hidden_good'][pair]
                assert int(hidden)==int(row['hidden_good'])==int(v['control_answer_hidden_good'][pair])
                clamps=((v['tokens'][0]==1)|(board==(v['tokens'][0]>=2))).all()
                accepted=hidden and v[prefix+'finite'][pair] and v[prefix+'stopped'][pair] and clamps and not v[prefix+'native_cap'][pair] and not v[prefix+'block_cap'][pair] and np.isfinite(margin) and margin>=-1e-6
                assert int(row['graph_valid_shortest'])==int(graph) and int(row['numerically_accepted'])==int(accepted)
                assert int(row['success'])==int(graph and accepted)
                assert np.array_equal(board,v[prefix+'prediction'][0])
                if arm!='BP_AdamW':assert int(v[prefix+'chosen_model'][pair])==model
                if arm=='cascade':
                    expected=v['control_answer_prediction'][pair] if model else v['candidate_bp_prediction'][pair]
                    assert np.array_equal(board,expected)
                checked+=1
    result={'status':'passed','protocol_sha256':ph,'scalar_rows':len(rows),'joint_inputs':4096,'descriptive_distance_bins_checked':4,'fixture_cases':len(manifest['records']),
        'actual_final_states_checked':checked,'maximum_original_margin_error':maximum,'measured_barrier_events':barrier_events,
        'independent_absolute_interval':[lo,hi],'independent_paired_interval':[gl-lu,gu-ll],
        'absolute_decision':absolute_decision,'paired_decision':ni_decision,'seconds':time.perf_counter()-started,'checker_sha256':sha(Path(__file__)),
        'scope':'All primary scalar outcomes and exact-binomial decisions, all measured complete costs, and independently reconstructed final margins/graphs for every route, every arm error, valid-longer and fixed-control supplement. Full internal trajectory proof remains the separately sealed audit.'}
    (directory/'public-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);verify(parser.parse_args().directory)
