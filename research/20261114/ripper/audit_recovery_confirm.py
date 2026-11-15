"""Independent256-bank construction-only recovery confirmation."""
import math
import ast
from pathlib import Path
from bootstrap_two_way import resample_means
from recovery_confirm_common import load_protocol
import csv
import datetime
import hashlib
import json
import numpy as np
import torch
from run import ROOT,HERE,save,sha,now
from audit_clues import validity

OUT=ROOT/'artifacts/research/20261114/ripper/recovery_confirm'
PUBLIC=ROOT/'results/research/ripper'
PROTOCOL=HERE/'protocol-recovery-confirm.json'


def main():
    p=load_protocol();registration=json.loads((ROOT/p['input_registration']).read_text())
    assert registration['common_sha256']==sha(HERE/'recovery_confirm_common.py')
    assert registration['acquisition_sha256']==sha(HERE/'recovery_confirm_acquire.py')
    manifest=json.loads((ROOT/p['inputs']).read_text())
    assert sha(ROOT/p['inputs'])==p['input_manifest_sha256']
    assert sha(HERE/'exponential_policy.py')==p['frozen_policy_sha256']
    assert sha(HERE/'online_policy.py')==p['original_policy_sha256']
    assert sha(HERE/'spike_policy.py')==p['mixture_policy_sha256']
    preflight_path=ROOT/p['policy_preflight_path'];assert sha(preflight_path)==p['policy_preflight_sha256']
    preflight=json.loads(preflight_path.read_text())
    assert preflight['status']=='validated' and preflight['exponential_policy_sha256']==p['frozen_policy_sha256']
    assert preflight['original_policy_sha256']==p['original_policy_sha256']
    assert preflight['mixture_policy_sha256']==p['mixture_policy_sha256']
    assert preflight['wrapper_sha256']==sha(HERE/'test_exponential_policy.py')
    assert preflight['exact_rational_score_checks']==301 and preflight['initial_tie_checks']==64
    assert preflight['unchanged_control_checks']==128 and preflight['read_only_purity_and_determinism_checks']==64
    input_audit=json.loads((OUT/'input_audit.json').read_text())
    assert input_audit['status']=='validated' and input_audit['input_manifest_sha256']==p['input_manifest_sha256']
    assert input_audit['input_protocol_sha256']==p['input_protocol_sha256']
    assert input_audit['wrapper_sha256']==sha(HERE/'audit_recovery_confirm_inputs.py')
    assert input_audit['unique_source_rows']==input_audit['independent_csv_answers_verified']==manifest['fresh_unique_rows']
    assert input_audit['within_bank_overlap']==input_audit['prior_bank_overlap']==0
    pair=np.triu_indices(81,1);i,j=pair
    peer=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3))
    assert peer.sum()==810
    structure=[];supports={};hashes={};audits=[];timings=[];completion={}
    for seed in p['seeds']:
        d=OUT/f's{seed}';r=json.loads((d/'receipt.json').read_text());a=json.loads((d/'acquisition_audit.json').read_text())
        assert r['protocol_sha256']==a['protocol_sha256']==sha(PROTOCOL)
        assert r['wrapper_sha256']==sha(HERE/'recovery_confirm_acquire.py') and r['policy_sha256']==sha(HERE/'exponential_policy.py')
        assert a['wrapper_sha256']==sha(HERE/'audit_recovery_confirm_acquisition.py')
        assert a['acquisition_receipt_sha256']==sha(d/'receipt.json') and a['status']=='validated'
        assert a['independent_masks_replayed']==1800 and a['prefix_supports_verified']==3 and a['hidden_value_recount_checks']==6
        assert r['current_row_permutation_invariance_checks']==r['premature_reveal_rejections']==1800
        audits.append(a);timings.append({'seed':seed,'acquisition_s':r['elapsed_s'],'selection_s':r['selection_s'],
            'reveal_s':r['reveal_s'],'independent_replay_s':a['elapsed_s']})
        e=next(e for e in manifest['inputs'] if e['seed']==seed)
        assert sha(ROOT/e['completion_path'])==e['completion_sha256']==r['source_completion_sha256']
        completion[seed]=np.load(ROOT/e['completion_path'])['solutions']
        for model in r['models']:
            assert sha(ROOT/model['path'])==model['sha256'];m=np.load(ROOT/model['path'])
            support=m['support'];assert support.dtype==bool and len(support)==3240
            assert model['support_bytes_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
            assert np.array_equal(support,m['equal_count']==0)
            missing=int((peer&~support).sum());extra=int((~peer&support).sum());assert missing==0
            arm,n=model['arm'],model['n'];supports[arm,n,seed]=support
            structure.append({'seed':seed,'n':n,'arm':arm,'queried_digits':25*n,'extra_edges':extra,
                'missing_true_edges':missing,'retained_edges':int(support.sum()),'exact_graph':extra==0,
                'joint_observation_min':int(m['jointly_given_count'].min()),
                'joint_observation_max':int(m['jointly_given_count'].max()),
                'joint_observation_sd':float(m['jointly_given_count'].std()),
                'support_bytes_sha256':model['support_bytes_sha256']})
        for name in ['receipt.json','acquisition_audit.json']:hashes[str((d/name).relative_to(ROOT))]=sha(d/name)
    extra={(arm,n):np.array([next(r['extra_edges'] for r in structure if (r['arm'],r['n'],r['seed'])==(arm,n,s)) for s in p['seeds']]) for arm in p['arms'] for n in p['sizes']}
    def cp_limits(k,n,alpha=.05):
        if k==0:lower=0.
        else:
            lo,hi=0.,1.
            for _ in range(100):
                prob=(lo+hi)/2
                tail=sum(math.comb(n,x)*prob**x*(1-prob)**(n-x) for x in range(k,n+1))
                if tail<alpha:lo=prob
                else:hi=prob
            lower=(lo+hi)/2
        if k==n:upper=1.
        else:
            lo,hi=0.,1.
            for _ in range(100):
                prob=(lo+hi)/2
                tail=sum(math.comb(n,x)*prob**x*(1-prob)**(n-x) for x in range(k+1))
                if tail>alpha:lo=prob
                else:hi=prob
            upper=(lo+hi)/2
        return lower,upper
    cp_validation=json.loads((ROOT/'artifacts/research/20261114/ripper/recovery_confirm_cp_validation.json').read_text())
    cp_node=next(node for node in ast.walk(ast.parse(Path(__file__).read_text())) if isinstance(node,ast.FunctionDef) and node.name=='cp_limits')
    assert cp_validation['status']=='validated' and cp_validation['limits_checked']==514
    assert cp_validation['cp_function_ast_sha256']==hashlib.sha256(ast.unparse(cp_node).encode()).hexdigest()
    k=int((extra['exponential',600]==0).sum());lower,upper=cp_limits(k,256)
    decision='supported_at_least_90percent_recovery' if lower>=.90 else 'falsified_90percent_recovery' if upper<.90 else 'inconclusive'
    primary={'decision':decision,'arm':'exponential','n':600,'exact_graphs':k,'banks':256,'observed_fraction':k/256,
        'one_sided95_lower':lower,'one_sided95_upper':upper,'target':.90,'method':'Exact binomial Clopper-Pearson, each limit one-sided95percent.',
        'scope':'Independent fresh source-bank draws from fixed eligible pool. Within-bank sampling without replacement; cross-bank overlap allowed. Not a superiority test.'}
    curve=[]
    for n in p['sizes']:
        for arm in p['arms']:
            v=extra[arm,n];kk=int((v==0).sum());ll,uu=cp_limits(kk,256)
            curve.append({'arm':arm,'n':n,'exact_graphs':kk,'banks':256,'total_extra_edges':int(v.sum()),
                'mean_extra_edges':float(v.mean()),'one_sided95_lower':ll,'one_sided95_upper':uu,'descriptive':True})
    reference_path=ROOT/p['reference_model'];reference_audit=ROOT/p['reference_audit']
    assert sha(reference_path)==p['reference_model_sha256'] and sha(reference_audit)==p['reference_audit_sha256']
    prior=json.loads(reference_audit.read_text());assert prior['status']=='validated' and prior['primary']['decision']=='inconclusive'
    reference=torch.load(reference_path,weights_only=True,map_location='cpu')
    trueW=torch.from_numpy(peer.astype(np.float32))[:,None,None]*torch.eye(9)
    assert torch.equal(reference['W'],trueW) and not reference['B'].any()
    true_support_hash=hashlib.sha256(peer.tobytes()).hexdigest();identities=[]
    for row in structure:
        support=supports[row['arm'],row['n'],row['seed']]
        if row['extra_edges']==0:
            assert np.array_equal(support,peer) and row['support_bytes_sha256']==true_support_hash
            identities.append({'arm':row['arm'],'seed':row['seed'],'n':600,'same_unit1_B0_energy':True})
    finished=now();elapsed=(datetime.datetime.fromisoformat(finished)-datetime.datetime.fromisoformat(manifest['preparation_started_utc'])).total_seconds()
    assert elapsed<=600
    report={'status':'validated','utc':finished,'primary':primary,'structural_controls':curve,'structure':structure,
        'independent_input_audit':input_audit,'input_audit_sha256':sha(OUT/'input_audit.json'),
        'input_registration':registration,'acquisition_audits':audits,'timings':timings,
        'independent_masks_replayed':460800,'prefix_supports_verified':768,'hidden_value_recount_checks':1536,
        'binomial_interval_validation':cp_validation,'reference_model_sha256':sha(reference_path),
        'reference_audit_sha256':sha(reference_audit),'exact_energy_identity_checks':len(identities),
        'exact_energy_identities':identities,'true_support_bytes_sha256':true_support_hash,
        'gpu_minutes':0,'new_solver_runs':0,'new_test_accuracy_observations':0,'cpu_wall_s':elapsed,
        'protocol_sha256':sha(PROTOCOL),'audit_sha256':sha(__file__),'artifact_sha256':hashes,
        'numpy':np.__version__,'earlier_inconclusive_result_not_pooled':True,
        'limitations':p['limitations']+[p['uncertainty_limit']]}
    save(OUT/'audit.json',report);save(PUBLIC/'recovery_confirm.json',report)
    with(PUBLIC/'recovery_confirm_structure.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(structure[0]));w.writeheader();w.writerows(structure)
    summary=json.loads((PUBLIC/'summary.json').read_text());summary['independent_256bank_recovery_confirmation']=report;save(PUBLIC/'summary.json',summary)
    lines=['# H1x: independent256-bank N600 recovery confirmation','',
        f'The frozen >=90% exact-recovery target is **{decision}**. Exponential recovered{k}/256exact graphs atN600 ({100*k/256:.4f}%). The exact binomial one-sided95%lower bound is{100*lower:.4f}%; the separate one-sided95%upper is{100*upper:.4f}%. Support requires at least239/256exact. These are not a95%two-sided interval.','',
        '| Policy | Exact N600 graphs / 256 | Total extra edges |','|---|---:|---:|',
        '| BP AdamW: outside this structural study; not run | — | — |']
    for c in curve:lines.append(f'| {c["arm"]} | {c["exact_graphs"]} | {c["total_extra_edges"]} |')
    lines+=['',f'There are{manifest["fresh_unique_rows"]:,}unique source boards among153,600bank-row instances, with{manifest["cross_bank_duplicate_row_instances"]:,}cross-bank repeated instances. Every bank samples600boards without replacement, independently of other banks. All110,220previously used source rows are excluded. Exact binomial interpretation assumes this independent sampling/tie-stream mechanism.','',
        'All policies remain fixed, with25querieddigitsperboard and15,000labelsperbank. Source-draw, tie, and shadow RNG ranges are disjoint. The full study and sampling protocols froze before input generation; the input hash registration preceded all acquisition. All460,800mask decisions were independently replayed;768supports and1,536hidden-value recounts passed.','',
        'This study has a construction-only primary. Candidate and uniform controls are descriptive. It does not establish superiority, universal recovery, or noise tolerance. The earlier61/64H1w result remains inconclusive and is not pooled or extended.','',
        f'{len(identities)}exact supports were independently matched to the already-audited unit1,B0,true810-edge energy. No new solver runs or accuracy observations were generated; exact energy identity does not create independent test evidence. CPU wall time from preparation through final audit was{elapsed:.2f}s; GPU usage was0minutes.','',
        '[Protocol](protocol-recovery-confirm.json) · [Acquisition](recovery_confirm_acquire.py) · [Independent replay](audit_recovery_confirm_acquisition.py) · [Final audit](audit_recovery_confirm.py) · [Public results](../../../results/research/ripper/recovery_confirm.json)','']
    (HERE/'REPORT-recovery-confirm.md').write_text('\n'.join(lines))
    print(json.dumps({'primary':primary,'controls':curve,'exact_energy_identity_checks':len(identities),'cpu_wall_s':elapsed}))


if __name__=='__main__':main()
