"""Read-only audited outcome unions; no inference, flag changes or policy claim."""
import json
import numpy as np
import study as S


def main():
    source=S.OUT/'loop80_transfer';a=json.loads((source/'audit.json').read_text());assert a['status']=='passed';p=json.loads((source/'protocol.json').read_text());rows={r['job']['id']:r for r in a['rows']};byarm={arm:np.concatenate([np.stack([rows[f'shard{s:02d}-s{seed}']['success'][v] for seed in range(3)],axis=1) for s in range(16)]) for v,arm in enumerate(p['arms'])};bp=byarm['pl'].astype(bool);lp=byarm['unified48'].astype(bool);joint_bp=bp.all(1);joint_lp=lp.all(1);joint_union=joint_bp|joint_lp;model_union=(bp|lp).all(1);assert int(joint_bp.sum())==3911 and int(joint_lp.sum())==4044;assert int((joint_bp&~joint_lp).sum())==45 and int(joint_union.sum())==4089
    result={'status':'passed','source_audit_sha256':S.sha(source/'audit.json'),'source_protocol_sha256':S.sha(source/'protocol.json'),'source_sha256':S.sha(__file__),'independent_source_inputs':4096,'model_inputs':12288,'joint_success_bp':int(joint_bp.sum()),'joint_success_unified48':int(joint_lp.sum()),'bp_only_joint_rescues':int((joint_bp&~joint_lp).sum()),'joint_union_success':int(joint_union.sum()),'joint_union_failure_indices':np.flatnonzero(~joint_union).tolist(),'per_model_union_joint_success':int(model_union.sum()),'per_model_union_failure_indices':np.flatnonzero(~model_union).tolist(),'extra_per_model_union_rescues_vs_joint_union':np.flatnonzero(model_union&~joint_union).tolist(),'both_model_failures':int((~bp&~lp).sum()),'new_trajectories':0,'changed_flags':0,'scope':'Archived audited outcome availability only. Joint union uses all-model success from either arm; per-model union may mix BP and LP checkpoints. Neither is a measured executable fallback policy or a new accuracy sample. BP successes retain BP-model certificate provenance. Cap-flag eligibility and cross-objective certificates require separate state inspection.'};dest=S.OUT/'loop80_residual_diagnosis';dest.mkdir(exist_ok=True);S.save(dest/'union.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
