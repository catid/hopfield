"""Portable H1cb manifest, independent exact-gap replay and resource checks."""
import json
from posterior_residual_gap import ROOT,PUBLIC,PROTOCOL,sha,read
from audit_posterior_residual_gap import main as audit


def main():
    manifest=read(ROOT/'results/research/ripper/posterior_residual_gap_public_manifest.json')
    for rel,digest in manifest['sha256'].items():assert sha(ROOT/rel)==digest,rel
    assert sha(PUBLIC/'summary.json')==manifest['raw_audit_sha256']
    audit();s=read(PUBLIC/'summary.json');runtime=read(PUBLIC/'runtime.json')
    assert s['runtime']==runtime and runtime['status']=='validated' and runtime['protocol_sha256']==sha(PROTOCOL)
    assert runtime['wall_s']<90 and runtime['aggregate_cpu_s']<120 and runtime['max_cpu_workers']==8 and runtime['gpu_seconds']==0
    commands=runtime['commands'];assert len(commands)==9 and all(c['returncode']==0 for c in commands)
    assert [c['action'] for c in commands[:8]]==['worker']*8 and sorted(c['seed'] for c in commands[:8])==list(range(8))
    assert commands[-1]['action']=='summary'
    progress=read(PUBLIC/'execution/progress.json')
    assert progress['status']=='complete' and progress['active_pids']==[] and progress['completed_commands']==9
    assert progress['selected_arms']==192 and progress['audited_banks']==64
    assert progress['wall_s']==runtime['wall_s'] and progress['cpu_s']==runtime['aggregate_cpu_s']
    assert s['new_LPs']==s['new_labels']==0 and s['selected_certificates_rechecked']==192 and s['proof_mutations_rejected']==576
    assert s['parent_physical_noisy_union']==2073600 and s['official_noisy_label_sequence']==16760919
    import numpy as np
    corollary=read(PUBLIC/'rounding-corollary.json')
    assert corollary['feasible_edge_weight']==810 and corollary['each_feasible_objective_perturbation_maximum']==405 and corollary['gap_perturbation_maximum']==810
    failed=[r for r in s['rows'] if r['arm']=='posterior_triple90' and not r['exact']]
    assert [r['seed'] for r in failed]==[6,7,57]
    for row,bound in zip(failed,corollary['rows'],strict=True):
        with np.load(ROOT/f'results/research/ripper/posterior_fresh_confirm/s{row["seed"]}/posterior_triple90-models.npz',allow_pickle=False) as z:
            assert z['d20_primal_twice'].sum()==1620 and (z['joint_count']==71).all()
        assert bound['bank']==row['seed'] and bound['rounded_gap_twice']==row['gap_twice']
        assert bound['rounded_gap']==row['gap_twice']/2 and bound['unrounded_witness_gap_lower_bound']==row['gap_twice']/2-810>0
    print(json.dumps(dict(status='passed',manifest_paths=len(manifest['sha256'])+1,raw_artifacts_required=False,new_queries=0,new_LPs=0)))


if __name__=='__main__':main()
