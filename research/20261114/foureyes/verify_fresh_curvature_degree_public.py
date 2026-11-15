"""Portable H4ak exact-proof replay without the upstream training checkout.

The frozen auditor is unchanged. This adapter invokes its complete bank checks,
validates public provenance and explicitly lists unavailable raw dependencies.
"""
import json,time
from pathlib import Path
from audit_fresh_curvature_degree import ROOT,HERE,PUBLIC,PARENT,PROTOCOL,sha,npz,bank,aggregate


def main():
    start=time.monotonic();p=json.loads(PROTOCOL.read_text());s=json.loads((PUBLIC/'summary.json').read_text());unavailable=[]
    assert s['status']=='validated' and s['protocol_sha256']==sha(PROTOCOL) and s['auditor_sha256']==sha(HERE/'audit_fresh_curvature_degree.py')
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():
        file=ROOT/rel
        if file.exists():assert sha(file)==digest,rel
        else:
            assert rel.startswith(('upstream/','artifacts/')),rel
            unavailable.append(rel)
    controls=npz(PARENT/'models.npz');saved_controls=json.loads((PARENT/'summary.json').read_text())
    assert s['controls_summary_sha256']==sha(PARENT/'summary.json') and s['controls_models_sha256']==sha(PARENT/'models.npz')
    assert saved_controls['status']=='validated' and saved_controls['protocol_sha256']==sha(PROTOCOL)
    assert controls['counts'].shape==(64,3240) and controls['groups'].tolist()==list(range(64)) and controls['conditions'].tolist()==['hot18_weight32']*64
    for rel,digest in s['public_sha256'].items():assert sha(PUBLIC/rel)==digest,rel
    rows=[]
    for index in range(64):
        folder=PUBLIC/f'j{index:03d}';receipt=json.loads((folder/'receipt.json').read_text());arrays=npz(folder/'models.npz')
        assert receipt['models_sha256']==sha(folder/'models.npz');checked=json.loads((PUBLIC/'audits'/f'j{index:03d}.json').read_text())
        assert checked['models_sha256']==sha(folder/'models.npz') and checked['receipt_sha256']==sha(folder/'receipt.json')
        assert checked['protocol_sha256']==sha(PROTOCOL) and checked['auditor_sha256']==sha(HERE/'audit_fresh_curvature_degree.py')
        row=bank(index,arrays,receipt,controls['counts'][index],p);assert row==checked['row']==s['rows'][index];rows.append(row)
    grouped,decisions=aggregate(rows);assert grouped==s['grouped'] and decisions['hot18_weight32']==s['primary']
    assert s['centered_exact']==sum(r['centered_control']['exact'] for r in rows)
    assert s['certified_LPs']==sum(r['certified_grid_LPs'] for r in rows) and s['mutations_rejected']==sum(r['mutations_rejected'] for r in rows)
    assert s['analytic_zero_proofs']==64 and s['all64_banks_retained'] and s['all2560_LPs_retained']
    assert s['new_unique_source_rows']==saved_controls['new_unique_source_rows'] and s['new_source_draw_instances']==153600 and s['new_solver_inputs']==0
    runtime=json.loads((PUBLIC/'runtime.json').read_text());assert runtime['status']=='complete' and runtime['error'] is None and runtime['returncodes']==[0]*17 and runtime['protocol_sha256']==sha(PROTOCOL)
    assert runtime['wall_seconds']<p['lp_budget']['wall_seconds'] and runtime['aggregate_cpu_seconds']<p['lp_budget']['aggregate_cpu_seconds']
    last=runtime['samples'][-1];assert last['complete_groups']==64 and last['complete_degrees']==2560 and last['active_pids']==[]
    print(json.dumps(dict(status='passed',primary=s['primary'],certified_LPs=s['certified_LPs'],analytic_zero_proofs=64,mutations_rejected=s['mutations_rejected'],centered_exact=s['centered_exact'],unavailable_raw_or_upstream_dependencies=unavailable,wall_seconds=time.monotonic()-start,
        scope='All public exact proofs, curvature/centered selections, controls bindings and resource receipt checked. Original training code and raw source inputs are hash-checked only when present; raw fitting/source audits are separate.')))


if __name__=='__main__':main()
