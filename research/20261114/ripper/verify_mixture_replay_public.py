"""Public H1at support/statistics/protocol verification; raw replay is separate."""
from pathlib import Path
from fractions import Fraction
import csv
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
PUBLIC=ROOT/'results/research/ripper'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest=json.loads((PUBLIC/'mixture_replay_public_manifest.json').read_text())
    assert manifest['status']=='validated_public_export'
    for path,digest in manifest['sha256'].items():assert sha(ROOT/path)==digest,path
    s=json.loads((PUBLIC/'mixture_replay.json').read_text())
    p=json.loads((HERE/'protocol-mixture-replay.json').read_text())
    assert s['status']=='validated' and s['protocol_sha256']==sha(HERE/'protocol-mixture-replay.json')
    for path,digest in p['code_sha256'].items():assert sha(ROOT/path)==digest
    assert p['seeds']==list(range(32)) and p['rows']==1110 and p['boundary']==648000
    assert p['q0']=='9/1600' and p['q1']==['1/36','1/18','1/9','2/9'] and p['weights']==['1/4']*4
    assert sha(PUBLIC/'mixture_replay_supports.npz')==s['support_export_sha256']
    with np.load(PUBLIC/'mixture_replay_supports.npz',allow_pickle=False) as z:
        supports=z['supports'];assert np.array_equal(z['banks'],np.arange(32)) and list(z['arms'])==p['arms']
    assert supports.shape==(32,2,3240) and supports.dtype==bool
    with np.load(PUBLIC/'adaptive_null_supports.npz',allow_pickle=False) as z:
        assert np.array_equal(supports[:,0],z['supports'])
    ii,jj=np.triu_indices(81,1)
    truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    assert len(s['rows'])==64 and len(s['bank_audits'])==32
    for bank in range(32):
        for index,arm in enumerate(p['arms']):
            row=next(r for r in s['rows'] if r['seed']==bank and r['arm']==arm);support=supports[bank,index]
            missing=int((truth&~support).sum());extra=int((~truth&support).sum())
            assert row['missing_true_edges']==missing and row['extra_edges']==extra
            assert row['all_true_retained']==(missing==0) and row['exact_graph']==(missing==extra==0)
            assert row['support_bytes_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
        a=s['bank_audits'][bank];assert a['seed']==bank and a['status']=='validated'
        assert a['auditor_sha256']==sha(HERE/'audit_mixture_replay.py') and a['protocol_sha256']==s['protocol_sha256']
        assert a['rows']==[r for r in s['rows'] if r['seed']==bank]
        assert a['cumulative_pair_states_checked']==3596400 and a['original_first_crossing_paths_matched']==3240
        assert a['mixture_first_crossing_paths_checked']==3240 and a['minimum_log_boundary_margin']>1e-10
    for group in s['grouped']:
        rows=[r for r in s['rows'] if r['arm']==group['arm']];assert len(rows)==32
        for key in group:
            if key!='arm':assert group[key]==sum(r[key] for r in rows)
    base,mix=s['grouped'];assert base['extra_edges']==2588 and base['missing_true_edges']==0
    decision='supported_finite_bank_conjunction' if mix['missing_true_edges']==0 and mix['extra_edges']<=1941 else 'falsified_finite_bank_conjunction'
    assert s['primary']['decision']==decision and s['primary']['pooled_extra_edges']==mix['extra_edges']
    assert s['primary']['missing_true_edges']==mix['missing_true_edges'] and s['primary']['extra_ceiling']==1941
    assert s['primary']['retention_target_met']==(mix['missing_true_edges']==0)
    assert s['primary']['extra_target_met']==(mix['extra_edges']<=1941)
    assert s['primary']['extra_reduction_pct']==100*(1-mix['extra_edges']/base['extra_edges'])
    t=s['threshold_audit'];assert t['status']=='validated' and t['auditor_sha256']==sha(HERE/'audit_mixture_replay.py')
    assert t['exposure_states']==1111 and t['arms']==2 and t['decimal_precision']==80
    assert t['decimal_endpoint_predecessor_checks']==4435 and t['minimum_log_boundary_margin']>1e-10
    q0=Fraction(p['q0'])
    for q1text,record in zip(p['q1'],t['conditional_expectations']):
        q1=Fraction(q1text);mean=q0*(q1/q0)+(1-q0)*((1-q1)/(1-q0))
        assert mean==1 and record['q1']==q1text and record['conditional_multiplier_expectation']=='1'
    for key,value in p['query_accounting'].items():assert s[key]==value
    assert s['all_banks_retained'] and s['gpu_seconds']==0
    rt=json.loads((PUBLIC/'mixture_replay_runtime.json').read_text());assert s['runtime']==rt
    assert rt['status']=='validated' and rt['protocol_sha256']==sha(HERE/'protocol-mixture-replay.json')
    assert rt['wrapper_sha256']==sha(HERE/'run_mixture_replay.py')
    assert rt['wall_s']<60 and rt['aggregate_cpu_s']<120 and rt['max_cpu_workers']==8
    assert len(rt['commands'])==34 and all(row['returncode']==0 for row in rt['commands'])
    assert sorted(row['seed'] for row in rt['commands'] if row['action']=='bank')==list(range(32))
    csv_rows=list(csv.DictReader((PUBLIC/'mixture_replay_per_bank.csv').open()))
    assert csv_rows==[{key:str(value) for key,value in row.items()} for row in s['rows']]
    print(json.dumps(dict(status='passed',decision=decision,supports_checked=64,
                         newly_removed_pairs=int((supports[:,0]&~supports[:,1]).sum()),
                         additionally_retained_pairs=int((~supports[:,0]&supports[:,1]).sum()),
                         checks=['protocol/source hashes','exported support statistics','original public support identity','aggregation and primary','conditional expectation algebra','query accounting','runtime','CSV'],
                         raw_observation_replay_required_for_full_likelihood_crossing_verification=True)))


if __name__=='__main__':main()
