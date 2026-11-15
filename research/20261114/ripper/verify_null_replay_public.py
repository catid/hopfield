"""Public support statistics, conditional algebra and correction-lineage check."""
from pathlib import Path
from fractions import Fraction
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;PUBLIC=ROOT/'results/research/ripper'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    s=json.loads((PUBLIC/'null_replay.json').read_text());p=json.loads((HERE/'protocol-null-replay-corrected.json').read_text());parent=json.loads((PUBLIC/'permuted_noise.json').read_text())
    assert s['status']=='validated'and s['protocol_sha256']==sha(HERE/'protocol-null-replay-corrected.json')
    for path,digest in p['code_sha256'].items():assert sha(ROOT/path)==digest
    correction=json.loads((HERE/'null-replay-pre-run-correction.json').read_text())
    assert correction['original_protocol_sha256']==sha(HERE/'protocol-null-replay.json')
    for field,name in [('original_producer_sha256','null_replay_study.py'),('corrected_producer_sha256','null_replay_study_corrected.py'),('original_auditor_sha256','audit_null_replay.py'),('corrected_auditor_sha256','audit_null_replay_corrected.py'),('synthetic_verifier_sha256','verify_null_replay_correction.py'),('synthetic_check_sha256','null-replay-pre-run-correction-check.json'),('diff_sha256','null-replay-pre-run-correction.diff'),('failure_record_sha256','null-replay-original-preflight-failure.txt')]:assert correction[field]==sha(HERE/name)
    assert not correction['outcome_access_before_correction']and not correction['experiment_launched_before_correction']and not correction['scientific_settings_changed']
    assert sha(PUBLIC/'null_replay_supports.npz')==s['support_export_sha256']
    with np.load(PUBLIC/'null_replay_supports.npz',allow_pickle=False)as z:
        supports=z['supports'];assert np.array_equal(z['banks'],np.arange(32))and list(z['arms'])==list(p['nulls'])
    assert supports.shape==(32,3,3240)and supports.dtype==bool
    i,j=np.triu_indices(81,1);truth=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3));assert truth.sum()==810
    for bank in range(32):
        for arm_index,arm in enumerate(p['nulls']):
            r=next(r for r in s['rows']if r['seed']==bank and r['arm']==arm);support=supports[bank,arm_index]
            assert r['support_bytes_sha256']==hashlib.sha256(support.tobytes()).hexdigest()
            missing=int((truth&~support).sum());extra=int((~truth&support).sum())
            assert r['missing_true_edges']==missing and r['extra_edges']==extra and r['all_true_retained']==(missing==0)and r['exact_graph']==(missing==extra==0)
            if arm=='calibrated':
                old=next(r for r in parent['acquisition_audits']if r['seed']==bank);assert old['support_bytes_sha256']==r['support_bytes_sha256']
    for g in s['grouped']:
        rows=[r for r in s['rows']if r['arm']==g['arm']];assert len(rows)==32
        for field in g:
            if field!='arm':assert g[field]==sum(r[field]for r in rows)
    for r in s['threshold_audit']['conditional_expectations']:
        q0=Fraction(p['nulls'][r['arm']]);q1=Fraction(1,9);actual=Fraction(9,1600)
        expected=actual*q1/q0+(1-actual)*(1-q1)/(1-q0)
        assert str(expected)==r['conditional_multiplier_expectation']and float(expected)==r['expectation_float']
    exact=next(g for g in s['grouped']if g['arm']=='exact_law');assert exact['all_true_retained']==s['primary']['all_true_retaining_banks']
    decision='supported_finite_bank_target'if exact['all_true_retained']>=29 else'falsified_finite_bank_target';assert decision==s['primary']['decision']
    print(json.dumps(dict(status='passed',support_cells=96,decision=decision,checks=['public support statistics','aggregation','conditional expectation algebra','preserved pre-run correction lineage'],raw_acquisition_replay_required_for_full_crossing_verification=True)))
if __name__=='__main__':main()
