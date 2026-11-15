"""Verify H4ab proof outcomes together with the preserved correction lineage."""
from pathlib import Path
import hashlib
import json
from verify_strong_support_soundness_corrected import audit

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PUBLIC=ROOT/'results/research/foureyes/strong_support_soundness'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    result=audit()
    old=HERE/'verify_strong_support_soundness.py';new=HERE/'verify_strong_support_soundness_corrected.py'
    before='def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()'
    after='def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()'
    assert old.read_text().count(before)==1 and old.read_text().replace(before,after)==new.read_text()
    initial=HERE/'strong-soundness-audit-correction-initial.json';current=HERE/'strong-soundness-audit-correction.json'
    first=json.loads(initial.read_text());note=json.loads(current.read_text());scope=json.loads((PUBLIC/'timing-scope.json').read_text())
    assert first['original_verifier_sha256']==note['original_verifier_sha256']==sha(old)
    assert first['corrected_verifier_sha256']==note['corrected_verifier_sha256']==sha(new)
    assert note['initial_metadata_receipt_sha256']==scope['initial_correction_receipt_sha256']==sha(initial)
    assert scope['current_correction_receipt_sha256']==sha(current)
    assert scope['initial_bound_withdrawn']and not scope['original_run_complete_cpu_accounting_available']
    assert 'original_cpu_conservative_upper_seconds'not in note and note['producer_rerun']
    replay=json.loads((PUBLIC/'replay-runtime.json').read_text())
    assert replay['status']=='validated'and replay['byte_identical_replay']and replay['correction_sha256']==sha(initial)
    assert replay['source_sha256']==sha(HERE/'replay_strong_support_soundness.py')
    for n,h in replay['original_artifacts_sha256'].items():assert sha(PUBLIC/n)==h,n
    for rel,h in note['producer_artifacts_sha256'].items():assert sha(ROOT/rel)==h,rel
    assert replay['aggregate_cpu_seconds']<180 and replay['wall_seconds']<240 and replay['gpu_seconds']==0
    print(json.dumps(dict(status='passed',counts=result['counts'],decision=result['primary']['decision'],correction_lineage=True,
                         complete_replay_cpu_seconds=replay['aggregate_cpu_seconds'],complete_replay_wall_seconds=replay['wall_seconds'],original_complete_cpu_accounting_available=False)))


if __name__=='__main__':main()
