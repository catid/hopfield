"""Portable H1br replay, with all parent inputs already public."""
import json
from null_signal_study import ROOT,PUBLIC,PROTOCOL,sha
from audit_null_signal import public_audit


def main():
    manifest=ROOT/'results/research/ripper/null_signal_public_manifest.json';m=json.loads(manifest.read_text())
    for rel,h in m['sha256'].items():assert sha(ROOT/rel)==h,rel
    assert sha(PUBLIC/'summary.json')==m['raw_audit_sha256'];public_audit();s=json.loads((PUBLIC/'summary.json').read_text());runtime=json.loads((PUBLIC/'runtime.json').read_text())
    assert runtime==s['runtime'] and runtime['protocol_sha256']==sha(PROTOCOL) and runtime['status']=='validated' and runtime['wall_s']<180 and runtime['aggregate_cpu_s']<900
    commands=runtime['commands'];assert len(commands)==33 and all(c['returncode']==0 for c in commands) and all(c['action']=='bank' for c in commands[:32]) and sorted(c['seed'] for c in commands[:32])==list(range(32)) and commands[-1]['action']=='summary'
    q=s['query_accounting'];assert q==dict(new_official_source_rows=0,new_official_cell_queries=0,new_clean_oracle_cells=0,synthetic_generated_cells=466560,each_null_mask_deployment=414720,null_shared_queries=368719,null_physical_query_union=460721,reused_real_physical_query_union=460721,official_noisy_label_sequence_unchanged=10540119)
    assert s['analytic_zero_witnesses_checked']==128 and s['complete_null_label_gates']==64
    print(json.dumps(dict(status='passed',manifest_paths=len(m['sha256'])+1,all32_banks_retained=True,new_official_labels=0,resource_and_query_caps_passed=True)))


if __name__=='__main__':main()
