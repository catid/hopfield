"""Lossless fixed-runtime export and independent raw/archive replay parity."""
import argparse
import json
from pathlib import Path
from common import SRC,read,save,sha,load,now
from parallel_export import export_tree
from audit_io import AuditIO
from audit_runtime import verify as verify_runtime
from audit_namespace_gate import verify as verify_namespace
from audit_cap_gate import verify as verify_cap
from role_gate import run as verify_roles


def run(source,preflight,destination):
    adapter=load(SRC.parent/'archive_export_layer.py','h3cb_fixed_export')
    manifest,timing=export_tree(source,destination,adapter)
    check_count=0
    def check(name,value):
        nonlocal check_count
        assert value,name;check_count+=1
    io=AuditIO(destination,SRC,check)
    for path in source.rglob('*'):
        if path.is_file():check('fixed_export_bytes_identical',io.data(destination/path.relative_to(source))==path.read_bytes())
    pre=read(preflight);namespace=pre['namespace_gate'];cap=pre['cap_gate']
    ns,nt=namespace['synthetic_memo_states'],namespace['synthetic_work_steps']
    prep=dict(utc=now(),runtime_gate_memo_states=pre['synthetic_memo_states']-ns-cap['synthetic_memo_states'],
              runtime_gate_work_steps=pre['synthetic_work_steps']-nt-cap['synthetic_work_steps'])
    canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    runtime=verify_runtime(destination,SRC,prep,check,io,canonical)
    paired=verify_namespace(destination/'namespace-gate',io,check,ns,nt)
    capped=verify_cap(destination/'cap-gate',io,check)
    roles=verify_roles(destination/'namespace-gate',ns,nt,archive_root=destination)
    check('role_mutations_replay_from_archive',roles==namespace['score_role_mutations'])
    check('paired_archive_receipts',all(namespace[k]==v for k,v in paired.items()))
    storage=io.finish()
    result=dict(status='passed_fixed_full_runtime_archive_replay',checks=check_count,export_timing=timing,
                runtime_gate_audit=runtime,namespace_gate_audit=paired,cap_gate_audit=capped,
                score_role_gate=roles,storage=storage,fresh_truth_draws=0,fresh_source_draws=0,actual_oracle_calls=0,
                source_sha256={n:sha(SRC/n) for n in ['archive_gate.py','parallel_export.py','audit_io.py','role_gate.py','audit_cap_gate.py']})
    save(destination/'archive-gate.json',result);print({k:v for k,v in result.items() if k in ['status','checks','storage','export_timing']});return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('preflight',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();run(a.source,a.preflight,a.destination)
