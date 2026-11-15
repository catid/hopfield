"""Independent runtime-proof and paid-query replay on retained fixed fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from common import SRC, read, save, sha, now
from audit_io import AuditIO
from audit_runtime import verify as verify_runtime
from query_audit_gate import run as query_gate
from audit_namespace_gate import verify as verify_namespace
from audit_cap_gate import verify as verify_cap
from role_gate import run as verify_roles


def main(existing, folder):
    folder.mkdir(parents=True, exist_ok=False)
    complete = folder/'runtime'
    complete.mkdir()
    for source, target in [('pruning', 'pruning-gate'), ('cache', 'cache-gate'),
                           ('cost', 'cost-gate'), ('baseline', 'baseline-gate'), ('wrapper', 'wrapper-gate'), ('namespace','namespace-gate'), ('cap','cap-gate')]:
        shutil.copytree(existing/source, complete/target)
    for source, target in [('synthetic-gate.json', 'synthetic-gate.json'),
                           ('pruning/summary.json', 'pruning-gate.json'),
                           ('cache/summary.json', 'cache-gate.json'),
                           ('cost-summary.json', 'cost-gate.json'),
                           ('baseline/summary.json', 'baseline-gate.json'),
                           ('wrapper/summary.json', 'wrapper-gate.json')]:
        shutil.copyfile(existing/source, complete/target)
    pre = read(existing/'preflight.json')
    ns, nt = pre['namespace_gate']['synthetic_memo_states'], pre['namespace_gate']['synthetic_work_steps']
    prep = dict(utc=now(), runtime_gate_memo_states=pre['synthetic_memo_states']-ns-pre['cap_gate']['synthetic_memo_states'],
                runtime_gate_work_steps=pre['synthetic_work_steps']-nt-pre['cap_gate']['synthetic_work_steps'])
    checks = 0
    def check(name, value):
        nonlocal checks
        assert value, name
        checks += 1
    io = AuditIO(complete, SRC, check)
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    runtime = verify_runtime(complete, SRC, prep, check, io, canonical)
    namespace = verify_namespace(complete/'namespace-gate',io,check,ns,nt)
    check('independent_namespace_receipts',all(pre['namespace_gate'][k]==v for k,v in namespace.items()))
    capped=verify_cap(complete/'cap-gate',io,check)
    roles=verify_roles(complete/'namespace-gate',ns,nt)
    check('independent_role_mutation_receipts',roles==pre['namespace_gate']['score_role_mutations'])
    storage = io.finish()
    queries = query_gate(folder/'queries')
    result = dict(status='passed_independent_complete_verifier_components', checks=checks,
                  runtime_gate_audit=runtime, namespace_gate_audit=namespace, cap_gate_audit=capped, score_role_gate=roles, all_fixed_proof_artifacts_reachable=storage,
                  paid_query_binding_gate=queries, fresh_truth_draws=0,
                  fresh_source_draws=0, actual_oracle_calls=0,
                  source_sha256={p.name: sha(p) for p in sorted(SRC.glob('audit_*.py'))})
    save(folder/'summary.json', result)
    print({k: v for k, v in result.items() if k not in ['source_sha256', 'paid_query_binding_gate']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('existing', type=Path)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    main(args.existing, args.folder)
