"""Replay serialized baseline fixtures without invoking any scientific compiler."""
import gzip
import json
from common import SRC, read, sha, load
from verify_tree import weighted_certificate
from audit_gate import paths


def verify(folder, source=None):
    source = SRC if source is None else source
    reference = load(source.parent/'h3bl/verify_public.py', 'original_weighted_audit')
    firstlex = load(source.parent/'h3bi/verify_public.py', 'original_firstlex_audit')
    summary = read(folder/'summary.json')
    checks = cases = states = work = 0
    def check(name, value):
        nonlocal checks
        assert value, name
        checks += 1
    with gzip.open(folder/'synthetic-baseline.jsonl.gz', 'rt') as stream:
        for line in stream:
            record = json.loads(line)
            fixture = record['input']
            catalog = [dict(signature=s, lexfirst_board_index=i, board_count=1)
                       for i, s in enumerate(fixture['signatures'])]
            old, new, first = record['original'], record['inline'], record['firstlex']
            ns, nt, _ = reference.weighted_certificate(old, catalog, check, 'original_')
            xs, xt, _ = weighted_certificate(new, catalog, check, 'inline_')
            fs, ft, _ = firstlex.certificate(first, catalog, check, 'firstlex_')
            expected = dict(original_states=ns, original_work=nt,
                            inline_states=xs, inline_work=xt,
                            firstlex_states=fs, firstlex_work=ft)
            check('complete_original_inline_parity', record['case'] == cases and
                  fixture['depth'] == 4 and record['counters'] == expected and
                  old['feasible'] == new['feasible'] == first['feasible'] and
                  old['optimal_cost'] == new['optimal_cost'] and paths(old) == paths(new))
            states += ns+xs+fs
            work += nt+xt+ft
            cases += 1
    check('fixture_totals_and_identity', cases == summary['cases'] == 4252 and
          states == summary['memo_states_added'] and work == summary['work_steps_added'] and
          sha(folder/'synthetic-baseline.jsonl.gz') == summary['proof_ledger_sha256'])
    for name, digest in summary['source_sha256'].items():
        check('baseline_gate_source', sha(source/name) == digest)
    return dict(status='passed_independent_baseline_artifact_audit', checks=checks,
                cases=cases, charged_memo_states=states, charged_work_steps=work,
                all_original_firstlex_and_weighted_proofs_verified=True,
                all_depth4_inline_costs_and_candidate_paths_identical=True,
                fresh_truth_draws=0, fresh_source_draws=0, actual_oracle_calls=0)
