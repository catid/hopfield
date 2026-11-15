"""Freeze complete N13 prediction equivalence and all hard-tail source cases."""
from common import ROOT,SRC,OUT,read,save,sha,now


def main():
    assert not(SRC/'protocol.json').exists() and not(OUT/'N13-partition.json').exists()
    old=read(ROOT/'research/20261114/loki/h3bf/protocol.json');cases=read(ROOT/'results/research/loki/h3bf/case-results.json');failed=[r['case'] for r in cases if not r['exact_true_singleton']];known=[r['case'] for r in cases if r['exact_true_singleton']];assert len(cases)==1408 and len(failed)==len(known)==704
    sources=[SRC/f'{name}.py' for name in ['common','study','gate','freeze']]+[SRC.parent/'h3at/source.py']
    inputs=['research/20261114/loki/h3bf/protocol.json','results/research/loki/h3bf/summary.json','results/research/loki/h3bf/case-results.json','results/research/loki/h3bf/ledger-export.json','results/research/loki/h3bf/query-ledger.jsonl.gz','results/research/loki/h3bc/summary.json','results/research/loki/h3bc/preparation.json','results/research/loki/h3bc/predictions-bits.npy','results/research/loki/h3bc/ordinary13.npy']+list(old['normalizers'].values())
    p=dict(id='H3bg-exact-N13-identifiability-ceilings',frozen_utc=now(),normalizers=old['normalizers'],candidate_prior=old['candidate_prior'],truths=old['truths'],all_cases=1408,primary_failed_case_ids=failed,known_success_case_ids=known,
           hypothesis='Every one of all704 H3bf four-query failures is still uniquely identifiable in principle using some finite sequence of noiseless N13 membership queries after its same16 source positives.',
           primary='For EACH failed case, intersect its full initial16-positive-consistent candidate set with the true model completeN13 prediction-equivalence class; require exactly [truth]. All704 required. Larger intersections are unavoidable identity ambiguity for any N13-only strategy under the supplied class.',
           scope='All1541 supplied model predictions, including empty sets and all30 source-ineligible models, enter the global equivalence partition. All1408 source cases retained. The704 previously successful cases are checked separately as known-success logical consistency, not new diagnostics.',
           witnesses='For EVERY distinct ordered (true model,other initial candidate) pair occurring in ANY of all1408 cases, retain exact true-valid/other-invalid and true-invalid/other-valid board counts plus lexicographically first ordinaryN13 board in each nonempty direction. Empty directions explicitly zero with null witness. Deduplicate pair calculations only, never cases or aliases.',
           logical_interpretation='Within the finite candidate class and complete73712ordinaryN13 board universe, equal prediction rows are indistinguishable under any adaptive N13-only query sequence. A singleton intersection proves existence of distinguishing queries because every competitor has a witness. Witnesses use truth-aware diagnosis, not a deployable selector; no four-query or optimal query-count guarantee.',
           labels='Zero new source samples or oracle labels. Existing exact populations and case evidence reused. No change to query budget, policy, prior, eligibility or target size.',
           proof_reuse='Hash-pin H3bc full literal census and H3bf full ledger/source/query audit. Fresh full1541 source-mask/target-graph-filter/metadata parity and every1408 source seed/initial-set reconstruction; no repeated literal enumeration.',
           budget=dict(aggregate_CPU_seconds=120,wall_seconds=180,GPUs=0,max_distinct_truth_competitor_pairs=22*1540,enforcement='Whole-study stop incomplete on cap/error. Retain every case and required pair; never prune pairs to fit budget. Public audit separate.'),
           history='Prospective exact ceiling diagnosis after H3bf6/22truths met48/64 and704/1408cases remained unresolved. No globalN13 prediction partition, case ceiling or new pair certificate inspected before this freeze.',
           source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},source_data_sha256={rel:sha(ROOT/rel) for rel in inputs},target_data_sha256={})
    save(SRC/'protocol.json',p);print('FROZEN',sha(SRC/'protocol.json'),flush=True)


if __name__=='__main__':main()
