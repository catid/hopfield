"""Freeze an eight-query intervention on the already examined hard-tail banks."""
from common import ROOT,SRC,OUT,read,save,sha,now


def main():
    assert not(SRC/'protocol.json').exists() and not(OUT/'continuation-checkpoints.json').exists()
    old=read(ROOT/'research/20261114/loki/h3bf/protocol.json');summary=read(ROOT/'results/research/loki/h3bf/summary.json');cases=read(ROOT/'results/research/loki/h3bf/case-results.json');assert summary['membership_labels']==5372 and len(cases)==1408 and sum(not r['exact_true_singleton'] for r in cases)==704
    sources=[SRC/f'{name}.py' for name in ['common','selector','ledger','checkpoints','study','gate','freeze']]+[SRC.parent/'h3at/source.py']
    inputs=['research/20261114/loki/h3bf/protocol.json','research/20261114/loki/h3bf/selector.py','research/20261114/loki/h3bf/ledger.py','results/research/loki/h3bf/summary.json','results/research/loki/h3bf/case-results.json','results/research/loki/h3bf/query-ledger.jsonl.gz','results/research/loki/h3bf/ledger-export.json','results/research/loki/h3bf/ledger-gate.json','results/research/loki/h3bg/summary.json','results/research/loki/h3bc/summary.json','results/research/loki/h3bc/preparation.json','results/research/loki/h3bc/predictions-bits.npy','results/research/loki/h3bc/ordinary13.npy']+list(old['normalizers'].values())
    p=dict(id='H3bh-eight-query-hard-tail-continuation',frozen_utc=now(),truths=old['truths'],normalizers=old['normalizers'],candidate_prior=old['candidate_prior'],
           hypothesis='Keeping every original H3bf decision fixed, at most4additional N13 membership queries per unresolved case (at most8total) raises EACH selected truth to at least48of64 exact identities; separately test stronger61of64 pertruth.',
           primary=dict(required_exact_per_truth=48,banks_per_truth=64,truths=22,overall_conjunction=True),secondary=dict(required_exact_per_truth=61,banks_per_truth=64,truths=22,overall_conjunction=True),
           cases='All1408 original H3bf cases retained.704alreadyunique cases stopped identically with0additional labels;704unresolved cases have4original labels and receive at most4more. No source draws, no original receipt changes, no extension beyond8.',
           policy=old['policy'],posterior=old['posterior'],initial_candidates=old['initial_candidates'],
           stopping='Start from exact sealed original remaining set and full original query history; never repeat a board. Stop unique/no-informative-query or8total labels. All ties/no-information unresolved failures. Retain failures without rescue beyond cap.',
           interpretation='Budget intervention on examined postselected hard-tail source banks, not fresh confirmation or optimal-policy claim. H3bg shows all are identifiable in principle but does not guarantee8queries. No random/balanced comparison.',
           label_accounting='Original5372 binary membership labels retained and reported separately from additional labels. Eachcase keeps original16sourcepositives; no freshsourceclaim. Counts of additional and cumulative labels both reported.',
           selector_isolation=old['selector_isolation'],
           ledger='Exact unchanged H3bf ledger writer and selector source bytes. Inherit prior durability/tamper synthetic gate by hash. New ledger starts with original ledger/summary hash and checkpoints; eachcase checkpoint references immutable original case-result record/final candidates/allqueriedboards. New queries fsynced before oracle and chained to newanswers; originalreceipts never regenerated.',
           predecision_gate='Before ANY new continuation decision, validate entire original ledger rawhash/order/timestamps/query-answer references and every1408case checkpoint, source seed, initialfilter and finalstate. Replay ALL5372 original selector requests in persistent process and require identical complete selection dictionaries. No re-acquired labels in this deterministic replay.',
           proof_reuse='H3bcfull literal census and H3bfledger/source/queryaudit inherited by hash; fresh whole1541graph-filter/source-score parity and full originaldecision/checkpoint replay. No repeated literal enumeration.',
           budget=dict(aggregate_CPU_seconds=120,wall_seconds=180,GPUs=0,additional_query_cap_per_unresolved_case=4,total_query_cap_per_case=8,enforcement='Whole-study stop incomplete on budget/error without pruning cases; allparity/replay/serialization costs charged. Publicauditseparate.'),
           history='Adaptive budget test after H3bfpersistentfailures and H3bgall704failures identifiableinprinciple. Sourcebanks/four-queryoutcomes/ceilings known; no fifth-or-later query outcomes inspected beforefreeze. No prior, policy, threshold or cap tuning after newoutcomes.',
           source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},source_data_sha256={rel:sha(ROOT/rel) for rel in inputs},target_data_sha256={})
    assert sha(SRC/'selector.py')==sha(SRC.parent/'h3bf/selector.py') and sha(SRC/'ledger.py')==sha(SRC.parent/'h3bf/ledger.py');save(SRC/'protocol.json',p);print('FROZEN',sha(SRC/'protocol.json'),flush=True)


if __name__=='__main__':main()
