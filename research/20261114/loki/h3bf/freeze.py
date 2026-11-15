"""Freeze all selected failures and sixty-four new banks per truth."""
from common import ROOT,SRC,OUT,read,save,sha,now


def main():
    assert not(SRC/'protocol.json').exists() and not(OUT/'preparation.json').exists()
    previous=read(ROOT/'results/research/loki/h3be/summary.json');prior=read(ROOT/'research/20261114/loki/h3be/protocol.json');selected=[r['candidate'] for r in previous['results'] if r['eligible'] and not r['exact_true_singleton']];assert len(selected)==22
    truths=[dict(candidate=cid,orbits=prior['truths'][cid]['orbits'],source_population=prior['truths'][cid]['source_population'],seeds=[2026200000+64*cid+bank for bank in range(64)]) for cid in selected];guided=read(ROOT/'results/research/loki/h3bd/summary.json');pairs=[(f'results/research/loki/h3bd/cases/case{r["case"]}/q{x["step"]}-sealed.json',f'results/research/loki/h3bd/cases/case{r["case"]}/q{x["step"]}-answer.json') for r in guided['results'] for x in r['steps']][:8]
    sources=[SRC/f'{name}.py' for name in ['common','selector','ledger','ledger_gate','study','gate','freeze']]+[SRC.parent/'h3at/source.py']
    inputs=['research/20261114/loki/h3be/protocol.json','research/20261114/loki/h3be/selector.py','results/research/loki/h3be/summary.json','results/research/loki/h3be/runtime-parity.json','results/research/loki/h3bc/summary.json','results/research/loki/h3bc/preparation.json','results/research/loki/h3bc/predictions-bits.npy','results/research/loki/h3bc/ordinary13.npy']+list(prior['normalizers'].values())+[rel for pair in pairs for rel in pair]
    p=dict(id='H3bf-selected-failure-tail-64-bank-robustness',frozen_utc=now(),truths=truths,normalizers=prior['normalizers'],candidate_prior=prior['candidate_prior'],
           hypothesis='For EACH of all22 truths that failed the single-bank H3be run, the unchanged MAP-guided four-query policy uniquely identifies the exact model in at least48of64 new source banks; all22 endpoints required.',
           selection='Explicitly postselected complete H3be failure tail only. All22 identities retained, no dropping hard truths. No whole-class success-rate extrapolation.',
           sampling='64 new independently seeded streams per truth; each draws16 distinct uniform N12 positives without replacement using default_rng(2026200000+64*candidateID+bank).choice(M,16,replace=False). Preserve repeated unordered banks and report unique-bank counts for each truth. No new population or real-world label acquisition.',
           primary=dict(required_exact_per_truth=48,banks_per_truth=64,all_truths_required=22,all_ties_fail=True),
           policy=prior['policy'],posterior=prior['posterior'],initial_candidates=prior['initial_candidates'],stopping=prior['stopping'],supervision=prior['supervision'],
           selector_isolation=prior['selector_isolation'],unchanged_selector_sha256=sha(SRC.parent/'h3be/selector.py'),
           ledger='One append-only JSONL ledger with sequence, previous-record SHA256, UTC, monotonic time, kind and data; canonical sorted compact JSON hash. Every append flushes and fsyncs before return. Query record is committed before oracle is called; next answer references query hash. Source draws and final results also retained. Never regenerate records after answers. Publish exact raw bytes or deterministic gzip with decompressed hash.',
           ledger_gate=dict(archival_query_answer_paths=[list(pair) for pair in pairs],required='Before new source draws, reproduce fixed8 archived query/answer logical fields in old separate-file and new ledger format. Oracle callbacks must read and verify committed query before returning label; enforce sequence/hash/time/reference replay. Reject altered answer, swapped order, broken chain, and rehashed wrong query-reference fixtures. Charge all serialization/gate costs.'),
           prediction_reuse=prior['prediction_reuse'],
           budget=dict(aggregate_CPU_seconds=600,wall_seconds=900,GPUs=0,cases=1408,query_cap_per_case=4,enforcement='Whole-study stop incomplete on budget/error without pruning; all22truths/all64banks retained. Public audit separate.'),
           history='Adaptive robustness test of all22 H3be failure truths after1489/1511overall exact. Newly seeded bank and membership outcomes uninspected before freeze; no policy, prior, threshold or query-budget changes. No new random or balanced comparison.',
           source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},source_data_sha256={rel:sha(ROOT/rel) for rel in inputs},target_data_sha256={})
    assert sha(SRC/'selector.py')==p['unchanged_selector_sha256'];save(SRC/'protocol.json',p);print('FROZEN',sha(SRC/'protocol.json'),selected,flush=True)


if __name__=='__main__':main()
