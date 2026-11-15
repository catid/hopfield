"""Freeze unchanged posterior guide and a prospective acquisition threshold."""
from common import ROOT,SRC,OUT,read,save,sha,now


def main():
    assert not(SRC/'protocol.json').exists() and not(OUT/'preparation.json').exists()
    old=read(ROOT/'research/20261114/loki/h3bc/protocol.json');summary=read(ROOT/'results/research/loki/h3bc/summary.json');assert summary['status']=='validated' and summary['identified']==27
    sources=[SRC/f'{name}.py' for name in ['common','selector','study','gate','freeze']]+[SRC.parent/'h3at/source.py']
    inputs=['research/20261114/loki/h3bc/protocol.json','research/20261114/loki/h3bc/verify_public.py','research/20261114/loki/h3au/protocol.json','results/research/loki/h3bc/summary.json','results/research/loki/h3bc/preparation.json','results/research/loki/h3bc/predictions-bits.npy','results/research/loki/h3bc/ordinary13.npy']+list(old['normalizers'].values())+[f'results/research/loki/h3bc/cases/case{cfg["case"]}/source-columns.npy' for cfg in old['cases']]
    p=dict(id='H3bd-MAP-guided-membership-verification',frozen_utc=now(),cases=old['cases'],inherited_truth_grid=old['inherited_truth_grid'],normalizers=old['normalizers'],candidate_prior=old['candidate_prior'],
           hypothesis='Source-MAP-guided verification identifies at least81of108 exact supplied models within four binary N13 queries AND all8 singleton-source cases remain exact; report these two primary endpoints separately and as conjunction.',
           posterior='UNCHANGED H3au prior: P(k)=1/3 for k0/1/2, uniform among C(55,k) models. Sixteen distinct uniform positive examples have likelihood1/C(M,16) for consistent models. Current MAP minimizes exact integer C(55,k)*C(M,16) over all surviving candidates; common3 omitted. All MAP ties retained in receipt; smallest candidate ID guides query only.',
           policy='At each step choose current MAP guide, then among all unqueried ordinary-queen N13 boards maximize the NUMBER of current candidates predicting the opposite label from that guide. Lexicographically first board among all maxima. Every compatible candidate survives oracle answer, regardless of posterior weight. No repeated boards.',
           initial_candidates='Exactly H3bc all1541 candidates filtered ONLY by16 positive N12 observations. No completeN12 class, no true complexity restriction, no empty-population filtering.',
           stopping='Stop unique or after4 queries. If no unqueried board has positive disagreement, retain all remaining identities and stop unresolved scientific failure, not numerical failure; no extra budget. Seal such terminal decision without oracle call.',
           primary=dict(total_exact_at_least=81,total_cases=108,singleton_exact_required=8,singleton_cases=8,separate_endpoint_decisions=True,overall_conjunction=True),
           secondary='All108 exact identity criterion reported separately; every paired win/loss/both-pass/both-fail against fixed H3bc trajectories retained. No random outcomes transferred.',
           supervision=old['supervision'],duplicates=old['duplicate_banks'],
           isolation='Selector separate process receives current candidate IDs, query history, shared packed predictions and shared fixed source-score table only. No true ID/oracle interface. Source-level isolation, not OS sandbox.',
           sealing='Source scores sealed before any query. Every query fsynced and read-only before oracle call, with answer hash reference and later times; all MAP ties, query tie counts, states, failures and terminal no-information states retained.',
           prediction_reuse='Hash-pin complete H3bc packed populations and successful independent full literal audit, preserving its coverage boundary. Fresh whole1541 graph-filter parity and metadata/cardinality/hash checks; no duplicate literal enumeration in this study. Prior full literal proof is inherited, not newly measured.',
           budget=dict(aggregate_CPU_seconds=120,wall_seconds=180,GPUs=0,query_cap_per_case=4,enforcement='Whole study cap includes source-score and full prediction parity plus108 cases; stop incomplete without pruning on error/budget. Public audit separate.'),
           history='Adaptive acquisition-policy experiment after H3bc balanced counts identified27/108. Known source-MAP results motivated guidance but current policy outcomes have not been computed. No prior strength, source banks, seed, threshold or four-label budget tuning after these outcomes.',
           source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},source_data_sha256={rel:sha(ROOT/rel) for rel in inputs},target_data_sha256={})
    save(SRC/'protocol.json',p);print('FROZEN',sha(SRC/'protocol.json'),flush=True)


if __name__=='__main__':main()
