"""Freeze the entire eligible truth class and one new source bank per truth."""
from common import ROOT,SRC,OUT,read,save,sha,now


def main():
    assert not(SRC/'protocol.json').exists() and not(OUT/'preparation.json').exists()
    prior=read(ROOT/'research/20261114/loki/h3bd/protocol.json');candidates=read(ROOT/prior['normalizers']['selection_path'])['candidates'];truths=[dict(candidate=c['candidate'],orbit_count=len(c['orbits']),orbits=c['orbits'],source_population=c['complete_source_cardinality'],eligible=c['complete_source_cardinality']>=16,seed=2026120000+c['candidate']) for c in candidates]
    strata=[dict(orbit_count=k,eligible_truths=sum(t['eligible'] and t['orbit_count']==k for t in truths)) for k in [0,1,2]]
    for row in strata:row['required_exact']=(3*row['eligible_truths']+3)//4
    assert len(truths)==1541 and sum(t['eligible'] for t in truths)==1511
    previous=read(ROOT/'results/research/loki/h3bd/summary.json');archival=[f'results/research/loki/h3bd/cases/case{r["case"]}/q{x["step"]}-sealed.json' for r in previous['results'] for x in r['steps']];assert len(archival)==278
    sources=[SRC/f'{name}.py' for name in ['common','selector','study','gate','freeze']]+[SRC.parent/'h3at/source.py',SRC.parent/'h3bd/selector.py']
    inputs=['research/20261114/loki/h3bd/protocol.json','research/20261114/loki/h3au/protocol.json','results/research/loki/h3bd/summary.json','results/research/loki/h3bc/summary.json','results/research/loki/h3bc/preparation.json','results/research/loki/h3bc/predictions-bits.npy','results/research/loki/h3bc/ordinary13.npy']+list(prior['normalizers'].values())+archival
    p=dict(id='H3be-full-truth-class-MAP-query-stress',frozen_utc=now(),truths=truths,strata=strata,normalizers=prior['normalizers'],candidate_prior=prior['candidate_prior'],
           hypothesis='The unchanged source-MAP-guided four-query policy identifies at least75percent of eligible truths SEPARATELY in each true orbit-count stratum0,1,2, on one new sixteen-positive source bank per truth.',
           primary='For each stratum require integer exact count >=ceil(3*n/4); all three stratum endpoints and overall conjunction retained. Exact means current consistent set equals true singleton. Every tie/no-information stop fails. k0 has one truth, not a population-rate estimate.',
           sampling='For each of all1511eligible truths draw16 distinct N12 valid boards uniformly without replacement with NumPy default_rng(2026120000+candidateID).choice(M,16,replace=False). All30M<16 truths remain ineligible, not successes/failures. One bank per truth; no seed search or additional retries.',
           policy=prior['policy'],posterior=prior['posterior'],initial_candidates=prior['initial_candidates'],stopping=prior['stopping'],
           supervision='New simulated draws from known finite source populations, not unseen populations or real-world label acquisition.16 positive N12 observations plus at most4 noiseless binary N13 labels; known finite candidate solver predictions. No complete-source class, true-complexity restriction or empty-model filtering.',
           controls='No new balanced/random control and no transfer of narrow-study paired improvement. Truth-scope generalization only. All prior sampled truth families may recur, but no new-bank outcomes have been inspected.',
           selector_isolation='Persistent separate selector process caches ONLY immutable packed predictions and fixed source scores. Every request fully supplies current candidateIDs and queried boards; no hidden per-case state, trueID or oracle access. Source-level interface isolation, not OS sandbox.',
           optimization=dict(archival_query_paths=archival,parity='Before ANY fresh draws, exactly replay all278 frozen H3bd query requests and require identical full selection dictionaries, including all MAP and board ties.',timing='On fixed first8 archived requests, time old one-shot selector including8 process startups versus8 warm persistent requests. Record persistent startup+first response separately. All overhead charged to study; no end-to-end speedup inferred from microbenchmark.'),
           sealing='Source score table and runtime parity receipt precede every new source draw. Seal each source draw/filter before queries. Fsync read-only query receipt before oracle; answer hash reference and later timestamps. Preserve no-information terminal receipt without answer.',
           prediction_reuse=prior['prediction_reuse'],
           budget=dict(aggregate_CPU_seconds=600,wall_seconds=900,GPUs=0,query_cap_per_truth=4,enforcement='Whole-study cap includes all source draws, full parity, optimization parity/timing and every eligible case; cap/error incomplete without pruning. Public audit separate.'),
           history='Adaptive full-truth-vocabulary stress after H3bd106/108 on restricted seen truth families. Candidate populations and counts already known. No source-bank or membership outcomes for the new seeds inspected before freeze; unchanged prior/policy/four-label cap and no later threshold tuning.',
           source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},source_data_sha256={rel:sha(ROOT/rel) for rel in inputs},target_data_sha256={})
    save(SRC/'protocol.json',p);print('FROZEN',sha(SRC/'protocol.json'),strata,flush=True)


if __name__=='__main__':main()
