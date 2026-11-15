"""Freeze reused positive banks and four membership labels without a source class."""
import itertools
from common import ROOT,SRC,OUT,read,save,sha,now


def main():
    assert not(SRC/'protocol.json').exists() and not(OUT/'preparation.json').exists()
    orbits=list(itertools.combinations(range(1,12),2));choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];cases=[]
    for i in range(8):
        true=[(2,3),(2,5)][i//4];cid=choices.index((orbits.index(true),))
        cases.append(dict(case=len(cases),archive='H3ao',family=i//4,bank=i%4,seed=2026113400+i,true_candidate=cid,true_orbits=[list(true)],draw_path=f'results/research/loki/h3at/b{i}/source-evidence.npz',draw_key='sample_columns'))
    previous=read(ROOT/'results/research/loki/h3ax/summary.json');grid=previous['family_results']
    for row in previous['rows']:
        family=row['family'];bank=row['bank'];cfg=grid[family]
        cases.append(dict(case=len(cases),archive='H3ax',family=family,bank=bank,seed=row['seed'],true_candidate=row['true_candidate'],true_orbits=cfg['true_orbits'],draw_path=f'results/research/loki/h3ax/f{family}/b{bank}/draw.npz',draw_key='columns'))
    assert len(cases)==108 and sum(not r['eligible'] for r in grid)==20
    normalizers=dict(selection_path='results/research/loki/h3at/source-selection.json',ordinary_path='results/research/loki/h3at/source-ordinary-columns.npy',bits_path='results/research/loki/h3at/candidate-validity-bits.npz')
    sources=[SRC/f'{name}.py' for name in ['common','selector','study','gate','freeze']]+[SRC/'reverse_literal.cpp',SRC.parent/'h3at/source.py',SRC.parent/'h3ad/generate.py']
    inputs=list(normalizers.values())+['results/research/loki/h3ax/summary.json','research/20261114/loki/h3ao/protocol.json','research/20261114/loki/h3ax/protocol.json']+[c['draw_path'] for c in cases]
    p=dict(id='H3bc-sixteen-positives-plus-four-membership-queries',frozen_utc=now(),
           hypothesis='Every one of the 108 predeclared eligible reused sixteen-positive source banks identifies its exact true supplied model within four adaptive N13 binary membership queries, starting from all1541 candidates filtered ONLY by those positives.',
           cases=cases,normalizers=normalizers,inherited_truth_grid=grid,
           candidate_prior=dict(orbits=[list(o) for o in orbits],models=1541,definition='Ordinary queens plus zero, one or two D4 leaper orbits 1<=a<b<=11. No likelihood/prior weighting or true-orbit-count restriction.'),
           initial_candidates='All1541 models that accept every distinct one of the sixteen observed N12 positive boards. No complete-N12-population identity, negative-source labels, source-normalizer ranking or empty-target-population filter.',
           query_universe='All73712 ordinary-queen N13 boards, lexicographic zero-based column vectors. No restriction to any candidate positive union.',
           active_policy='Among all unqueried boards minimize abs(2*positive_prediction_count-K) over every currently compatible candidate; choose lexicographically first board among all optimal ties. Retain every compatible candidate; stop at singleton or four labels.',
           primary='Every one of all108 cases has remaining candidate set exactly [true_candidate] after at most four active queries. Every unresolved tie fails. All20 ineligible H3ax grid families are retained separately, not successes or failures. No exclusions or cap tuning.',
           supervision='16 reused uniform distinct positive N12 examples plus up to4 simulated noiseless binary N13 membership labels; report their costs separately and total16+labels. Known finite candidate definitions/solver predictions. No given completeN12 class. Not an equal-oracle comparison to additional uniformly positive examples.',
           random_control='No new random rerun. H3bb random outcomes are not transferred to this study.',
           duplicate_banks='Preserve all100 H3ax cases, including four banks in each of three M12=16 families whose unordered positive samples necessarily coincide. Preserve actual overlaps elsewhere;108 cases are not claimed independent.',
           isolation='Separate selector process gets only packed prediction matrix, candidate-ID ordering, board count, current candidates, queried board IDs. No true-ID or oracle through interface; source-level isolation, not OS sandbox.',
           sealing='fsync read-only query receipt before oracle call; answer references hash and later monotonic/UTC timestamps. Retain all decisions and histories, including incomplete and failure cases.',
           budget=dict(aggregate_CPU_seconds=600,wall_seconds=900,GPUs=0,query_cap_per_case=4,enforcement='Whole-study stop on cap/error without pruning; includes all1541 N13 precompute/literal parity and all108 cases; public audit separate.'),
           performance='Reuse hash-pinned fullN12 bits with full mask/cardinality/hash validation. Compute N13 displacement incidence once, combine candidate orbit masks, and independently literal-enumerate every candidate. Store packed predictions; selector memory-maps and unpacks only current rows. Fixed first17-row parity and two-repetition unpack microbenchmark charged to study.',
           history='Adaptive followup after H3bb all17 complete-source aliases identified within3 active queries. All108 source banks and source-only ML/MAP outcomes already seen; new active trajectories not inspected before freeze. No new seeds or source draws.',
           source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},source_data_sha256={rel:sha(ROOT/rel) for rel in inputs},target_data_sha256={})
    save(SRC/'protocol.json',p);print('FROZEN',sha(SRC/'protocol.json'),flush=True)


if __name__=='__main__':main()
