"""Freeze the four-query policy and controls before any query or seed outcomes."""
from common import ROOT,SRC,OUT,save,sha,now,read


def main():
    assert not(SRC/'protocol.json').exists() and not(OUT/'preparation.json').exists()
    archive='results/research/loki/h3ba/N12-partition.json';partition=read(ROOT/archive)
    classes=[c for c in partition['classes'] if c['eligible_M_at_least16'] and len(c['members'])>1]
    ids=sorted(cid for c in classes for cid in c['members']);assert len(ids)==17 and len(classes)==5
    sources=[SRC/f'{name}.py' for name in ['common','selector','study','gate','freeze']]+[SRC/'reverse_literal.cpp',SRC.parent/'h3at/source.py',SRC.parent/'h3ad/generate.py']
    inputs=[archive]+[f'results/research/loki/h3ba/N13/candidate{cid}/{name}' for cid in ids for name in ['columns.npy','F.npy']]
    p=dict(id='H3bb-four-active-membership-queries',frozen_utc=now(),
           hypothesis='Every one of the 17 models in all five eligible N12 alias classes is uniquely identified within four adaptive binary membership queries at N13.',
           classes=classes,archive_partition=archive,
           given_information='Complete N12 equivalence class of true model; known supplied finite model definitions and all candidate predictions. All N13 populations have already been seen in H3ba. No new-size extrapolation or solver-learning claim.',
           query_universe='All 73712 ordinary-queen N13 boards, sorted lexicographically by zero-based row-to-column vector. No restriction to positive states under any candidate.',
           active_policy='For every not-previously-queried board compute abs(2*positive_prediction_count - K) over ALL currently compatible candidates. Minimize this imbalance, then use lexicographically first board. Do not filter empty candidate populations. Stop only at unique model or four labels.',
           supervision='Noiseless simulated binary membership oracle returns both positive and negative labels; requires supplied candidate solver predictions and given complete N12 class. Label count excludes these assumptions and candidate computation. Not a fair same-oracle comparison to sixteen uniformly positive examples.',
           isolation='Selector is a separate process receiving only candidate prediction matrix, global candidate-ID ordering, current candidate set, and previously queried board IDs. No true-ID or oracle access through its interface. This is source-level interface isolation, not an adversarial OS sandbox.',
           sealing='Write and fsync immutable query receipt before calling hidden-truth oracle; answer receipt references its hash and follows monotonic and UTC time. Preserve every query, answer, and compatible candidate, including failures.',
           primary='All 17 true-model cases leave exactly the true singleton after at most four active queries. Every remaining tie is failure. No class exclusion, no budget tuning after outcomes.',
           random_control=dict(description='Descriptive only: four distinct boards sampled uniformly without replacement from all ordinary N13 boards, using NumPy default_rng(seed).choice(73712,4,replace=False). Entire query order sealed before first answer; stop early when unique; report used labels and unused precommitted queries.',seed_by_candidate={str(cid):2026118000+cid for cid in ids}),
           budget=dict(aggregate_CPU_seconds=120,wall_seconds=180,GPUs=0,query_cap_per_method_per_truth=4,enforcement='Stop whole study as incomplete on cap/error without pruning; public audit separate.'),
           performance='Precompute boolean candidate membership matrix after compiled literal graph enumeration parity; vectorized exact balance scan. Keep selector process isolation despite import overhead. No GPU-eligible fitting work in this exact study.',
           history='Adaptive followup to all five complete-source alias classes separating at N13. Those populations and pair differences are known. No active-query decisions or random-control seed outcomes inspected before this freeze.',
           source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},source_data_sha256={rel:sha(ROOT/rel) for rel in inputs},target_data_sha256={})
    save(SRC/'protocol.json',p);print('FROZEN',sha(SRC/'protocol.json'),flush=True)


if __name__=='__main__':main()
