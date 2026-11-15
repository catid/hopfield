"""Evaluator enumerates paired outcomes of the sealed pure learner, source only."""
import hashlib
import itertools
import math
import numpy as np
from common import ROOT, SRC, OUT, frozen, read, save, sha, now
from learner import views, offsets, repair
from distribution import union_mass, validate


def literal_valid(board):
    assert np.asarray(board).shape == (169,) and np.sum(board) == 13
    for a, b in itertools.combinations(np.flatnonzero(board), 2):
        dr, dc = abs(int(a)//13-int(b)//13), abs(int(a)%13-int(b)%13)
        assert dr and dc and dr != dc and (dr, dc) not in ((1, 3), (3, 1))


def main():
    p = frozen()
    assert not (OUT/'source-outcomes.json').exists()
    save(OUT/'distribution-validation.json', validate())
    # Validate eight coordinate maps independently of numpy rotation/reflection.
    for cell in range(169):
        board = np.zeros(169, bool)
        board[cell] = True
        r, c = divmod(cell, 13)
        mapped = {(r,c), (c,12-r), (12-r,12-c), (12-c,r), (r,12-c), (12-c,12-r), (12-r,c), (c,r)}
        assert {int(np.flatnonzero(v)[0]) for v in views(board)} == {13*a+b for a,b in mapped}
    population = np.load(ROOT/p['source_population']['path'])
    raw_seen, augmented_seen, counts = [], [], []
    for board in population:
        literal_valid(board)
        transformed = views(board)
        for view in transformed:
            literal_valid(view)
        raw_seen.append(offsets([board]))
        augmented_seen.append(offsets(transformed))
        counts.append(len(transformed))
    assert all(a <= b for a, b in zip(raw_seen, augmented_seen))
    all_seen = set().union(*augmented_seen)
    support_records, arrays, banks, fixtures = [], [], [], []
    total_masks = 0
    for bank, cfg in enumerate(p['models']):
        model = np.load(ROOT/cfg['model_path'])
        base = model['F']
        finite = {tuple(o) for o in np.load(ROOT/cfg['source_design_path'])['offsets']}
        assert not model['h'].any() and (model['J'][base.astype(bool)] > 0).all() and not model['J'][~base.astype(bool)].any()
        classes = sorted((dr,dc) for dr,dc in all_seen & finite if base[max(0,-dc),15*dr+max(0,dc)])
        m = len(classes)
        assert m <= p['complexity']['classes_per_arm_bank_max']
        total_masks += 1 << (2*m)
        assert total_masks <= p['complexity']['sum_paired_bank_masks_max']
        raw_masks = np.array([sum(1 << i for i,o in enumerate(classes) if o in seen) for seen in raw_seen], np.int64)
        aug_masks = np.array([sum(1 << i for i,o in enumerate(classes) if o in seen) for seen in augmented_seen], np.int64)
        paired = raw_masks | (aug_masks << m)
        frequency = np.bincount(paired, minlength=1 << (2*m))
        masses = union_mass(frequency.tolist(), 32)
        assert sum(masses) == math.comb(1932,32)
        for mask, weight in enumerate(masses):
            if weight:
                assert (mask & ((1 << m)-1)) | (mask >> m) == mask >> m
        mapping = []
        for mask in range(1 << m):
            F = base.copy()
            deleted = {classes[i] for i in range(m) if mask & (1 << i)}
            for a,b in itertools.combinations(range(225),2):
                if (b//15-a//15,b%15-a%15) in deleted:
                    F[a,b] = F[b,a] = 0
            digest = hashlib.sha256(F.tobytes()).hexdigest()
            existing = [r['group'] for r in support_records if r['support_sha256'] == digest]
            if existing:
                group = existing[0]
                assert np.array_equal(F,arrays[group])
            else:
                group = len(arrays)
                assert group < p['complexity']['distinct_supports_max']
                folder = OUT/f'support{group}'
                folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/'F.npz',F=F)
                arrays.append(F)
                support_records.append(dict(group=group,support_sha256=digest,F_npz_sha256=sha(folder/'F.npz'),members=[]))
            mapping.append(group)
            support_records[group]['members'].append(dict(bank=bank,mask=mask))
        # Source-only concrete32-board calls verify union-factorization and actual view cost.
        for seed in p['validation']['sample_call_seeds']:
            indices = np.random.default_rng(seed).choice(1932,32,replace=False)
            for augment, masks in [(False,raw_masks),(True,aug_masks)]:
                F, cost = repair(population[indices],base,finite,augment)
                union = int(np.bitwise_or.reduce(masks[indices]))
                assert np.array_equal(F,arrays[mapping[union]])
                fixtures.append(dict(bank=bank,seed=seed,augment=augment,indices=indices.tolist(),union_mask=union,cost=cost,union_factorization_exact=True))
        folder=OUT/f'b{bank}'
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'source-patterns.npz',raw_masks=raw_masks,augmented_masks=aug_masks,paired_masks=paired,frequency=frequency,classes=np.array(classes,np.int16),unique_views_per_board=np.array(counts,np.uint8))
        banks.append(dict(bank=bank,class_count=m,classes=[list(o) for o in classes],frequency=frequency.tolist(),all_paired_masks=len(masses),draw_masses=[str(weight) for weight in masses],total_draws=str(math.comb(1932,32)),mask_to_support=mapping,source_patterns_sha256=sha(folder/'source-patterns.npz')))
    save(OUT/'source-validation.json',dict(status='passed',coordinate_basis_fixtures=169,all_eight_maps_and_dedup_checked=True,all_source_and_D4_view_labels_valid=True,unique_views_per_board_histogram={str(k):counts.count(k) for k in sorted(set(counts))},pure_learner_accepts_only_selected32_positives_inherited_model_and_finite_classes=True,no_learner_file_or_population_reads=True,concrete32_calls=fixtures,learner_sha256=sha(SRC/'learner.py')))
    save(OUT/'source-outcomes.json',dict(status='complete_source_only',utc=now(),banks=banks,supports=support_records,total_paired_bank_masks=total_masks,distinct_full_supports=len(arrays),all_zero_mass_masks_retained=True,full_joint_raw_and_augmented_union_distribution=True,source_validation_sha256=sha(OUT/'source-validation.json'),no_target_truth_keys_or_scores_used=True,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))


if __name__ == '__main__':
    main()
