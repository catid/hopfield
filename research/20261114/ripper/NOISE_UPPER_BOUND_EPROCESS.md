# Conservative calibration for adaptive noisy queries

For independent replacement by a uniformly different digit at rate eta, a truly never-equal queried pair has noisy equality probability q0(eta)=eta/4−9eta²/64. Its derivative is 1/4−9eta/32, which is nonnegative on 0<=eta<=8/9. Therefore eta_true<=eta_bound<=8/9 implies q0_true<=q0_bound.

Keep the whole 25-cell mask predictable as in the [original filtration proof](NOISY_ACQUISITION_EPROCESS.md). With fixed q1=1/9 and q0_bound<q1, the equality LR multiplier A=q1/q0_bound exceeds 1 and the unequal multiplier B=(1−q1)/(1−q0_bound) is below 1. Under the true never-equal probability q, its conditional expected multiplier is B+q(A−B), which is at most B+q0_bound(A−B)=1. The product is therefore a nonnegative supermartingale when the calibration is a valid upper bound. The same first-crossing threshold324000 and Ville/union argument protect against deleting any true pair with probability at most1% for one learned model.

The informative regime is 0<eta_bound<8/9. At eta_bound=8/9, q0_bound=q1 and every multiplier is1, so protection is trivial and the test cannot exclude pairs. A rate underbound does not satisfy this argument: its conditional expected multiplier can exceed1. Finite observations of no missing edges or no wrong stops do not repair that missing guarantee.

The experiment fixes actual eta=.01, with assumed rates .01 (exact), .02 (conservative), and .005 (an unprotected stress control). It fixes q1=1/9, exponential exposure weights,25queriesperboard,N1500,and the10pp noninferiority margin. Only the assumed null probability and resulting exact crossing thresholds differ. The pilot checks implementation and cost; it does not select these settings.

The theorem concerns true-edge deletion. It does not guarantee exact graph recovery, limited extra edges, solver accuracy, or zero wrong stops for arbitrary learned models. The fresh experiment separately measures a zero-wrong-stop condition for its two protected arms, without presenting this observation as a consequence stronger than the per-model deletion bound.
