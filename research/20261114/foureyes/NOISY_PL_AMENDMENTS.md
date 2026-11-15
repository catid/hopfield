# H4r implementation amendments

The scientific models, training arrays, seeds, inference settings and hypothesis were unchanged by these two execution fixes.

Before fresh fitting, the source-only CPU/GPU trajectory gate failed its original absolute parameter tolerance of 0.00002. AdamW differed by 0.00013135 and the paper optimizer by 0.00002098. Both selected exactly the same support on CPU and GPU. This is a graph-selection experiment: the complete parameter tensors are discarded before inference. The gate was changed to require identical supports and a canonical-coefficient discrepancy below one quarter of the minimum distance to the selection threshold. Raw parameter discrepancies remain reported. The direct loss, gradient and finite-difference checks were unchanged.

The revised gate observed maximum canonical-coefficient errors of 0.000000548 and 0.000003673. Both passed the separation condition. The independent pair-sum gradient error was 4.66e-10; central finite-difference error was 2.32e-11. The [original protocol](noisy_pl_prelaunch_v1/noisy_pl.json), [original gate](noisy_pl_prelaunch_v1/gate_noisy_pl.py) and [diagnostic record](noisy_pl_prelaunch_v1/failure.json) are retained. The source-only diagnostic record is not evidence that the original gate passed. No fresh model or target search preceded this amendment.

After all fresh fits completed, search initialization found that the copied runner still pointed to H4q's existing artifact directory. Its assertion against an existing queue stopped execution before writing there. The [execution wrapper](execute_noisy_pl_search.py) sets the intended H4r directory explicitly. The frozen search implementation and protocol were preserved; no fits were repeated and no search outcome informed this correction. Use this wrapper, not the copied runner directly.

The final scientific protocol SHA-256 is `8e5f220c1944e4e8f9334c798cf1809008994f838892d6fa3b5a358d26f79da0`. The original preflight protocol SHA-256 was `71a78d110ac28471fe564f4b401e965a9a39486633d1f08236a6aa77178fdf85`.
