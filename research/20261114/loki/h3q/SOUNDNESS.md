# Conditional soundness is different from certificate availability

This argument concerns the fixed H3o/H3p energy. It does not establish that an arbitrary learned model is sound.

Let the target cardinality be q = 15. Let F be a symmetric, zero-diagonal forbidden-pair matrix containing every true queen/knight attack. The model energy has pair coefficient J_ij = q F_ij − 1 and no unary terms. The initial givens are a valid partial superqueen configuration, and inference never changes those cells. Stopping requires exactly q occupied cells and every free-cell flip cost to be at least q − 1 − ε, with ε = 0.001.

For any occupied free cell i in a q-cell board, its removal cost is

\[
\Delta_i = -\sum_{j\ne i} J_{ij}x_j
         = (q-1)-q\,d_F(i),
\]

where d_F(i) is the number of its occupied forbidden neighbors. A true attack between two occupied cells cannot have both endpoints among the givens, because the givens are valid. At least one endpoint is free. Since F contains every true attack, that endpoint has d_F(i) ≥ 1 and removal cost at most −1, violating the stopping threshold 13.999. Therefore a stopped q-cell board contains no true attack and, with preserved givens, is a valid superqueen completion.

The conclusion is conditional on the supplied q, valid givens, the forbidden-graph superset property and the exact stopping rule. A graph that misses true attacks can certify invalid boards; earlier line-only experiments demonstrate that boundary. Extra exclusions preserve this soundness argument but can prevent certificates for valid completions or alter search efficiency.

For certificate **existence**, take an already valid complete board. Compute every free-cell margin. A deficit at an empty cell cannot be clamped by a positive given; that completion cannot be certified by any permitted given subset. A deficit at an occupied cell disappears from the stopping test only when that cell is a given. Thus a partial input admits that valid completion as a certificate exactly when it clamps every occupied deficit and there is no empty deficit. The census enumerates this condition for every compatible completion and every valid one-, two- and three-clue input.

Certificate existence does not imply that finite search finds it. Conversely, a fallback decoder can return a valid board without a stopped certificate. The population fraction lacking certificates is therefore not a universal bound on arbitrary decoder accuracy. The H3o/H3p output audits separately measure stopped-board validity, certificate availability and search failures.

The coefficients in these fixed models are integers in {−1, 0, 14}. For 225 binary cells, absolute field magnitude is at most 14×224 = 3,136 and absolute energy at most 14×C(225,2) = 352,800. The doubled-energy accumulation is bounded by 705,600, and an incremental energy update by 355,936. These bounds are below the binary32 exact-integer limit 2^24. With the frozen implementation's integer-valued field updates, this prevents accumulated rounding drift in fields/energies. Its binary32 threshold lies strictly between 13 and 14, so integer flip costs pass exactly when they are at least 14. The separate saved-output audits also recompute every stopped margin in float64; the mathematical argument does not substitute for those implementation checks.
