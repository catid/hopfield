# H1aq: missing true edges versus global certificate soundness

The target of at least 29 globally sound supports is **inconclusive**: 28 proved sound, 2 explicitly unsound, and 2 unresolved out of all 32 H1ap fits.

| Audit category | Banks |
|---|---:|
| BP AdamW: outside this source-only study; not run | — |
| Complete independently replayed Sudoku-unit proof | 28 |
| Explicit invalid zero-energy coloring | 2 |
| Unresolved | 2 |

All fixed H1ap supports are retained unchanged. Sudoku geometry is supplied only for this exact audit. A retained nine-clique forces all nine colors. Where certified units form two partitions of the same cells, digit-count identities can prove a final missing unit. Reaching all 27 units proves every zero-energy coloring over nine symbols is an ordinary valid Sudoku board, independent of any puzzle or clue set.

The independent Counter-based verifier replayed 35 count-identity steps and rejected 18 mutated proofs. This proof concerns the learned zero set; it does not assert that the graph equals the peer graph or that every capped solver return is valid.

For any unresolved support, the frozen deterministic exact-coloring search forces each missing true pair equal, with a shared two-CPU-second / 100,000-node ceiling per graph. Only a returned board independently satisfying every learned inequality and violating an ordinary Sudoku unit counts as unsound. Timeouts, incomplete closure and no-witness outcomes remain unresolved.

No learned fit, queried label, source board, puzzle row or GPU experiment was added or modified. This audit separates structural edge loss from the stronger claim of a false global certificate.

[Protocol](protocol-permuted-soundness.json) · [Proof and witness certificates](../../../results/research/ripper/permuted_soundness_certificates.json) · [Independent audit](verify_permuted_soundness.py) · [Results](../../../results/research/ripper/permuted_soundness.json)

Execution used 4.4714 CPU-seconds and 4.5057 wall seconds, within the frozen 120 CPU-second / 180-second ceilings.
