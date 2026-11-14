# Manual conversion audit: Appendices D–F

Reviewed `04_appendices_def.md` against the original LaTeX (`source/arxiv_main.tex`, lines 1012–1395), the arXiv HTML, extracted PDF text, and rendered PDF pages 19–24. This review covers all prose in Appendices D, E, and F, Tables 6–8, Equations (5)–(15), and the Figure 6 caption. The figure image is retained separately by the main conversion process.

## Corrections made

- Restored missing equation numbers (14) and (15), and corrected the two-line alignment in Equation (15), whose line separator had become a single backslash.
- Preserved all eleven numbered equations, including signs, indices, transposes, conditioning, underbrace labels, case conditions, and the `m > 4` condition. Added stable anchors to each numbered equation.
- Replaced the unsupported `shortstack` in Equation (5) with an equivalent two-line `substack` label: “lower energy / on data.” Its label and mathematical expression are unchanged.
- Restored the original input/given color `#4A3AA7`; the HTML-to-hex conversion had rounded it to `#4A3AA8`.
- Replaced Table 7's raw HTML with a Markdown table. Repeated task names in the flattened column headers to preserve the original two-level header. Checked all 72 result cells against LaTeX and PDF, including dashes, units, standard deviations, bold best entries, and three dagger markers. Preserved the full longer-schedule footnote.
- Checked all 12 Table 6 method descriptions and all 6 Table 8 model rows. Removed accidental bolding of entire method names in Tables 6–7; the original emphasizes only the class letters L/S/D and best result cells. Preserved the class legend, citations, baseline links, Table 8 dagger meaning, and all published-versus-local distinctions.
- Joined the adjacent math fragments in Table 8's `99.90 ± 0.00` cell into one readable value.
- Removed duplicate bullet characters introduced by conversion. Kept every learning-rate/L1/objective search setting, seed-0 selection rule, update budget, training-timing distinction, and checkpoint-selection detail.
- Changed HTML's “Sec. C.4” references to “App. C.4” to match the actual PDF on pages 22 and 24.

## Source details deliberately preserved

- Table 7's pseudolikelihood Sudoku accuracy is not bold, although it rounds to the same displayed value as the bold contrastive Hebbian entry; this is exactly how the source/PDF marks them.
- Figure 4's full-BPTT comparison runs 60,000 updates, while Table 7's extended full-BPTT schedule runs 3,840,000. These are different source protocols and have not been collapsed.
- Table 7 times are from a shared harness that synchronizes every update; they must not be presented as the optimized fit times of Table 2.
- Appendix E explicitly limits its guarantee to continuous-time dynamics and does not prove uniqueness, correct solutions, or discrete-step behavior. Those qualifications remain intact.
- Equation (14) is transcribed as published. The text separately states that hidden outputs are zero on cells disconnected from G, but the displayed normalized attention formula does not specify handling a zero denominator there. No implementation-dependent guard was invented in the transcription.
- Appendix F keeps the stated breadth-first-search neighbor order, exact-match target convention, three runs with 999/1,000 each, common failure maze, two failed ablations, and fused-Triton implementation statement.

## Mechanical checks after manual review

- Table 6: 12 data rows; Table 7: 12 data rows; Table 8: 6 data rows.
- Numbered equations present once each: 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15.
- No raw HTML table tags, duplicate bullet glyphs, or converter placeholder tokens remain.
