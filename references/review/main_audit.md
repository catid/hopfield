# Main-text transcription audit

Reviewed `01_main.md` against PDF pages 1–9, the official HTML, and `source/arxiv_main.tex`. Scope: Sections 1–5, acknowledgments, Equations (1)–(4), Tables 1–3, and captions of Figures 1–4. Figure images are retained separately from the text.

## Corrections and fidelity checks

- Rebuilt Table 2 as two task-specific Markdown tables and Table 3 as four task-specific Markdown tables. All captions, units, model links, citations, footnotes, values, and hand-set labels remain. A cell-by-cell comparison with the official HTML confirmed all 72 Table 2 data cells and all 120 Table 3 data cells. Table 1 was checked manually against the PDF/source.
- Preserved bold best-value means separately from unbolded standard deviations, following the TeX source rather than the HTML conversion's broadened emphasis. Gray hand-set styling is represented by the explicit `(hand-set)‡` label; no hand-set result is relabeled as a trained result.
- Checked the entropy, interaction, input terms, summation exclusions, transposes, signs, expectation distributions, underbrace labels, and numbers in Equations (1)–(4). Retained the separate equality-up-to-a-scalar explanation after Equation (2), the symmetrized coupling definition, and the zero self-interaction condition.
- Retained blue/orange/green mathematical semantics and their prose legends. Colored formulas retain the exact mathematical expressions.
- Corrected the duplicated TRMP footnote marker and moved its text to a named Markdown footnote; replaced the converter-only section-sign macro with `§`; changed malformed subscript markup to `$T_{99}$`; restored `App.` references matching the PDF. Corrected only the missing space in `(CRAM).We`.
- Moved floating material into readable order: Figure 1 after the opening three introduction paragraphs, Tables 1–2 after the complete method derivation, and Table 3 after the associated multi-solution-task paragraph. No body text or captions were discarded.
- Added section anchors. Checked the four figure captions, all five main sections, and the acknowledgment names against the source. Equations (1)–(4) and Figures 1–4 each occur once. No raw HTML tables remain in this part.

## Source wording and numerical distinctions retained

- The abstract/main text reports 99.996% Sudoku-Extreme accuracy; Table 2 rounds its mean and standard deviation to `100.00 ±0.00`; the Figure 2 discussion reports 99.997% for a separate 10,000-puzzle inference sweep. These are retained as printed, without harmonizing precision or evaluation scope.
- The prose says the hand-set energy solves every puzzle, while Table 3 reports `99.99 ±0.0` for 10×10 N-Queens with one chain and `100.00 ±0.0` with 20 chains. The broader claim and the row values both remain.
- Table 3's N-Queens Cov@20 note says normalization over every valid solution; Appendix C.5 specifies a denominator capped at `min(S, 20)`. This is an original wording discrepancy, not a transcription correction.
- The paper's initial 263K Sudoku parameter count is the full trainable parameter count; Table 2's 7.7K count is nonzero contributing parameters. Both labels and values are preserved.
- Timing is explicitly warm median-of-seven CRAM fits versus cold fresh-process baseline fits, on a shared H200 cluster. No local timing or reproduction claim is introduced.
- Original grammatical/typographical wording remains, including “trained to minimizes,” “using with SGD,” “shaded are,” and “suggested the looping.”

No implementation, training, evaluation, or experiments were started during this transcription review.
