# Appendices A–C conversion audit

Reviewed `03_appendices_abc.md` against the original LaTeX (`references/source/arxiv_main.tex`, lines 720–1002) and the rendered PDF, pages 14–19. Visually inspected PDF page images for Tables 4 and 5 (pages 14 and 16), and Figure 5 (`hand-set-energies.png`) against its caption and PDF text.

## Cleanup completed

- Rebuilt Table 4 as two Markdown tables, one per task. Checked all 33 numeric cells against both the PDF and LaTeX. The source's merged Maze-Unique EBR cell, “no stopping rule,” is repeated in its three metric columns because Markdown has no merged cells.
- Rebuilt Table 5 as separate training and search Markdown tables. Checked all 40 task-setting cells, including parenthesized larger-task values, batch/update counts, regularization strengths, temperatures, budgets and stopping rules.
- Removed duplicate bullet glyphs and conversion indentation, normalized nonbreaking/thin spaces, and simplified nested color wrappers while preserving the source-exact orange `#EB6834` and bold italic matrix notation. Preserved all prose, formulas, hyperlinks, reference links, captions, footnote references and numerical precision.
- Preserved the Figure 5 raster image and original vector link. Its graphic contains additional parameter values beyond the caption: Maze center +1.50, direct neighbors −1.00, diagonals −0.10; N-Queens per-queen cost −0.5 and attack cost 1; graph-edge color cost is the 3×3 identity. These remain in the original figure without editorial interpolation into the paper text.
- Corrected HTML-generated “Sec. C.5” cross-reference labels to “App. C.5,” matching the PDF.
- Restored all original HTML section anchor IDs for Appendices A–C so the assembled document can preserve cross-references. Figure links remain relative to the final document in `references/`, as supplied by the root agent.

## Source ambiguities and inconsistencies retained

1. **N-Queens temperature and stopping scale are underspecified.** Table 5 calls `u` “the N-Queens temperature unit” and directs readers to C.5, but C.5 only uses it in `exp(-g_i(a)/u)` and never defines how it is calculated. The stopping rule uses `delta - 10^-3` without defining `delta`. A search of the complete LaTeX found no relevant definitions (the `beta_delta` in the Maze-Hard appendix is unrelated). Source: lines 856, 874, 935–936. Resolve using the released implementation before reproducing N-Queens; do not silently invent definitions in the conversion.
2. **EBR inference wording is ambiguous.** The C.1 bullet literally says “the lowest-energy of five samples; OR when its energy-gradient norm falls below 0.2, or after 64 segments (Sudoku Extreme); OR after 16 updates (Maze-Unique).” Both the source and PDF contain these capitalized “OR” words. The conversion preserves them. Appendix A separately states the single-sample halting diagnostic and that Maze-Unique has no gradient-norm stopping rule. Source: lines 722–726, 795.
3. **The broad stopping description is task-specific in the detailed recipe.** Appendix A describes CRAM as stopping at an energy threshold, while Table 5/C.4–C.6 prescribe a fixed point for Maze-Unique, local margins for N-Queens, and per-term training maxima for graph coloring. Preserve each task's detailed rule during implementation. Source: lines 724, 876, 911, 936, 951.
4. **Graph stopping inequality wording differs.** Table 5 says “every term below its training maximum,” while C.6 says “at most its highest value,” an inclusive inequality. Both phrasings are preserved. Source: lines 876 and 951.
5. **Source copy errors are retained, not conversion errors.** C.1 says “more 2 points” (line 784); C.2 says the inference-ready model “is read” (line 818); C.8 says “basline” (line 980). These exact phrases also appear in the PDF. No substantive editorial corrections were made.

## Reproduction caveats explicitly disclosed by the paper

- Main comparisons generally use three training runs; the Figure 3b data-efficiency study explicitly uses six runs (seeds 0–5). This is an explicit exception, not a conversion error.
- Sudoku-Extreme development data are 10,000 rows from the official test split; final testing uses the remaining 412,786. Different main-table baselines use different training draws. The data-efficiency study uses matched rows.
- N-Queens recipe transfer from 10×10 to 8×8 was selected after inspecting test results. Timing uses a matrix-product construction while reported test models use indexing; the authors claim parameter agreement within 10^-7 of the largest parameter.
- CRAM warm-fit timing is the median of seven further fits; long baseline timing is one cold fit. CRAM is eager FP32, baselines compiled BF16. These comparisons cannot be treated as a matched-precision or matched-startup timing experiment.
- Some baseline accuracy entries are published values substituted for weaker local reproductions, with locally counted resources. EqR's high-restart Sudoku evaluation uses 10,000 test puzzles instead of the full held-out set.

No implementation, model fitting or reproduction experiments were performed during this review.
