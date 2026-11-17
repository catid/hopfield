# Attribution and licenses

This repository implements and extends **Classical Energies Learn to Reason 1,000× Faster than Looped Models**, Benjamin Hoover, Hugh Blayney and James C. R. Whittington, [arXiv:2611.17309v1](https://arxiv.org/abs/2611.17309v1). It is an independent reproduction and research project, not an official authors' release.

The authors' [CRAM implementation](https://github.com/Prefrontal-AI/CRAM) is pinned at `62bbe830c7d6f87f2ba18f30a28039258b986454`. Adapted search kernels and related derivatives retain its [Apache 2.0 license](references/CRAM-LICENSE.txt). New code in this repository is also provided under [Apache 2.0](LICENSE). Original file notices take precedence where present. The upstream project is downloaded separately, not vendored into this repository.

The retained paper PDF, source, figures and reviewed Markdown conversion in `references/` are by the paper's authors and distributed under the paper's [Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/). The Markdown conversion changes the format and repairs conversion errors, as recorded in [the conversion review](references/CONVERSION_REVIEW.md); it does not claim authorship of the paper.

Dataset licenses remain with their respective providers. Large datasets and checkpoints are not included. Published baseline numbers are explicitly labeled imported results rather than local measurements.

Weight-Space Programming (WSP) baseline and associative tensor-program comparisons reference **Weight-Space Programming Doesn't Need Transformers**, Hyunjong Ok and Jaeho Lee (EFFL Lab, POSTECH, 2026), and the upstream [Weight-Space-Programming repository](https://github.com/effl-lab/Weight-Space-Programming) pinned at `02c31abd3b34d84854cc80d9717717c45365c719` under the MIT License (`Copyright (c) 2026 Hyunjong Ok and Jaeho Lee`). The static tensor-program AST contract (`wsp/free_model.py:static_problems`) is verified against that pinned upstream source, which remains an external clone outside Git rather than vendored code.
