# Classical Energies Learn to Reason 1,000$\times$ Faster than Looped Models

Benjamin Hoover<sup>1</sup>    Hugh Blayney<sup>1</sup>    James C. R. Whittington<sup>1</sup>

**arXiv:2611.17309v1 [cs.AI], 12 November 2026.**

## Abstract

Looped models have emerged as a prominent paradigm for reasoning, with successes on simple reasoning puzzles such as Sudoku-Extreme and Maze-Unique. However, these models can be unstable, slow to train, and use many parameters and FLOPs to solve simple puzzles. Descent on an energy not only mitigates these issues but also offers advantages: built-in early stopping, compute that adapts to problem difficulty, fast training without differentiating through a solver, and interpretable weights that learn the constraints of the puzzle. Here we demonstrate this with a learned version of Hopfield & Tank’s energy from the 80s, which we term CRAM, solves 99.996% of Sudoku-Extreme and 100% of Maze-Unique test puzzles after 42 ms and 132 ms of training on one GPU, at least 1,000${\times}$ faster than looped models on these tasks. CRAM also matches baseline looped models at their highest accuracy on Sudoku-Extreme using at least 282${\times}$ fewer inference FLOPs. Our work suggests two things. First, that energies are an extremely promising direction for training reasoning from data. Second, a caution against using trivially solvable benchmarks like Sudoku and Maze to test the scalability of looped reasoning techniques.

<sup>1</sup> Prefrontal, London, United Kingdom. Correspondence to Benjamin Hoover `<ben@prefrontal.ai>`, James C.R. Whittington `<james@prefrontal.ai>`
