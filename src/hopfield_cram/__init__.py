"""Independent, readable CRAM reference implementation (arXiv:2611.17309v1)."""
from .energy import DensePairwiseEnergy, GraphColoringEnergy, MazeEnergy, NQueensEnergy, PairwiseEnergy, SudokuEnergy
from .inference import (ColoringThresholds, GeometricTemperature, RelaxationResult, SearchResult,
                        TrainingStatistics, euler_step, fit_coloring_thresholds, fit_statistics,
                        focused_gibbs, relax)
from .training import TrainingConfig, fit_pseudolikelihood, pseudolikelihood

__all__ = ["PairwiseEnergy", "DensePairwiseEnergy", "SudokuEnergy", "MazeEnergy", "NQueensEnergy",
           "GraphColoringEnergy", "pseudolikelihood", "TrainingConfig", "fit_pseudolikelihood",
           "euler_step", "relax", "focused_gibbs", "fit_statistics", "fit_coloring_thresholds",
           "ColoringThresholds", "GeometricTemperature", "TrainingStatistics", "SearchResult", "RelaxationResult"]

from .latent import MazeHardContext, MazeHardEnergy, MazeHardSearchResult, fit_maze_hard, solve_maze_hard
__all__ += ["MazeHardContext", "MazeHardEnergy", "MazeHardSearchResult", "fit_maze_hard", "solve_maze_hard"]
