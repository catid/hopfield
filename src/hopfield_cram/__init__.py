"""Independent, readable CRAM reference implementation (arXiv:2611.17309v1)."""
from .energy import DensePairwiseEnergy, GraphColoringEnergy, MazeEnergy, NQueensEnergy, PairwiseEnergy, SudokuEnergy
from .inference import (ColoringThresholds, GeometricTemperature, RelaxationResult, SearchResult,
                        TrainingStatistics, euler_step, fit_coloring_thresholds, fit_statistics,
                        focused_gibbs, relax)
from .training import TrainingConfig, fit_pseudolikelihood, pseudolikelihood
from .counting import fit_never_equal, fit_rarely_equal

__all__ = ["PairwiseEnergy", "DensePairwiseEnergy", "SudokuEnergy", "MazeEnergy", "NQueensEnergy",
           "GraphColoringEnergy", "pseudolikelihood", "TrainingConfig", "fit_pseudolikelihood",
           "euler_step", "relax", "focused_gibbs", "fit_statistics", "fit_coloring_thresholds",
           "ColoringThresholds", "GeometricTemperature", "TrainingStatistics", "SearchResult", "RelaxationResult",
           "fit_never_equal", "fit_rarely_equal"]

from .latent import MazeHardContext, MazeHardEnergy, MazeHardSearchResult, fit_maze_hard, solve_maze_hard
__all__ += ["MazeHardContext", "MazeHardEnergy", "MazeHardSearchResult", "fit_maze_hard", "solve_maze_hard"]

from .wsp_contract import (atomic_write_json, check_wsp_static_tensor_program, measure_build_times,
                           sha256_file, static_problems, verify_fullcounter_affine, wsp_weight_metadata)
from .wsp_sudoku import (WSPAssocSearch, build_sudoku_incidence_tensors, build_wsp_assoc_weights,
                         build_wsp_det_weights, build_wsp_transformer_weights, compute_wsp_assoc_puzzle_flops,
                         count_wsp_sudoku_flops, encode_sudoku_clues, evaluate_deterministic_wsp,
                         verify_sudoku_boards, wsp_assoc_move_pytorch)
from .wsp_maze import (WSPMazeResult, arch_deadend_forward, arch_potential_forward, arch_wavefront_forward,
                       bfs_shortest_path_np, build_weights_deadend, build_weights_potential, build_weights_wavefront,
                       path_audit_batch, solve_wsp_deadend, solve_wsp_potential, solve_wsp_wavefront,
                       wsp_potential_energy_and_margin)
from .wsp_multi import (build_graph_coloring_assoc_wsp, build_graph_coloring_det_wsp, build_nqueens_arbitrary_wsp,
                        build_nqueens_det_wsp, build_nqueens_perm_wsp, build_queen_attack_matrix,
                        build_queen_leaper_det_wsp, build_queen_leaper_wsp, compile_wsp_arch, solve_arbitrary_swap,
                        solve_graph_coloring_assoc, solve_graph_coloring_det, solve_nqueens_arbitrary,
                        solve_nqueens_det, solve_permutation_swap, weight_disclosure)

__all__ += [
    "atomic_write_json", "check_wsp_static_tensor_program", "measure_build_times",
    "sha256_file", "static_problems", "verify_fullcounter_affine", "wsp_weight_metadata",
    "WSPAssocSearch", "build_sudoku_incidence_tensors", "build_wsp_assoc_weights",
    "build_wsp_det_weights", "build_wsp_transformer_weights", "compute_wsp_assoc_puzzle_flops",
    "count_wsp_sudoku_flops", "encode_sudoku_clues", "evaluate_deterministic_wsp",
    "verify_sudoku_boards", "wsp_assoc_move_pytorch",
    "WSPMazeResult", "arch_deadend_forward", "arch_potential_forward", "arch_wavefront_forward",
    "bfs_shortest_path_np", "build_weights_deadend", "build_weights_potential", "build_weights_wavefront",
    "path_audit_batch", "solve_wsp_deadend", "solve_wsp_potential", "solve_wsp_wavefront",
    "wsp_potential_energy_and_margin",
    "build_graph_coloring_assoc_wsp", "build_graph_coloring_det_wsp", "build_nqueens_arbitrary_wsp",
    "build_nqueens_det_wsp", "build_nqueens_perm_wsp", "build_queen_attack_matrix",
    "build_queen_leaper_det_wsp", "build_queen_leaper_wsp", "compile_wsp_arch", "solve_arbitrary_swap",
    "solve_graph_coloring_assoc", "solve_graph_coloring_det", "solve_nqueens_arbitrary",
    "solve_nqueens_det", "solve_permutation_swap", "weight_disclosure",
]
