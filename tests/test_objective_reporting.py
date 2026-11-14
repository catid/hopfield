import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from objective_queue import common_t99


def final(points):
    return {'curve': [{'update': u, 'solved': s, 'n': 512, 'fit_s': u / 10} for u, s in points]}


def test_mean_crossing_can_exist_without_majority_individual_crossings():
    # Two seeds remain below 99%, but their three-run mean exceeds 99%.
    finals = [final([(1, 512)]), final([(1, 506)]), final([(1, 506)])]
    got = common_t99(finals, [.01, .02, .03], [.005] * 3)
    assert got['update'] == 1
    assert got['mean_solved_percent'] == 100 * 1524 / 1536
    assert got['fit_s_per_seed'] == [.1] * 3
    assert got['paper_adjusted_fit_s_per_seed'] == [.09500000000000001, .085, .07500000000000001]


def test_requires_shared_probe_update_and_uses_earliest_crossing():
    finals = [final([(1, 512), (2, 512), (4, 512)]),
              final([(1, 490), (4, 512)]), final([(1, 490), (2, 512), (4, 512)])]
    got = common_t99(finals, [0] * 3, [0] * 3)
    assert got['update'] == 4
    assert [p['update'] for p in got['common_curve']] == [1, 4]


def test_missing_mean_crossing_is_censored():
    got = common_t99([final([(1, 500)])] * 3, [0] * 3, [0] * 3)
    assert got['update'] is None
    assert got['fit_s_per_seed'] == [None] * 3
