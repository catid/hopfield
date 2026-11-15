"""Independent endpoint walk and grid BFS for the H2bm source analysis."""
from collections import deque
import numpy as np


def checked_arrays(tokens, board):
    t = np.asarray(tokens)
    b = np.asarray(board)
    if t.ndim != 2 or t.shape != b.shape or b.dtype != np.bool_:
        raise ValueError('Expected matching 2D tokens and boolean board')
    if not np.isin(t, [0, 1, 2, 3]).all():
        raise ValueError('Invalid token code')
    starts = np.flatnonzero(t.ravel() == 2)
    goals = np.flatnonzero(t.ravel() == 3)
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError('Expected one start and one goal')
    return t, b, int(starts[0]), int(goals[0])


def neighbors(index, height, width):
    row, col = divmod(index, width)
    if row:
        yield index - width
    if col + 1 < width:
        yield index + 1
    if row + 1 < height:
        yield index + width
    if col:
        yield index - 1


def simple_path(tokens, board):
    """Walk the sole continuation from the start; reject branches and leftovers."""
    t, b, start, goal = checked_arrays(tokens, board)
    flat = b.ravel()
    if not flat[start] or not flat[goal] or np.any(flat & (t.ravel() == 0)):
        return False
    height, width = t.shape
    previous = -1
    current = start
    seen = set()
    while True:
        if current in seen:
            return False
        seen.add(current)
        forward = [v for v in neighbors(current, height, width)
                   if flat[v] and v != previous]
        if current == goal:
            return not forward and len(seen) == int(flat.sum())
        if len(forward) != 1:
            return False
        previous, current = current, forward[0]


def shortest_distance(tokens):
    t = np.asarray(tokens)
    _, _, start, goal = checked_arrays(t, np.zeros(t.shape, dtype=bool))
    height, width = t.shape
    flat = t.ravel()
    distances = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            return distances[current]
        for nxt in neighbors(current, height, width):
            if flat[nxt] != 0 and nxt not in distances:
                distances[nxt] = distances[current] + 1
                queue.append(nxt)
    return None


def classify(tokens, board):
    valid = simple_path(tokens, board)
    distance = shortest_distance(tokens)
    edges = int(np.count_nonzero(board)) - 1 if valid else None
    if valid and (distance is None or edges < distance):
        raise AssertionError('Simple path contradicts independent BFS')
    return {'graph_valid_path': int(valid),
            'graph_valid_shortest': int(valid and edges == distance),
            'shortest_edges': distance, 'selected_path_edges': edges,
            'extra_edges': edges - distance if valid else None}


def partition(valid, shortest):
    if valid not in (0, 1) or shortest not in (0, 1):
        raise ValueError('Graph fields must be binary')
    if shortest and not valid:
        raise ValueError('A shortest path must have valid geometry')
    return 'valid_shortest' if shortest else ('valid_longer' if valid else 'invalid_geometry')


def endpoint(invalid_geometry_failures, failures):
    if failures != 510 or not 0 <= invalid_geometry_failures <= failures:
        raise ValueError('Changed source denominator or invalid count')
    return {'invalid_geometry_failures': invalid_geometry_failures,
            'all_shortestness_failures': failures,
            'observed_percent': 100 * invalid_geometry_failures / failures,
            'minimum_percent': 90,
            'decision': 'supported' if 100 * invalid_geometry_failures >= 90 * failures else 'falsified'}
