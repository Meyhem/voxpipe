"""Dynamic time warping between reference and target feature sequences."""

from __future__ import annotations

import numpy as np


def cosine_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    def normalise(x: np.ndarray) -> np.ndarray:
        centred = x - x.mean(axis=1, keepdims=True)
        return centred / (np.linalg.norm(centred, axis=1, keepdims=True) + 1e-9)

    return 1.0 - normalise(a) @ normalise(b).T


def dtw_path(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classic DTW. Each row is computed vectorised: with v[k] the best arrival into
    (i, k) from row i-1 and S the row's cumulative cost, horizontal moves give
    acc[i, j] = S[j] + min_{k<=j}(v[k] - S[k])."""
    n, m = cost.shape
    acc = np.empty((n, m))
    acc[0] = np.cumsum(cost[0])
    for i in range(1, n):
        prev = acc[i - 1]
        diagonal = np.concatenate(([np.inf], prev[:-1]))
        arrival = cost[i] + np.minimum(prev, diagonal)
        running = np.cumsum(cost[i])
        acc[i] = running + np.minimum.accumulate(arrival - running)

    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = int(np.argmin((acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])))
            i, j = (i - 1, j - 1) if step == 0 else (i - 1, j) if step == 1 else (i, j - 1)
        path.append((i, j))
    path.reverse()
    indices = np.array(path)
    return indices[:, 0], indices[:, 1]
