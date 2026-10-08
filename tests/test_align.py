import numpy as np

from voxpipe.tuning.align import cosine_cost, dtw_path


def brute_force_dtw_cost(cost):
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
    return acc[n, m]


def path_cost(cost, ia, ib):
    return cost[ia, ib].sum()


def test_identical_sequences_align_diagonally():
    feats = np.random.default_rng(0).normal(size=(30, 8))
    ia, ib = dtw_path(cosine_cost(feats, feats))
    np.testing.assert_array_equal(ia, np.arange(30))
    np.testing.assert_array_equal(ib, np.arange(30))


def test_stretched_sequence_maps_back():
    a = np.random.default_rng(1).normal(size=(20, 8))
    b = np.repeat(a, 2, axis=0)  # b is a slowed down 2x
    ia, ib = dtw_path(cosine_cost(a, b))
    np.testing.assert_array_equal(ia, ib // 2)


def test_path_is_monotonic_and_complete():
    rng = np.random.default_rng(2)
    ia, ib = dtw_path(cosine_cost(rng.normal(size=(25, 6)), rng.normal(size=(40, 6))))
    assert (ia[0], ib[0]) == (0, 0) and (ia[-1], ib[-1]) == (24, 39)
    steps = np.stack([np.diff(ia), np.diff(ib)], axis=1)
    assert {tuple(s) for s in steps} <= {(1, 1), (1, 0), (0, 1)}


def test_matches_brute_force_optimum():
    rng = np.random.default_rng(3)
    cost = rng.random((15, 22))
    ia, ib = dtw_path(cost)
    np.testing.assert_allclose(path_cost(cost, ia, ib), brute_force_dtw_cost(cost))


def test_cosine_cost_range():
    rng = np.random.default_rng(4)
    c = cosine_cost(rng.normal(size=(5, 4)), rng.normal(size=(7, 4)))
    assert c.shape == (5, 7)
    assert c.min() >= -1e-9 and c.max() <= 2 + 1e-9
