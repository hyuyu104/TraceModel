import numpy as np
import pandas as pd
import pytest
from traceHMM.utils.func import stationary_dist


def test_stationary_dist1():
    P = np.array([
        [0.5, 0.5, 0, 0],
        [0.5, 0.5, 0, 0],
        [0, 0, 0.5, 0.5],
        [0, 0, 0.5, 0.5]
    ])
    with pytest.warns(UserWarning) as record:
        stationary_dist(P)
        if not record:
            pytest.fail(
                "stationary_dist failed to throw warnings when there are " +
                "multiple stationary distributions."
            )
            

def test_stationary_dist2():
    for i in range(5):
        np.random.seed(i)
        P = np.random.uniform(0, 1, (4, 4))
        P = P/P.sum(axis=1)[:, None]
        pi, prev = np.ones(4)/4, None
        while prev is None or np.linalg.norm(prev - pi) > 1e-8:
            prev = pi
            pi = pi @ P
        mu = stationary_dist(P)
        assert np.mean(np.abs(pi - mu)) < 1e-5
        