"""
Validate MAP fitting method from paper [1].

[1] Horvath, G., Buchholz, P., & Telek, M. (2005). A MAP fitting approach
with independent approximation of the inter-arrival time distribution and
the lag correlation. Second International Conference on the Quantitative
Evaluation of Systems (QEST’05), 124–133.
https://doi.org/10.1109/QEST.2005.1
"""
from pytest import fixture
import numpy as np
from numpy.testing import assert_allclose

from pyqumo.fitting import optimize_lag1, fit_map_horvath05
from pyqumo.random import PhaseType


@fixture
def horvath_illustration_2():
    """
    Example PH from Illustration 2 [1] (p.4)
    """
    return {
        's': np.asarray([
            [-3.721, 0.5, 0.002],
            [0.1, -1.206, 0.005],
            [0.001, 0.002, -0.031]
        ]),
        'p': np.asarray([0.4995, 0.48918, 0.01132]),
    }


def test_optimize_lag1_boundaries(horvath_illustration_2):
    """
    Validate that optimize_lag1() with optype = 'min' and optype = 'max'
    produce expected results. Boundary values are taken from paper [1].
    """
    ph = PhaseType(horvath_illustration_2['s'], horvath_illustration_2['p'])
    lag1_min = optimize_lag1(ph, optype='min', n_iters=10).fun
    lag1_max = optimize_lag1(ph, optype='max', n_iters=10).fun
    # FIXME: relative error for MIN is about 50% (absolute is .004). TOO HIGH.
    assert_allclose(lag1_min, -0.01015, atol=.01, err_msg="Min. lag-1 failed")
    # FIXME: relative error for MAX is about 20% (absolute is .063). TOO HIGH.
    assert_allclose(lag1_max, 0.32645, atol=.09, err_msg="Max. lag-1 failed")


def test_fit_map_horvath05():
    """
    Validate that fit_map_horvath05() finds a MAP with valid autocorrelation.
    """
    s = np.asarray([
        [-4.35, 4.35, 0, 0, 0, 0],
        [0, -4.35, 4.35, 0, 0, 0],
        [0, 0, -4.35, 0, 0, 0],
        [0, 0, 0, -1.15, 1.15, 0],
        [0, 0, 0, 0, -1.15, 1.15],
        [0, 0, 0, 0, 0, -1.15]
    ])
    p = np.asarray([0.838, 0, 0, 0.162, 0, 0])

    ph = PhaseType(s, p)
    expected_lag1 = 0.129

    markov_arrival, errors = fit_map_horvath05(ph, expected_lag1, n_iters=10)
    actual_lag1 = markov_arrival.lag(1)

    assert_allclose(actual_lag1, expected_lag1, rtol=.05)
    assert_allclose(errors[0], abs(actual_lag1 - expected_lag1) / actual_lag1,
                    atol=.01, rtol=.01)

    assert_allclose(ph.mean, markov_arrival.mean, rtol=.01)
    assert_allclose(ph.cv, markov_arrival.cv, rtol=.01)
    assert_allclose(ph.skewness, markov_arrival.skewness, rtol=.01)
