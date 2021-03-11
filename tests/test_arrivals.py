from unittest.mock import patch

import pytest

import numpy as np
from numpy.testing import assert_allclose

from pyqumo import stats
from pyqumo.arrivals import MarkovArrival, Poisson, \
    GIProcess
from pyqumo.random import PhaseType


#
# POISSON PROCESS
# #######################
from pyqumo.random import Const, Uniform


@pytest.mark.parametrize('proc, m1, m2, m3, l1, string', [
    # Poisson process:
    (Poisson(1.0), 1, 2, 6, 0.0, '(Poisson: r=1)'),
    (Poisson(2.5), 0.4, 0.32, 0.384, 0.0, '(Poisson: r=2.5)'),
    # GI with uniform or constant distributions:
    (GIProcess(Const(3)), 3, 9, 27, 0, '(GI: f=(Const: value=3))'),
    (GIProcess(Uniform(2, 10)), 6, 124 / 3, 312, 0,
     '(GI: f=(Uniform: a=2, b=10))'),
    # MAP variants of Poisson or Erlang processes:
    (
        MarkovArrival.poisson(2.5),
        0.4, 0.32, 0.384, 0.0,
        '(MAP: d0=[[-2.5]], d1=[[2.5]])'
    ), (
        MarkovArrival.erlang(3, rate=4.2),
        0.714286, 0.680272, 0.809848, 0.0,
        '(MAP: d0=[[-4.2, 4.2, 0], [0, -4.2, 4.2], [0, 0, -4.2]], '
        'd1=[[0, 0, 0], [0, 0, 0], [4.2, 0, 0]])'
    )
])
def test__props(proc, m1, m2, m3, l1, string):
    """
    Validate basic statistical properties of the random process.
    """
    # Compute basic props:
    var = m2 - m1**2
    std = var**0.5
    cv = std / m1

    assert_allclose(proc.mean, m1, rtol=1e-5, err_msg=string)
    assert_allclose(proc.var, var, rtol=1e-5, err_msg=string)
    assert_allclose(proc.std, std, rtol=1e-5, err_msg=string)
    assert_allclose(proc.cv, cv, rtol=1e-5, err_msg=string)
    assert_allclose(proc.moment(1), m1, rtol=1e-5, err_msg=string)
    assert_allclose(proc.moment(2), m2, rtol=1e-5, err_msg=string)
    assert_allclose(proc.moment(3), m3, rtol=1e-5, err_msg=string)
    assert_allclose(proc.lag(1), l1, atol=1e-9, err_msg=string)

    # Validate that random generated sequence have expected mean and std:
    samples = proc(100000)
    assert_allclose(samples.mean(), m1, rtol=0.05, err_msg=string)
    assert_allclose(samples.std(), std, rtol=0.05, err_msg=string)

    # Validate string output:
    assert str(proc) == string


#
# Markovian Arrival Process
# ##############################
def test_map__props():
    d0 = [[-1, 0.5], [0.5, -1]]
    d1 = [[0, 0.5], [0.2, 0.3]]
    proc = MarkovArrival(d0, d1)
    assert_allclose(proc.generator, [[-1, 1], [0.7, -0.7]])
    assert_allclose(proc.d0, d0)
    assert_allclose(proc.d(0), d0)
    assert_allclose(proc.d1, d1)
    assert_allclose(proc.d(1), d1)
    assert_allclose(proc.d(0), d0)
    assert proc.order == 2

    # Test inverse of D0:
    assert_allclose(proc.d0n(-1), [[4/3, 2/3], [2/3, 4/3]])
    assert_allclose(proc.d0n(-2), [[20/9, 16/9], [16/9, 20/9]])

    # Test chains:
    assert_allclose(proc.ctmc.matrix, [[-1, 1], [0.7, -0.7]])
    assert_allclose(proc.dtmc.matrix, [[2/15, 13/15], [4/15, 11/15]])


def test_map__invalid_matrices_call_fix_markovian_process():
    d0 = np.asarray([[-0.9, -0.1], [0, -1]])
    d1 = np.asarray([[0, 1.1], [1., 0.]])
    with patch('pyqumo.arrivals.fix_markovian_arrival',
               return_value=((d0, d1), (0.1, 0.1))) as mock:
        _ = MarkovArrival(d0, d1, tol=0.2)
        mock.assert_called_once()


@pytest.mark.parametrize('d0,d1', [
    (
        [[-9.0,  0.0,  0.0,   0.0],
         [0.0,  -9.0,  9.0,   0.0],
         [0.0,   0.0, -0.1,   0.0],
         [0.1,   0.0,  0.0,  -0.1]],
        [[8.0, 1.0, 0.00, 0.00],
         [0.0, 0.0, 0.00, 0.00],
         [0.0, 0.0, 0.09, 0.01],
         [0.0, 0.0, 0.00, 0.00]]
    )
])
def test_map__sampling(d0, d1):
    NUM_SAMPLES = 100000
    proc = MarkovArrival(d0, d1)
    samples = proc(NUM_SAMPLES)

    assert len(samples) == NUM_SAMPLES

    assert_allclose(np.mean(samples), proc.mean, rtol=0.05)
    assert_allclose(np.std(samples), proc.std, rtol=0.05)
    assert_allclose(np.var(samples), proc.var, rtol=0.05)

    assert_allclose(
        stats.lag(samples, 2), [proc.lag(1), proc.lag(2)], rtol=0.05
    )


@pytest.mark.parametrize('ph', [
    PhaseType.exponential(5.0),
    PhaseType.hyperexponential([1, 5, 20], [0.5, 0.2, 0.3]),
    PhaseType.erlang(10, 0.2),
])
def test_map__build_from_ph(ph):
    """Validate MAP construction from PH distribution.
    """
    assert isinstance(ph, PhaseType)
    map_ = MarkovArrival.phase_type(ph.s, ph.p)

    assert isinstance(map_, MarkovArrival)
    assert_allclose(map_.mean, ph.mean)
    assert_allclose(map_.std, ph.std)
    assert_allclose(map_.skewness, ph.skewness)
    assert_allclose(map_.lag(1), 0.0, atol=1e-8)
