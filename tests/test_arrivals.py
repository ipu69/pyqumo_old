from unittest.mock import patch

import pytest

import numpy as np
from numpy.testing import assert_allclose

from pyqumo import stats
from pyqumo.arrivals import MarkovArrival, Poisson, \
    GIProcess


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
    rate = 1 / m1
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


# class TestMAP(ut.TestCase):

#     def test_erlang_constructor(self):
#         m1 = ar.MAP.erlang(1, 1.0)
#         m2 = ar.MAP.erlang(2, 5.0)
#         m3 = ar.MAP.erlang(3, 10.0)

#         assert_allclose(m1.D0, [[-1.0]])
#         assert_allclose(m1.D1, [[1.0]])
#         assert_allclose(m2.D0, [[-5, 5], [0, -5]])
#         assert_allclose(m2.D1, [[0, 0], [5, 0]])
#         assert_allclose(m3.D0, [[-10, 10, 0], [0, -10, 10], [0, 0, -10]])
#         assert_allclose(m3.D1, [[0, 0, 0], [0, 0, 0], [10, 0, 0]])

#     def test_moments_like_erlang(self):
#         e1 = Erlang(1, 1.0)
#         e2 = Erlang(2, 5.0)
#         e3 = Erlang(3, 10.0)
#         m1 = ar.MAP.erlang(e1.shape, e1.rate)
#         m2 = ar.MAP.erlang(e2.shape, e2.rate)
#         m3 = ar.MAP.erlang(e3.shape, e3.rate)

#         for k in range(10):
#             self.assertAlmostEqual(m1.moment(k), e1.moment(k))
#             self.assertAlmostEqual(m2.moment(k), e2.moment(k))
#             self.assertAlmostEqual(m3.moment(k), e3.moment(k))

#     # noinspection PyTypeChecker

#     # noinspection PyTypeChecker
#     def test_call(self):
#         D0 = [
#             [-99.0,  0.0,   0.0,   0.0],
#             [0.0,  -99.0,  99.0,   0.0],
#             [0.0,    0.0, -0.01,   0.0],
#             [0.01,   0.0,   0.0, -0.01],
#         ]
#         D1 = [
#             [98.0, 1.00, 0.000, 0.000],
#             [0.00, 0.00, 0.000, 0.000],
#             [0.00, 0.00, 0.009, 0.001],
#             [0.00, 0.00, 0.000, 0.000],
#         ]
#         m = ar.MAP(D0, D1, check=True)
#         NUM_SAMPLES = 1
#         samples = [m() for _ in range(NUM_SAMPLES)]

#         self.assertEqual(len(samples), NUM_SAMPLES)
#         assert_allclose(np.mean(samples), m.mean(), rtol=0.1)
#         assert_allclose(np.std(samples), m.std(), rtol=0.1)
#         assert_allclose(np.var(samples), m.var(), rtol=0.1)
#         assert_allclose(stats.lag(samples, 2), [m.lag(1), m.lag(2)], rtol=0.1)
