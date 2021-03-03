import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyqumo.random import Const, Exponential, Uniform, Normal, Erlang, \
    HyperExponential, PhaseType, Choice, SemiMarkovAbsorb, \
    MixtureDistribution, CountableDistribution, HyperErlang


#
# TESTING VARIOUS DISTRIBUTIONS STATISTIC AND ANALYTIC PROPERTIES
# ----------------------------------------------------------------------------
# First of all we test statistical and analytic properties:
# - moments, k = 1, 2, 3, 4
# - variance
# - standard deviation
# - PDF (probability density function) at 3-5 points grid
# - CDF (cumulative distribution function) at 3-5 points grid
# - random samples generation
#
# To valid random samples generation we draw a large number of samples
# (100'000) and compute this sample set mean and variance using NumPy. Then
# we compare these estimation with expected mean and standard variance.
#
# We also validate that distribution is nicely printed.
#
# Since all distributions extend `Distribution` class and implement its
# methods, e.g. properties `var`, `mean`, methods `_moment()` and `_eval()`,
# we need to specify only the distribution itself and expected values.
# The test algorithm is the same. To achieve this, we use PyTest
# parametrization and define only one common test with a large number of
# parameters.
#
# Some properties are specific to continuous or discrete distributions,
# e.g. PMF or PMF. To test them, we define separate tests.
#
# Each parameters tuple specify one test: distribution, expected moments
# (four values), a grid for CDF and expected string form
# of the distribution.
@pytest.mark.parametrize('dist, m1, m2, m3, m4, string, atol, rtol', [
    # Constant distribution:
    (Const(2), 2, 4, 8, 16, '(Const: value=2)', 1e-2, 2e-2),
    (Const(3), 3, 9, 27, 81, '(Const: value=3)', 1e-2, 2e-2),
    # Uniform distribution:
    (Uniform(0, 1), 0.5, 1/3, 1/4, 1/5, '(Uniform: a=0, b=1)', 1e-2, 2e-2),
    (Uniform(2, 10), 6, 124/3, 312, 2499.2, '(Uniform: a=2, b=10)', .01, .02),
    # Normal distribution:
    (Normal(0, 1), 0, 1, 0, 3, '(Normal: mean=0, std=1)', 1e-2, 2e-2),
    (
        Normal(1, 0.5),
        1, 1.25, 1.75, 2.6875,
        '(Normal: mean=1, std=0.5)',
        1e-2, 2e-2
    ),
    # Exponential distribution:
    (Exponential(1.0), 1, 2, 6, 24, '(Exp: rate=1)', 1e-2, 2e-2),
    (Exponential(2.0), 1/2, 1/2, 6/8, 24/16, '(Exp: rate=2)', 1e-2, 2e-2),
    # Erlang distribution:
    (Erlang(1, 1), 1, 2, 6, 24, '(Erlang: shape=1, rate=1)', 1e-2, 2e-2),
    (
        Erlang(5, param=2.5),
        2, 4.8, 13.44, 43.008,
        '(Erlang: shape=5, rate=2.5)',
        1e-2, 2e-2
    ),
    # Hyperexponential distribution
    (
        HyperExponential([1], [1]), 1, 2, 6, 24,
        '(HyperExponential: probs=[1], rates=[1])',
        1e-2, 2e-2
    ), (
        HyperExponential([2, 3, 4], probs=[0.5, 0.2, 0.3]),
        0.39167, 0.33194, 0.4475694, 0.837384,
        '(HyperExponential: probs=[0.5, 0.2, 0.3], rates=[2, 3, 4])',
        1e-2, 2e-2
    ),
    # Hypererlang distribution
    (
        HyperErlang([1], [1], [1]), 1, 2, 6, 24,
        '(HyperErlang: probs=[1], shapes=[1], params=[1])',
        1e-2, 2e-2
    ),
    (
        HyperErlang(params=[1, 5, 8], shapes=[3, 4, 5], probs=[.2, .5, .3]),
        1.1875, 2.941, 12.603, 72.795,
        '(HyperErlang: probs=[0.2, 0.5, 0.3], shapes=[3, 4, 5], '
        'params=[1, 5, 8])',
        1e-2, 1e-2,
    ),
    # Phase-type distribution
    (
        PhaseType.exponential(1.0),
        1, 2, 6, 24,
        '(PH: s=[[-1]], p=[1])',
        1e-2, 2e-2
    ), (
        PhaseType.erlang(shape=3, rate=4.2),
        0.714286, 0.680272, 0.809848, 1.15693,
        '(PH: '
        's=[[-4.2, 4.2, 0], [0, -4.2, 4.2], [0, 0, -4.2]], '
        'p=[1, 0, 0])',
        1e-2, 2e-2
    ), (
        PhaseType.hyperexponential(rates=[2, 3, 4], probs=[0.5, 0.2, 0.3]),
        0.39167, 0.33194, 0.4475694, 0.837384,
        '(PH: '
        's=[[-2, 0, 0], [0, -3, 0], [0, 0, -4]], '
        'p=[0.5, 0.2, 0.3])',
        1e-2, 2e-2
    ), (
        PhaseType(np.asarray([[-2, 1, 0.2], [0.5, -3, 1], [0.5, 0.5, -4]]),
                  np.asarray([0.5, 0.4, 0.1])),
        0.718362, 1.01114, 2.12112, 5.92064,
        '(PH: '
        's=[[-2, 1, 0.2], [0.5, -3, 1], [0.5, 0.5, -4]], '
        'p=[0.5, 0.4, 0.1])',
        1e-2, 2e-2
    ),
    # Choice (discrete) distribution
    (Choice([2]), 2, 4, 8, 16, '(Choice: values=[2], p=[1.0])', 1e-2, 2e-2),
    (
        Choice([3, 7, 5], weights=[1, 4, 5]),
        5.6, 33.0, 202.4, 1281.0,
        '(Choice: values=[3, 5, 7], p=[0.1, 0.5, 0.4])',
        1e-2, 2e-2
    ),
    # Semi-Markov Absorbing Process
    (
        SemiMarkovAbsorb(
            [[0]], [Exponential(1.0)], probs=0, num_samples=250000),
        1, 2, 6, 24,
        '(SemiMarkovAbsorb: trans=[[0, 1], [0, 1]], '
        'time=[(Exp: rate=1)], p0=[1])',
        1e-1, 1e-1
    ), (
        SemiMarkovAbsorb([[0]], [Normal(1, 0.5)], num_samples=250000),
        1, 1.25, 1.75, 2.6875,
        '(SemiMarkovAbsorb: trans=[[0, 1], [0, 1]], '
        'time=[(Normal: mean=1, std=0.5)], p0=[1])',
        1e-1, 1e-1
    ), (
        SemiMarkovAbsorb(
            trans=[[0, 0.5, 0.1], [1/6, 0, 1/3], [1/8, 1/8, 0]],
            time_dist=[Exponential(2), Exponential(3), Exponential(4)],
            probs=[0.5, 0.4, 0.1],
            num_samples=250000
        ),
        0.718362, 1.01114, 2.12112, 5.92064,
        '(SemiMarkovAbsorb: trans=['
        '[0, 0.5, 0.1, 0.4], [0.167, 0, 0.333, 0.5], '
        '[0.125, 0.125, 0, 0.75], [0, 0, 0, 1]'
        '], time=[(Exp: rate=2), (Exp: rate=3), (Exp: rate=4)], '
        'p0=[0.5, 0.4, 0.1])',
        1e-1, 1e-1
    ),
    # Mixture of constant distributions (choice):
    (
        MixtureDistribution(
            states=[Const(3), Const(7), Const(5)],
            weights=[1, 4, 5]
        ),
        5.6, 33.0, 202.4, 1281.0,
        '(Mixture: '
        'states=[(Const: value=3), (Const: value=7), (Const: value=5)], '
        'probs=[0.1, 0.4, 0.5])',
        1e-2, 2e-2,
    ),
    # Countable discrete distributions:
    (
        # Geom(0.25) (number of attempts to get 1 success),
        CountableDistribution(
            lambda k: 0.25 * 0.75**(k-1) if k > 0 else 0,
            precision=1e-9  # to get 1e-2 precision for m4: 1e-2^4 = 1e-8
        ),
        4, 28, 292, 4060,
        '(Countable: p=[0, 0.25, 0.188, 0.141, 0.105, ...], precision=1e-09)',
        0.01, 0.01
    ), (
        # Geom(0.25) (number of attempts to get 1 success), explicit moments
        CountableDistribution(
            lambda k: 0.25 * 0.75 ** (k - 1) if k > 0 else 0,
            precision=0.001,
            moments=[4, 28, 292, 4060]
        ),
        4, 28, 292, 4060,
        '(Countable: p=[0, 0.25, 0.188, 0.141, 0.105, ...], precision=0.001)',
        0.05, 0.05
    )
])
def test_common_props(dist, m1, m2, m3, m4, string, atol, rtol):
    """
    Validate common distributions properties: first four moments and repr.
    """
    var = m2 - m1**2
    std = var**0.5

    # Validate statistic properties:
    assert_allclose(dist.mean, m1, atol=atol, rtol=rtol, err_msg=string)
    assert_allclose(dist.var, var, atol=atol, rtol=rtol, err_msg=string)
    assert_allclose(dist.std, std, atol=atol, rtol=rtol, err_msg=string)
    assert_allclose(dist.moment(1), m1, atol=atol, rtol=rtol, err_msg=string)
    assert_allclose(dist.moment(2), m2, atol=atol, rtol=rtol, err_msg=string)
    assert_allclose(dist.moment(3), m3, atol=atol, rtol=rtol, err_msg=string)
    assert_allclose(dist.moment(4), m4, atol=atol, rtol=rtol, err_msg=string)

    # Validate that random generated sequence have expected mean and std:
    samples = dist(100000)
    assert_allclose(samples.mean(), m1, atol=atol, rtol=rtol, err_msg=string)
    assert_allclose(samples.std(), std, atol=atol, rtol=rtol, err_msg=string)

    # Validate string output:
    assert str(dist) == string


#
# VALIDATE CUMULATIVE DISTRIBUTION FUNCTIONS
# ------------------------------------------
@pytest.mark.parametrize('dist, grid', [
    # Constant distribution:
    (Const(2), [(1.9, 0), (2.0, 1), (2.1, 1)]),
    (Const(3), [(2, 0), (3, 1), (3.1, 1)]),
    # Uniform distribution:
    (Uniform(0, 1), [(-1, 0), (0, 0), (0.5, 0.5), (1, 1), (2, 1)]),
    (Uniform(2, 10), [(1, 0), (2, 0), (6, 0.5), (10, 1), (11, 1)]),
    # Normal distribution:
    (
        Normal(0, 1),
        [(-2, 0.023), (-1, 0.159), (0, 0.500), (1, 0.841), (2, 0.977)],
    ), (
        Normal(1, 0.5),
        [(0, 0.023), (1, 0.500), (1.25, 0.691), (1.5, 0.841), (2, 0.977)],
    ),
    # Exponential distribution:
    (Exponential(1.0), [(0, 0.0), (1, 1 - 1/np.e), (2, 0.865)]),
    (Exponential(2.0), [(0, 0.0), (1, 0.865), (2, 0.982)]),
    # Erlang distribution:
    (Erlang(1, 1), [(0, 0.000), (1, 0.632), (2, 0.865), (3, 0.950)]),
    (Erlang(5, param=2.5), [(0, 0.000), (1, 0.109), (2, 0.559), (3, 0.868)]),
    # Hyperexponential distribution
    (
        HyperExponential([1], [1]),
        [(0, 0.000), (1, 0.632), (2, 0.865), (3, 0.950)]
    ), (
        HyperExponential(rates=[2, 3, 4], probs=[0.5, 0.2, 0.3]),
        [(0, 0.000), (0.25, 0.492), (0.5, 0.731), (1, 0.917)]
    ),
    # Hyper-Erlang distribution
    (
        HyperErlang([1], [1], [1]),
        [(0, 0.000), (1, 0.632), (2, 0.865), (3, 0.950)]
    ), (
        HyperErlang(params=[1, 5, 8], shapes=[3, 4, 5], probs=[.2, .5, .3]),
        [(0, 0.000), (0.25, 0.035), (1, 0.654), (1.5, 0.806)]
    ),
    # Phase-type distribution
    (
        PhaseType.exponential(1.0),
        [(0, 0.000), (1, 0.632), (2, 0.865), (3, 0.950)]
    ), (
        PhaseType.erlang(shape=3, rate=4.2),
        [(0, 0.00), (0.5, 0.350), (1, 0.790), (1.5, 0.950)]
    ), (
        PhaseType.hyperexponential(rates=[2, 3, 4], probs=[0.5, 0.2, 0.3]),
        [(0, 0.000), (0.25, 0.492), (0.5, 0.731), (1, 0.917)]
    ), (
        PhaseType(np.asarray([[-2, 1, 0.2], [0.5, -3, 1], [0.5, 0.5, -4]]),
                  np.asarray([0.5, 0.4, 0.1])),
        [(0, 0.00), (0.5, 0.495), (1, 0.752), (1.5, 0.879)]
    ),
    # Choice (discrete) distribution
    (Choice([5]), [(4.9, 0), (5.0, 1), (5.1, 1)]),
    (
        Choice([3, 5, 7], weights=[1, 5, 4]),
        [(2, 0), (3, 0.1), (4.9, 0.1), (5, 0.6), (6.9, 0.6), (7, 1), (8, 1)]
    ),
    # Countable discrete distributions:
    (
        # Geom(0.25) (number of attempts to get 1 success),
        CountableDistribution(
            lambda k: 0.25 * 0.75 ** (k - 1) if k > 0 else 0,
            precision=1e-1  # to get 1e-2 precision for m4: 1e-2^4 = 1e-8
        ),
        [(0, 0), (1, 0.25), (2, 0.437), (3, 0.578), (4, 0.684), (5, 0.763)]
    ),
])
def test_cdf(dist, grid):
    """
    Validate cumulative distribution function.
    """
    cdf = dist.cdf
    for x, y in grid:
        assert_allclose(cdf(x), y, atol=1e-3, err_msg=f'{dist} CDF, x={x}')


@pytest.mark.parametrize('dist, grid', [
    (
        SemiMarkovAbsorb([[0]], [Exponential(2.0)], probs=0,
                         num_kde_samples=20000),
        [(0.1, 0.181), (0.5, 0.632), (1.0, 0.865), (1.4, 0.939)]
    )
])
def test_gaussian_kde_cdf(dist, grid):
    """
    Validate cumulative distribution function when it is evaluated from
    sample data using Gaussian kernel.
    """
    cdf = dist.cdf
    for x, y in grid:
        assert_allclose(cdf(x), y, rtol=0.1, err_msg=f'{dist} CDF, x={x}')


#
# VALIDATE PROBABILITY DENSITY FUNCTIONS (CONT. DIST.)
# ----------------------------------------------------
@pytest.mark.parametrize('dist, grid', [
    # Constant distribution:
    (Const(2), [(1.9, 0), (2.0, np.inf), (2.1, 0)]),
    (Const(3), [(2, 0), (3, np.inf), (3.1, 0)]),
    # Uniform distribution:
    (Uniform(0, 1), [(-1, 0), (0, 1), (0.5, 1), (1, 1), (2, 0)]),
    (Uniform(2, 10), [(1, 0), (2, 0.125), (6, 0.125), (10, 0.125), (11, 0)]),
    # Normal distribution:
    (Normal(0, 1), [(-2, 0.054), (-1, 0.242), (0, 0.399), (1, 0.242)]),
    (Normal(1, 0.5), [(0, 0.108), (1, 0.798), (1.25, 0.704), (2, 0.108)]),
    # Exponential distribution:
    (Exponential(1.0), [(0, 1.0), (1, 1/np.e), (2, 0.135)]),
    (Exponential(2.0), [(0, 2.0), (1, 0.271), (2, 0.037)]),
    # Erlang distribution:
    (Erlang(1, 1), [(0, 1.000), (1, 0.368), (2, 0.135), (3, 0.050)]),
    (Erlang(5, param=2.5), [(0, 0.000), (1, 0.334), (2, 0.439), (3, 0.182)]),
    # Hyperexponential distribution
    (
        HyperExponential([1], [1]),
        [(0, 1.000), (1, 0.368), (2, 0.135), (3, 0.050)]
    ), (
        HyperExponential(rates=[2, 3, 4], probs=[0.5, 0.2, 0.3]),
        [(0, 2.800), (0.25, 1.331), (0.5, 0.664), (1, 0.187)]
    ),
    # Hyper-Erlang distribution
    (
        HyperErlang([1], [1], [1]),
        [(0, 1.000), (1, 0.368), (2, 0.135), (3, 0.050)]
    ),
    (
        HyperErlang(params=[1, 5, 8], shapes=[3, 4, 5], probs=[.2, .5, .3]),
        [(0, 0), (0.25, 0.454), (1, 0.525), (1.5, 0.160)]
    ),
    # Phase-type distribution
    (
        PhaseType.exponential(1.0),
        [(0, 1.000), (1, 0.368), (2, 0.135), (3, 0.050)]
    ), (
        PhaseType.erlang(shape=3, rate=4.2),
        [(0, 0.00), (0.5, 1.134), (1, 0.555), (1.5, 0.153)]
    ), (
        PhaseType.hyperexponential(rates=[2, 3, 4], probs=[0.5, 0.2, 0.3]),
        [(0, 2.800), (0.25, 1.331), (0.5, 0.664), (1, 0.187)],  # PDF
    ), (
        PhaseType(np.asarray([[-2, 1, 0.2], [0.5, -3, 1], [0.5, 0.5, -4]]),
                  np.asarray([0.5, 0.4, 0.1])),
        [(0, 1.30), (0.5, 0.710), (1, 0.355), (1.5, 0.174)]
    ),
])
def test_pdf(dist, grid):
    """
    Validate continuous distribution probability density function.
    """
    pdf = dist.pdf
    for x, y in grid:
        assert_allclose(pdf(x), y, atol=1e-3, err_msg=f'{dist} PDF, x={x}')


@pytest.mark.parametrize('dist, grid', [
    (
        SemiMarkovAbsorb([[0]], [Exponential(2.0)], probs=0,
                         num_kde_samples=20000),
        [(0.1, 1.637), (0.5, 0.736), (1.0, 0.271), (1.4, 0.122), (2.0, 0.0366)]
    )
])
def test_gaussian_kde_pdf(dist, grid):
    """
    Validate probability density function when it is evaluated from
    sample data using Gaussian kernel.
    """
    pdf = dist.pdf
    for x, y in grid:
        assert_allclose(pdf(x), y, rtol=0.15, err_msg=f'{dist} PDF, x={x}')


#
# VALIDATE PROBABILITY MASS FUNCTIONS AND ITERATORS (DISCRETE DIST.)
# ------------------------------------------------------------------
@pytest.mark.parametrize('dist, grid', [
    # Constant distribution:
    (Const(2), [(2, 1.0)]),
    (Const(3), [(3, 1.0)]),
    # Choice (discrete) distribution:
    (Choice([10]), [(10, 1.0)]),
    (Choice([5, 7, 9], weights=[1, 5, 4]), [(5, 0.1), (7, 0.5), (9, 0.4)]),
    # Countable discrete distributions:
    (
        # Geom(0.25) (number of attempts to get 1 success),
        CountableDistribution(
            lambda k: 0.25 * 0.75 ** (k - 1) if k > 0 else 0,
            precision=1e-1  # to get 1e-2 precision for m4: 1e-2^4 = 1e-8
        ),
        [(0, 0), (1, 0.25), (2, 0.1875), (3, 0.1406), (4, 0.1055), (5, 0.0791)]
    ),
])
def test_pmf_and_iterators(dist, grid):
    """
    Validate discrete distribution probability mass function and iterator.
    """
    pmf = dist.pmf
    for x, y in grid:
        assert_allclose(pmf(x), y, atol=1e-3, err_msg=f'{dist} PMF, x={x}')
    for i, (desired, actual) in enumerate(zip(grid, dist)):
        assert_allclose(actual[0], desired[0],
                        err_msg=f'{i}-th values mismatch, dist: {dist}')
        assert_allclose(actual[1], desired[1], rtol=1e-3,
                        err_msg=f'{i}-th probability mismatch, dist: {dist}')


#
# CUSTOM PROPERTIES OF SIMPLE DISTRIBUTIONS
# ----------------------------------------------------------------------------
@pytest.mark.parametrize('choice, value, expected_index, comment', [
    (Choice([1]), 1, 0, 'choice of length 1, search for existing value'),
    (Choice([1]), 0, -1, 'choice of length 1, search for too small value'),
    (Choice([1]), 2, 0, 'choice of length 1, search for large value'),
    (Choice([1, 2]), 2, 1, 'choice of length 2, search for existing value'),
    (Choice([1, 2]), 0, -1, 'choice of length 2, search for too small value'),
    (Choice([1, 2]), 1.5, 0, 'choice of length 2, s.f. value in the middle'),
    (Choice([10, 20, 30, 40, 50]), 10, 0, 'choice len. 5, existing value #1'),
    (Choice([10, 20, 30, 40, 50]), 30, 2, 'choice len. 5, existing value #2'),
    (Choice([10, 20, 30, 40, 50]), 40, 3, 'choice len. 5, existing value #3'),
    (Choice([10, 20, 30, 40, 50]), 9, -1, 'choice len. 5, too small value'),
    (Choice([10, 20, 30, 40, 50]), 51, 4, 'choice len. 5, too large value'),
    (Choice([10, 20, 30, 40, 50]), 22, 1, 'choice len. 5, val inside'),
])
def test_choice_find_left(choice, value, expected_index, comment):
    assert choice.find_left(value) == expected_index, comment


@pytest.mark.parametrize('fn, precision, truncated_at', [
    (lambda k: 0.25 * 0.75**(k-1) if k > 0 else 0, 0.01, 17),
    (lambda k: 0.1 * 0.9**k, 0.1, 21),
])
def test_countable_distribution_with_fn_props(fn, precision, truncated_at):
    """
    Validate that CountableDistribution stops when tail probability is less
    then precision.
    """
    dist = CountableDistribution(fn, precision=precision)
    assert dist.truncated_at == truncated_at, str(dist)


def test_countable_distribution_with_fn_and_max_value_ignores_large_values():
    """
    Validate that if probability is given in functional form, but max_value
    is provided, then any value above it is ignored.
    """
    pmf = [0.2, 0.3, 0.5]
    touched = [False] * 10  # elements after 2 should not be touched!

    def prob(x: int) -> float:
        touched[x] = True
        return pmf[x] if 0 <= x < len(pmf) else 0.0

    dist = CountableDistribution(prob, max_value=2)  # probs given to 0, 1, 2
    assert_allclose(dist.pmf(0), pmf[0])
    assert_allclose(dist.pmf(1), pmf[1])
    assert_allclose(dist.pmf(2), pmf[2])
    assert all(touched[:3])
    assert not any(touched[3:])

    # Now ask for cdf at, say, 4, and make sure no elements above 3 touched:
    assert_allclose(dist.cdf(4), 1.0)
    assert not any(touched[3:]), "touched items >= 3 in CDF call"

    # Also ask for PMF at even larger element, and make sure no items touched:
    assert_allclose(dist.pmf(9), 0.0)
    assert not any(touched[3:]), "touched items >= 3 in PMF call"


@pytest.mark.parametrize('pmf, string', [
    ([1.0], '(Countable: p=[1])'),
    ([0.5, 0.5], '(Countable: p=[0.5, 0.5])'),
    ([0.2, 0.5, 0.1, 0.2], '(Countable: p=[0.2, 0.5, 0.1, 0.2])')
])
def test_countable_distribution_with_pmf_props(pmf, string):
    """
    Validate that CountableDistribution can be defined with PMF.
    """
    dist = CountableDistribution(pmf)
    assert dist.truncated_at == len(pmf) - 1
    assert dist.max_value == len(pmf) - 1

    # Validate values:
    for i, prob in enumerate(pmf):
        assert_allclose(dist.pmf(i), prob,
                        err_msg=f"PMF mismatch at i={i} (PMF: {pmf})")

    # Validate CDF values:
    cdf = np.cumsum(pmf)
    for i, prob in enumerate(cdf):
        assert_allclose(dist.cdf(i), prob,
                        err_msg=f"CDF mismatch at i={i} (PMF: {pmf})")

    # Validate points outside the PMF:
    assert_allclose(dist.pmf(-1), 0.0)
    assert_allclose(dist.pmf(len(pmf) + 1), 0.0)
    assert_allclose(dist.cdf(-1), 0.0)
    assert_allclose(dist.cdf(len(pmf) + 1), 1.0)

    # Validate converting to string:
    assert str(dist) == string


@pytest.mark.parametrize('avg', [1.0, 2.0, 10.0, 0.3])
def test_exponential_fit(avg):
    dist = Exponential.fit(avg)
    assert_allclose(dist.mean, avg)


@pytest.mark.parametrize('avg, std, param, shape', [
    (1, 1, 1, 1), (2, 2, 0.5, 1), (10, 5, 0.4, 4),
    (5, 10, 0.2, 1)  # bad values - use Poisson (Erlang representation of)
])
def test_erlang_fit(avg, std, param, shape):
    dist = Erlang.fit(avg, std)
    assert dist.shape == shape
    assert_allclose(dist.param, param)
    assert_allclose(dist.mean, avg)


@pytest.mark.parametrize('avg, std, skew, order', [
    (1, 1, 0, 1),
    (1, 5, 0, 2),
    (8, 10, 0, 2),
    (0.1, 0.3, 0, 2),
])
def test_hyperexponential_fit(avg, std, skew, order):
    dist = HyperExponential.fit(avg, std, skew)
    assert_allclose(dist.mean, avg)
    assert_allclose(dist.std, std)
    assert dist.order == order


#
# VALIDATE CONVERTING TO PH (as_ph()) METHODS
# ----------------------------------------------------------------------------
@pytest.mark.parametrize('dist, s, p', [
    (Exponential(2.0), [[-2.0]], [1.0]),
    (Erlang(3, 7.0), [[-7, 7, 0], [0, -7, 7], [0, 0, -7]], [1, 0, 0]),
    (HyperExponential([3.4, 4.2], [.3, .7]), [[-3.4, 0], [0, -4.2]], [.3, .7]),
    (
        HyperErlang([3.4, 4.2], [2, 3], [.2, .8]),
        [[-3.4, 3.4, 0.0, 0.0, 0.0],
         [0.0, -3.4, 0.0, 0.0, 0.0],
         [0.0, 0.0, -4.2, 4.2, 0.0],
         [0.0, 0.0, 0.0, -4.2, 4.2],
         [0.0, 0.0, 0.0, 0.0, -4.2]],
        [.2, 0, .8, 0, 0]
    ),
    # The following distribution will be casted since non-markovian
    # state has very small probability < 1e-5:
    (
        MixtureDistribution([Exponential(1), Normal(10, 2)], [9.999999, 1e-6]),
        [[-1]], [1.0]
    )
])
def test__as_ph(dist, s, p):
    comment = repr(dist)
    ph = dist.as_ph()
    assert_allclose(
        ph.s, s,
        err_msg=f"PH matrix mismatch for {comment}")
    assert_allclose(
        ph.init_probs, p,
        err_msg=f"PH probs mismatch for {comment}")


def test__as_ph__mixture_with_non_markovian_state_raise_runtime_error():
    """
    Validate that MixtureDistribution.as_ph() raises ValueError if one of
    the states is not Markovian (e.g., normal or uniform distribution).
    """
    dist = MixtureDistribution(
        [Exponential(1.0), Exponential(8.0), Normal(1.0, 0.5)],
        weights=[0.495, 0.495, 0.01])
    with pytest.raises(RuntimeError):
        dist.as_ph()
    # However, converting this distribution to PH with min_prob=0.1 is OK:
    ph = dist.as_ph(min_prob=0.1)
    assert_allclose(ph.s, [[-1, 0], [0, -8]])
    assert_allclose(ph.p, [0.5, 0.5])
