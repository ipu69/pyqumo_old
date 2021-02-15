from unittest.mock import patch, Mock

import numpy as np
from numpy.testing import assert_allclose
import pytest

from pyqumo.chains import DiscreteTimeMarkovChain, ContinuousTimeMarkovChain
from pyqumo.errors import CellValueError, RowSumError, MatrixShapeError


# ############################################################################
# TEST DiscreteTimeMarkovChain
# ############################################################################
@pytest.mark.parametrize('matrix, order, string', [
    ([[1.0]], 1, '(DTMC: t=[[1]])'),
    ([[0.5, 0.5], [0.8, 0.2]], 2, '(DTMC: t=[[0.5, 0.5], [0.8, 0.2]])')
])
def test_dtmc__props(matrix, order, string):
    chain = DiscreteTimeMarkovChain(matrix)
    assert_allclose(chain.matrix, matrix)
    assert chain.order == order
    assert str(chain) == string


@pytest.mark.parametrize('matrix, tol, exc_type', [
    ([[0.9]], 0.05, RowSumError),
    ([[-0.0005, 0.9995], [0.5, 0.5]], 1e-4, CellValueError),
    ([[0.5, 0.5]], 0.1, MatrixShapeError)
])
def test_dtmc__bad_matrix_raise_error(matrix, tol, exc_type):
    with pytest.raises(exc_type):
        DiscreteTimeMarkovChain(matrix, tol=tol)


def test_dtmc__bad_matrix_fix():
    """
    Validate matrix is checked by default, if errors are not more then `tol`.
    """
    matrix = np.asarray([
        [-0.08,  0.02, 0.72, 0.12],
        [-0.04, -0.01, 0.03, 0.76],
        [1.09, 0.00, 0.02, 0.00],
        [0.40, 0.40, 0.00, 0.20]
    ])
    tol = 0.12
    with patch('pyqumo.chains.fix_stochastic', return_value=matrix) as fix:
        DiscreteTimeMarkovChain(matrix, tol=tol)
        fix.assert_called_once()
        assert_allclose(fix.call_args[0][0], matrix)
        assert_allclose(fix.call_args[1]['tol'], tol)
        #
        # fix.assert_called_once_with(matrix, tol=tol)


def test_dtmc__bad_matrix_no_raise_when_safe_true():
    """
    Validate that no exception is raised if matrix is not stochastic, but
    `safe = True`.
    """
    mat = [[-0.2, 1.1], [0.4, 0.4]]  # very bad matrix
    chain = DiscreteTimeMarkovChain(mat, safe=True, tol=0.01)
    assert_allclose(mat, chain.matrix)


def test_dtmc__matrix_copied():
    """
    Validate that if matrix is passed as an array, it is copied, so changes
    in the argument don't affect the chain matrix.
    """
    matrix = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    chain = DiscreteTimeMarkovChain(matrix)
    matrix[0, 0] = 0.42
    assert chain.matrix[0, 0] == 0.5


#
# Testing steady-state PMF and traces.
# ------------------------------------

# DTMC FIXTURES:
# --------------
# To test these properties, we will use three fixtures:
# - dtmc1: chain with single (absorbing) state
# - dtmc2: periodic chain with interchanging states 0-1-0-1...
# - dtmc9: large non-trivial chain
# - dtmc4_line: a DTMC with sequential deterministic transitions
#     0 -> 1 -> 2 -> 3 -> 3 -> ... (3 - absorbing state)
# - dtmc4_circle: a periodic DTMC with sequential deterministic transitions:
#     0 -> 1 -> 2 -> 3 -> 0 -> ... (no absorbing states)
#
def dtmc1():
    return DiscreteTimeMarkovChain([[1.0]])


def dtmc2():
    return DiscreteTimeMarkovChain([[0, 1], [1, 0]])


def dtmc9():
    matrix = [
        [.00, .50, .00, .50, .00, .00, .00, .00, .00],
        [.25, .00, .25, .00, .25, .00, .00, .25, .00],
        [.00, .50, .00, .00, .00, .50, .00, .00, .00],
        [1/3, .00, .00, .00, 1/3, .00, 1/3, .00, .00],
        [.00, .25, .00, .25, .00, .25, .00, .25, .00],
        [.00, .00, 1/3, .00, 1/3, .00, .00, .00, 1/3],
        [.00, .00, .00, .50, .00, .00, .00, .50, .00],
        [.00, .25, .00, .00, .25, .00, .25, .00, .25],
        [.00, .00, .00, .00, .00, .50, .00, .50, .00]]
    return DiscreteTimeMarkovChain(matrix)


def dtmc4_line():
    matrix = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]]
    return DiscreteTimeMarkovChain(matrix)


def dtmc4_circle():
    matrix = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    return DiscreteTimeMarkovChain(matrix)


# (end of fixtures)
# -----------------

# Testing DiscreteTimeMarkovChain.steady_pmf
# ------------------------------------------

@pytest.mark.parametrize('chain, pmf, comment', [
    (dtmc1(), [1.0], 'trivial DTMC of order 1 PMF is also trivial'),
    (dtmc2(), [0.5, 0.5], 'periodic DTMC of order 2 states are equivalent'),
    (
        dtmc9(), [.077, .154, .077, .115, 0.154, 0.115, 0.077, 0.154, 0.077],
        'DTMC of order 9 with non-trivial matrix and steady-state PMF'
    )
])
def test_dtmc__steady_pmf(chain: DiscreteTimeMarkovChain, pmf, comment):
    """
    Validate steady state PMF evaluation of a discrete time Markov chain.
    """
    assert_allclose(chain.steady_pmf, pmf, rtol=0.01, err_msg=comment)


# Testing DiscreteTimeMarkovChain.trace()
# ---------------------------------------

@pytest.mark.parametrize('chain, init, path', [
    (dtmc2(), 0, (0, 1, 0, 1, 0, 1, 0, 1)),
    (dtmc2(), 1, (1, 0, 1, 0)),
    (dtmc4_circle(), 2, (2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0)),
])
def test_dtmc__trace_from_concrete_state(chain, init, path):
    """
    Validate dtmc trace() method call with a specific init state.
    """
    size = len(path) - 1
    trace = chain.trace(size, init=init)
    for i in range(size):
        step = next(trace)
        assert step[0] == path[i], f"step #{i} = {step}, expected path: {path}"


def test_dtmc__trace_generates_exactly_size_steps():
    """
    Validate trace() call generates a trace with a given number of steps.
    """
    chain = dtmc2()
    assert len(list(chain.trace(7))) == 7


@pytest.mark.parametrize('chain, size, init, ends, path, comment', [
    (dtmc2(), 10, 0, (0,), (), 'if initial state in ends, return empty trace'),
    (dtmc2(), 10, 0, (1,), ((0, 1),), 'path with single step till end'),
    (dtmc4_line(), 2, 0, (3,), ((0, 1), (1, 2)), 'trace stop before ends'),
    (dtmc4_line(), 8, 0, (3,), ((0, 1), (1, 2), (2, 3)), 'path of size 3'),
])
def test_dtmc__trace_till_ends(chain, size, init, ends, path, comment):
    """
    Validate trace() call stops when reaches specified `ends` vertices.
    """
    assert isinstance(chain, DiscreteTimeMarkovChain)
    real_path = tuple(chain.trace(size, init=init, ends=ends))
    assert path == real_path, comment


@pytest.mark.parametrize('chain, pmf, comment', [
    (dtmc2(), [0.2, 0.8], 'DTMC-2 with non-equal initial probabilities'),
    (dtmc2(), None, 'DTMC-2 with steady-state probability [0.5, 0.5]'),
    (dtmc9(), [0.1, 0.3, 0.2, 0.0, 0.0, 0.1, 0.2, 0.1, 0.0], 'DTMC-9'),
])
def test_dtmc__trace_from_pmf(chain, pmf, comment):
    """
    Validate dtmc trace() method starting from either given PMF, or
    from steady state PMF (if `pmf = None`). To validate that, we call trace()
    multiple times and measure rates at which states were chosen.
    """
    NUM_RUNS = 10000
    expected_pmf = chain.steady_pmf.copy() if pmf is None else np.asarray(pmf)
    hits = np.zeros(chain.order)
    for _ in range(NUM_RUNS):
        trace = chain.trace(1, init=pmf)
        transition = next(trace)
        state = transition[0]
        hits[state] += 1
    # Estimate probabilities:
    est_pmf = hits / NUM_RUNS
    # Validate:
    assert_allclose(est_pmf, expected_pmf, rtol=0.1, err_msg=comment)


@pytest.mark.parametrize('chain, comment', [
    (dtmc2(), 'Periodic DTMC-2'),
    (dtmc9(), 'Arbitrary DTMC-9'),
])
def test_dtmc__trace_visits_states_as_steady_pmf(chain, comment):
    """
    Validate that on a long run trace visits state approximately with steady
    PMF ratio. Trace selects initial state with steady state PMF distribution.
    """
    assert isinstance(chain, DiscreteTimeMarkovChain)
    TRACE_SIZE = 10000

    hits_prev = np.zeros(chain.order)
    hits_next = np.zeros(chain.order)
    for step in chain.trace(TRACE_SIZE):
        hits_prev[step[0]] += 1
        hits_next[step[1]] += 1
    hits_prev = hits_prev / TRACE_SIZE
    hits_next = hits_next / TRACE_SIZE

    assert_allclose(hits_prev, chain.steady_pmf, rtol=5e-2, atol=0.01,
                    err_msg=f"{comment} (hits_prev)")
    assert_allclose(hits_next, chain.steady_pmf, rtol=5e-2, atol=0.01,
                    err_msg=f"{comment} (hits_next)")


# Testing DiscreteTimeMarkovChain.random_path()
# ---------------------------------------------
@pytest.mark.parametrize('chain, path, comment', [
    (dtmc1(), (0, 0, 0), 'degenerate case of DTMC-1'),
])
def test_dtmc__random_path(chain, path, comment):
    """
    Validate DTMC `random_path()` method by calling it from a chain with
    completely determined transition matrix.
    """
    size = len(path) - 1
    real_path = chain.random_path(size, init=path[0])
    assert real_path == path, comment


# ############################################################################
# TEST DiscreteTimeMarkovChain
# ############################################################################
@pytest.mark.parametrize('matrix, order, dtmc_matrix, string', [
    ([[0.0]], 1, [[1.0]], '(CTMC: g=[[0]])'),
    ([[-1, 1], [5, -5]], 2, [[0, 1], [1, 0]], '(CTMC: g=[[-1, 1], [5, -5]])'),
    (
        [[-2, 1.5, 0.5], [3, -3, 0], [0, 0, 0]],
        3,
        [[0, 0.75, 0.25], [1, 0, 0], [0, 0, 1]],
        '(CTMC: g=[[-2, 1.5, 0.5], [3, -3, 0], [0, 0, 0]])'
    ),
])
def test_ctmc__props(matrix, order, dtmc_matrix, string):
    """
    Validate CTMC creation and basic properties
    """
    chain = ContinuousTimeMarkovChain(matrix)
    assert_allclose(chain.matrix, matrix, err_msg=string)
    assert chain.order == order, string
    assert str(chain) == string, string
    assert_allclose(chain.embedded_dtmc.matrix, dtmc_matrix, err_msg=string)


@pytest.mark.parametrize('matrix, tol, exc_type', [
    (
        [[-1.0, -0.101, 1.0], [1.0, -1.0, -0.11], [0, -0.09, 0]],
        0.1, CellValueError
    ),
    ([[-1.21, 1.0], [5, -5.23]], 0.2, RowSumError),
    ([[-1, 1, 0], [0, 0, 0]], 0.1, MatrixShapeError)
])
def test_ctmc__bad_matrix_raise_error(matrix, tol, exc_type):
    """
    Validate CTMC creation raises MatrixError if matrix is not infinitesimal.
    """
    with pytest.raises(exc_type):
        ContinuousTimeMarkovChain(matrix)


def test_ctmc__bad_matrix_fix():
    """
    Validate that matrix is fixed if its maximum error is less then tolerance.
    Here we use the same data as in fix_stochastic() call.
    (In fact, this can be mocked)
    """
    matrix = np.asarray([
        [-2.0, 1.5, 0.5, 0.0, 0.0],    # good row, no modifications
        [-0.1, -0.9, 0.4, 0.5, 0.0],   # set -0.1 to 0.0, err = 0.1
        [0.0, 2.0, -3.0, 1.2, 0.0],    # set diagonal to -3.2, err = 0.2
        [0.5, -0.15, 3.4, -4.0, 0.0],  # -0.15 => 0.0 -4.0 => -3.9, err = 0.15
        [0.0, 0.0, 0.0, 0.0, 0.0]      # good row, no modifications
    ])
    tol = 0.21
    with patch('pyqumo.chains.fix_infinitesimal', return_value=matrix) as fix:
        ContinuousTimeMarkovChain(matrix, tol=tol)
        fix.assert_called_once()
        assert_allclose(fix.call_args[0][0], matrix)
        assert_allclose(fix.call_args[1]['tol'], tol)


def test_ctmc__bad_matrix_no_raise_when_safe_true():
    """
    Validate no exception raised when safe = True passed to CTMC constructor.
    """
    matrix = [[-2, 1, 0], [1, -1, 1], [0, 0, 0.5]]  # very bad matrix
    chain = ContinuousTimeMarkovChain(matrix, safe=True, tol=0.01)
    assert_allclose(matrix, chain.matrix)


def test_ctmc__matrix_copied():
    """
    Validate that if matrix is passed as an array, it is copied, so changes
    in the argument don't affect the chain matrix.
    """
    matrix = np.asarray([[-1, 1], [1, -1]])
    chain = ContinuousTimeMarkovChain(matrix)
    matrix[0, 0] = -42
    assert chain.matrix[0, 0] == -1


#
# Testing steady-state PMF and traces of CTMC
# -------------------------------------------

# CTMC FIXTURES:
# --------------
# To test these properties, we will use three fixtures:
# - dtmc1: chain with single (absorbing) state
# - dtmc2: periodic chain with interchanging states 0-1-0-1...
# - dtmc3: non-trivial non-periodic chain with three states
#
def ctmc1():
    return ContinuousTimeMarkovChain([[0.0]])


def ctmc2():
    return ContinuousTimeMarkovChain([[-1, 1], [1, -1]])


def ctmc3():
    return ContinuousTimeMarkovChain(
        [[-.025, .02, .005], [.3, -.5, .2], [.02, .4, -.42]]
    )


# (end of fixtures)
# -----------------

# Testing ContinuousTimeMarkovChain.steady_pmf
# --------------------------------------------
@pytest.mark.parametrize('chain, pmf, comment', [
    (ctmc1(), [1.0], 'trivial CTMC with 1 state'),
    (ctmc2(), [0.5, 0.5], 'periodic CTMC with 2 states'),
    (ctmc3(), [.885, .071, .044], 'CTMC with 3 states'),
])
def test_ctmc__steady_pmf(chain, pmf, comment):
    """
    Validate steady state PMF evaluation of a continuous time Markov chain.
    """
    assert isinstance(chain, ContinuousTimeMarkovChain)
    assert_allclose(chain.steady_pmf, pmf, rtol=0.01, err_msg=comment)


# Testing ContinuousTimeMarkovChain.trace()
# -----------------------------------------
@pytest.mark.parametrize('chain, size, init, ends, safe', [
    (ctmc2(), 5, 0, (), True),
    (ctmc3(), 8, None, [3], False),
    (ctmc3(), 42, [0.3, 0.3, 0.4], (), True)
])
def test_ctmc__trace_calls_dtmc_trace(chain, size, init, ends, safe):
    """
    Validate that calling ctmc.trace() make a call to embedded DTMC trace().
    """
    trace_mock = Mock(return_value=())
    chain.embedded_dtmc.trace = trace_mock  # patch (very ugly)

    # Make a call to trace(). Since it is a generator, to actually call
    # embedded chain trace() method, convert it to list:
    _ = list(chain.trace(size, init, ends, safe))

    # Make sure embedded DTMC trace() method was properly called:
    trace_mock.assert_called_once_with(size, init=init, ends=ends, safe=safe)


def test_ctmc__trace_intervals_converge_to_matrix_means():
    """
    Validate that intervals in the trace path converge to the mean values
    -1/M[i,i] from the chain generator M. To test this, we generate a
    reasonably large trace.
    """
    chain = ctmc3()
    intervals = np.zeros(chain.order)
    hits = np.zeros(chain.order)
    CHAIN_SIZE = 10000
    # noinspection PyTypeChecker
    for (state, _, interval) in chain.trace(CHAIN_SIZE):
        intervals[state] += interval
        hits[state] += 1
    intervals /= hits
    est_rates = 1 / intervals
    assert_allclose(est_rates, chain.rates, rtol=0.1)
