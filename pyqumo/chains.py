from functools import cached_property
from typing import Union, Sequence, Iterable, Optional, Tuple, Iterator

import numpy as np

from pyqumo.errors import MatrixShapeError
from pyqumo.matrix import is_infinitesimal, order_of, is_pmf, is_stochastic, \
    fix_stochastic, identity, fix_infinitesimal, str_array, is_square


class DiscreteTimeMarkovChain:
    """
    Class representing discrete time Markov chain.
    """
    def __init__(self, matrix: Union[Sequence[Sequence[float]], np.ndarray],
                 safe: bool = False, tol: float = 1e-3):
        """
        Discrete time Markov chain constructor.

        Parameters
        ----------
        matrix : 2-D array_like
            Transition matrix. If `safe = False` it is checked to be stochastic
            and, if check fails, attempt to fix it with `fix_stochastic()`
            call is made. When fixing, use tolerance `tol`.
        safe : bool, optional
            Flag indicating whether there is no need to validate matrix.
            Default: `False`.
        tol : float, optional
            Tolerance used when fixing broken transition matrix.
            If `safe = True` is set, this field is ignored. Default: 1e-3.
        """
        if not isinstance(matrix, np.ndarray):
            matrix = np.asarray(matrix)
        else:
            matrix = matrix.copy()  # copy to avoid side-effects
        if not is_square(matrix):
            raise MatrixShapeError('(N, N)', matrix.shape, 'transition matrix')
        if not safe and not is_stochastic(matrix):
            matrix = fix_stochastic(matrix, tol=tol)[0]
        self._matrix = matrix
        self._order = order_of(matrix)

    @property
    def matrix(self) -> np.ndarray:
        """
        Get transitions matrix.
        """
        return self._matrix

    @property
    def order(self) -> int:
        """
        Get chain order, i.e. number of states.
        """
        return self._order

    @cached_property
    def steady_pmf(self) -> np.ndarray:
        """
        Returns the steady-state probabilities distribution (mass function).

        The algorithm will attempt to use `numpy.linalg.solve()` method to
        solve the system. If it fails due to singular matrix,
        `numpy.linalg.lstsq()` method will be used.

        Notes
        -----
        the method is cached: while the first call may take quite a
        lot of time, all the succeeding calls will require only cache
        lookups and have O(1) complexity.
        """
        n = self.order
        left_side = np.vstack((self.matrix.T - np.eye(n), np.ones((1, n))))
        right_side = np.zeros(n + 1)
        right_side[-1] = 1.
        try:
            left_side_ = left_side[1:, :]
            right_side_ = right_side[1:]
            return np.linalg.solve(left_side_, right_side_)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(left_side, right_side)[0]

    def trace(self, size: Optional[int] = None,
              init: Union[int, Sequence[float], None] = None,
              ends: Sequence[int] = (),
              safe: bool = False) -> Iterator[Tuple[int, int]]:
        """
        Generates a generator for a random path produced by the chain.

        A path is represented as tuples `(prev_state, next_state)`.
        The initial state is provided in `init` as either state number (int),
        or probability mass function (sequence of float). If omitted,
        steady state PMF is used.

        Parameters
        ----------
        size : int, optional
            If provided, limits the maximum number of steps.
        ends : sequence of ints, optional
            Path terminating states. If arrived at any of them, path
            generation will stop.
        init : int, sequence of floats or None, optional
            If integer, specifies the initial state. If a sequence of floats,
            specifies probability mass function. If omitted (default),
            steady state PMF is used to select the initial state.
        safe : bool, optional
            Flag indicating whether to check `init` PMF. If `init` is an
            integer or None, this value is ignored. Default: `False`.

        Returns
        -------
        generator : iterator of (int, int) tuples
            Path generator, builds a sequence of int pairs
            `(curr_state, next_state)`
        """
        # Process `init`: if it is None or integer, set it to identity or
        # steady probability mass function, accordingly:
        if init is None:
            safe = True
            init = self.steady_pmf
        elif not isinstance(init, Iterable):
            safe = True
            init = identity(self.order, init)
        elif not isinstance(init, np.ndarray):
            init = np.asarray(init)

        # Now `init` is np.ndarray and `safe` is `True` if it was filled
        # from the state number of steady PMF. Check and fix `init` if safe
        # is `False`:
        if not safe and not is_pmf(init):
            init = fix_stochastic(init)

        # Select initial state:
        state: int = np.random.choice(np.arange(self.order), p=init)

        # Generate trace:
        step = 0
        while size is None or step < size and (not ends or state not in ends):
            probs = self.matrix[state]
            next_state = np.random.choice(np.arange(self.order), p=probs)
            yield state, next_state
            state = next_state
            step += 1

    def random_path(
            self, size: int,
            ends: Sequence[int] = (),
            init: Union[int, Sequence[float], None] = None
    ) -> Sequence[int]:
        """
        Build a random path of the given maximum length, end and init states.

        In contrast to `trace()`, this method returns a sequence of states
        instead of a generator. Due to this, `size` parameter here is
        mandatory.

        Parameters
        ----------
        size : int
            Maximum length of the path.
        ends : sequence of int, optional
            Path terminating states. If arrived at any of them, path
            generation will stop.
        init : int, sequence of floats or None, optional
            If integer, specifies the initial state. If a sequence of floats,
            specifies probability mass function. If omitted (default),
            steady state PMF is used to select the initial state.

        Returns
        -------
        path : sequence of int
            Sequence of visited states along the path.
        """
        generator = self.trace(size=size, ends=ends, init=init)
        try:
            first = next(generator)
        except StopIteration:
            return ()
        return first + tuple(step[1] for step in generator)

    def __repr__(self):
        return f"(DTMC: t={str_array(self.matrix)})"


class ContinuousTimeMarkovChain:
    """
    Class representing continuous time Markov chain.
    """
    def __init__(self, matrix: Sequence[Sequence[float]],
                 safe: bool = False,
                 tol: float = 1e-3):
        """
        Constructor of a continuous time Markov chain.

        Parameters
        ----------
        matrix : 2-D array_like
            An infinitesimal matrix (chain generator).
        safe : bool, optional
            Flag indicating the matrix is safe to use. If `False` (default),
            no validation or attempts to fix will be performed.
        tol : float, optional
            Tolerance used when fixing broken infinitesimal generator matrix.
            If `safe = True` is set, this field is ignored. Default: 1e-3.
        """
        need_copy = False
        if not isinstance(matrix, np.ndarray):
            matrix = np.asarray(matrix)
        else:
            need_copy = True

        if not safe and not is_infinitesimal(matrix):
            matrix = fix_infinitesimal(matrix, tol=tol)[0]
            need_copy = False

        self._matrix = matrix if not need_copy else matrix.copy()
        self._order = order_of(matrix)

    @property
    def matrix(self) -> np.ndarray:
        """
        Get infinitesimal generator of the chain.
        """
        return self._matrix

    @property
    def order(self) -> int:
        """
        Get number of states in the chain.
        """
        return self._order

    @cached_property
    def rates(self) -> np.ndarray:
        """
        Get rates of departures from the states.
        """
        return -self.matrix.diagonal()

    @cached_property
    def steady_pmf(self) -> np.ndarray:
        """
        Returns the steady-state probabilities distribution (mass function).

        Depending on the generator matrix  the algorithm will use either
        `numpy.linalg.solve()` (if the generator matrix has rank N-1), or
        `numpy.linalg.lstsq()` (if the generator matrix has rank N-2 or less).

        Notes
        -----
        the method is cached: while the first call may take quite a
        lot of time, all the succeeding calls will require only cache
        lookups and have O(1) complexity.
        """
        left_side = np.vstack((self.matrix.T, np.ones((1, self.order))))
        right_side = np.zeros(self.order + 1)
        right_side[-1] = 1.
        try:
            left_side_ = left_side[1:, :]
            right_side_ = right_side[1:]
            return np.linalg.solve(left_side_, right_side_)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(left_side, right_side)[0]

    def trace(self, size: Optional[int] = None,
              init: Union[int, Sequence[float], None] = None,
              ends: Sequence[int] = (),
              safe: bool = False) -> Iterator[Tuple[int, int, float]]:
        """
        Build a generator for a random path in the chain.

        A path is a sequence of tuples `(prev_state, next_state, interval)`.

        The initial state is provided in `init` as either state number (int),
        or probability mass function (sequence of float). If omitted,
        steady state PMF is used.

        Path is retrieved from embedded DTMC using its `trace()` method
        with the given `size`, `init` and `ends` parameters. Intervals are
        random samples of the time spent in `prev_state`.

        Parameters
        ----------
        size : int, optional
            If provided, limits the maximum number of steps.
        ends : sequence of ints, optional
            Path terminating states. If arrived at any of them, path
            generation will stop.
        init : int, sequence of floats or None, optional
            If integer, specifies the initial state. If a sequence of floats,
            specifies probability mass function. If omitted (default),
            steady state PMF is used to select the initial state.
        safe : bool, optional
            Flag indicating whether to check `init` is a PMF.
            If `init` is an integer or `None` this value is ignored.
            Default: `False`.

        Returns
        -------
        generator : iterable of tuples
            Path generator, builds a sequence of int pairs
            `(prev_state, next_state, interval)`
        """
        dtmc_trace = self.embedded_dtmc.trace(
            size, init=init, ends=ends, safe=safe
        )
        for curr_state, next_state in dtmc_trace:
            rate = self.rates[curr_state]
            interval = np.random.exponential(1/rate) if rate > 1e-12 else np.inf
            yield curr_state, next_state, interval

    @cached_property
    def embedded_dtmc(self) -> DiscreteTimeMarkovChain:
        """
        Get the discrete time Markov chain embedded in this CTMC.
        """
        # Build transition matrix:
        n = self.order
        d1 = self.matrix + np.diag(self.rates)
        trans_matrix_rows = []
        for i, rate in enumerate(self.rates):
            # If rate is non-zero, divide infinitesimal generator row
            # with zeroed diagonal on the rate:
            if rate > 1e-12:
                row = d1[i] / rate
            # Otherwise, if state is absorbing, put 1 to transition matrix
            # diagonal to make it absorbing in DTMC as well:
            else:
                row = identity(n, i, axis=1)
            # Record the row:
            trans_matrix_rows.append(row)
        trans_matrix = np.vstack(trans_matrix_rows)

        # Build DTMC:
        return DiscreteTimeMarkovChain(trans_matrix)

    def __repr__(self):
        return f"(CTMC: g={str_array(self.matrix)})"
