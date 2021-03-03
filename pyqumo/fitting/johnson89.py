"""
This module contains an implementation of the PH fitting algorithm using a
mixture of two Erlang distribution with common order N by three moments match.
The method is defined in paper [1].

[1] Mary A. Johnson & Michael R. Taaffe (1989) Matching moments to phase
    distributions: Mixtures of erlang distributions of common order,
    Communications in Statistics. Stochastic Models, 5:4, 711-743,
    DOI: 10.1080/15326348908807131
"""
from typing import Sequence, Tuple
import numpy as np

from pyqumo.random import HyperErlang
from pyqumo.stats import get_cv, get_skewness, get_noncentral_m3
from pyqumo.errors import BoundsError


def fit_mern2(
    moments: Sequence[float],
    strict: bool = True,
    max_shape_inc: int = 0
) -> Tuple[HyperErlang, np.ndarray]:
    """
    Fit moments with a mixture of two Erlang distributions with common order.

    The algorithm is defined in [1].

    In strict mode (`strict = True`) requires at least three moments to fit.
    Note, that if more moments provided, function will compute errors in
    their estimation, while not taking them into account actually.
    If the first three moments do not fit into feasible area, function will
    raise `BoundsError` exception.

    In non-strict mode (`strict=False`) user can provide two or even one
    moment, and M3 may go beyond the feasible area. In this case,
    M3 will be selected using the rule:

    - if `cv > 1`, then skewness will be equal to `(cv - 1/cv) * 1.2`;
    - if `cv == 1`, then skewness will be equal 2 (exponential distrib.);
    - if `0 < cv < 1`, then skewness will be equal to `(cv - 1/cv) * .8`.

    User can provide `max_shape_inc` parameter. If so, the algorithm will
    try to improve the ratio between Erlang distribution parameters and
    minimum probability as described in section 7.3 of [1] by increasing
    the Erlang distributions shape (up to `max_shape_inc`).

    Parameters
    ----------
    moments : sequence of float
        Only the first three moments are taken into account
    strict : bool, optional (default: `True`)
        If `True`, require at least three moments explicitly defined,
        and do not perform any attempt to adjust moments if they are out of
        bounds. Otherwise, try to fit even for bad or incomplete data.
    max_shape_inc : int, optional (default: 0)
        if non-zero, maximum increase in shape when attempting to build
        a more stable distribution (refer to 7.3 "Numerical Stability" section
        of [1] for details)

    Raises
    ------
    BoundsError
        raise this if moments provided are out of bounds (in strict mode),
        or can not be recovered (non-strict mode).
    ValueError
        raise this in strict mode if number of moments provided is less then
        three, or if no moments provided at all (also in non-strict mode).

    Returns
    -------
    dist : HyperErlang
        an instance of HyperErlang distribution fitted
    errors : tuple of float
        errors computed for each moment provided
    """
    if (num_moments := len(moments)) == 3:
        m1, m2, m3 = moments[:3]
        cv = get_cv(m1, m2)
    else:
        if (strict and num_moments < 3) or num_moments == 0:
            raise ValueError(f"Expected 3 moments, but {num_moments} found")
        m1 = moments[0]
        m2 = moments[1] if num_moments > 1 else 2*pow(m1, 2)
        cv = get_cv(m1, m2)
        if cv < 1 - 1e-5:
            m3 = get_noncentral_m3(m1, cv, (cv - 1/cv) * 0.8)
        elif abs(cv - 1) <= 1e-4:
            m3 = moments[2] if num_moments > 2 else 6*pow(m1, 3)
        else:
            m3 = get_noncentral_m3(m1, cv, (cv - 1/cv) * 1.2)

    gamma = get_skewness(m1, m2, m3)

    # Check boundaries and raise BoundsError if fail:
    if (min_skew := cv - 1/cv) >= gamma:
        if strict:
            raise BoundsError(
                f"Skewness = {gamma:g} is too small for CV = {cv:g}\n"
                f"\tmin. skewness = {min_skew:g}\n"
                f"\tm1 = {m1:g}, m2 = {m2:g}, m3 = {m3:g}")
        else:
            if cv < 1 - 1e-5:
                m3 = get_noncentral_m3(m1, cv, (cv - 1/cv) * 0.8)
            elif abs(cv - 1) <= 1e-4:
                m3 = 6 * pow(m1, 3)
            else:
                m3 = get_noncentral_m3(m1, cv, (cv - 1/cv) * 1.2)
            print('previous gamma: ', gamma)
            gamma = get_skewness(m1, m2, m3)
            print('new gamma: ', gamma)

    # Compute minimal shape for Erlang distributions:
    shape = int(max(
        np.ceil(1 / cv**2),
        np.ceil((-gamma + 1/cv**3 + 1/cv + 2*cv) / (gamma - (cv - 1/cv)))
    )) + (2 if cv <= 1 else 0)

    # If allowed to use higher order of Erlang to make results more stable,
    # try to optimize shape. Otherwise, just get the parameters for
    # the shape found above:
    shape, l1, l2, p = _optimize_stability(m1, m2, m3, shape, max_shape_inc)

    # Build hyper-Erlang distribution:
    dist = HyperErlang([l1, l2], [shape, shape], [p, 1 - p])

    # Estimate errors:
    errors = np.asarray([
        abs(m - dist.moment(i+1)) / abs(m) for i, m in enumerate(moments)
    ])

    return dist, errors


def get_mern2_props(
        m1: float,
        m2: float,
        m3: float,
        n: int) -> Tuple[float, float, float]:
    """
    Helper function to estimate Erlang distributions rates and
    probabilities from the given moments and Erlang shape (n).

    See theorem 3 in [1] for details about A, B, C, p1, x, y and lambdas
    computation.

    Parameters
    ----------
    m1 : float
        mean value
    m2 : float
        second non-central moment
    m3 : float
        third non-central moment
    n : int
        shape of the Erlang distributions

    Returns
    -------
    l1 : float
        parameter of the first Erlang distribution
    l2 : float
        parameter of the second Erlang distribution
    n : int
        shape of the Erlang distributions

    Raises
    ------
    BoundsError
        raise this if skewness is below CV - 1/CV (CV - coef. of variation)
    """
    # Check boundaries:
    cv = get_cv(m1, m2)
    gamma = get_skewness(m1, m2, m3)
    if (min_skew := cv - 1/cv) >= gamma:
        raise BoundsError(
            f"Skewness = {gamma:g} is too small for CV = {cv:g}\n"
            f"\tmin. skewness = {min_skew:g}\n"
            f"\tm1 = {m1:g}, m2 = {m2:g}, m3 = {m3:g}")

    # Compute auxiliary variables:
    x = m1 * m3 - (n + 2) / (n + 1) * pow(m2, 2)
    y = m2 - (n + 1) / n * pow(m1, 2)
    c = m1 * x
    b = -(
        n * x +
        n * (n + 2) / (n + 1) * pow(y, 2) +
        (n + 2) * pow(m1, 2) * y
    )
    a = n * (n + 2) * m1 * y
    d = pow(b**2 - 4 * a * c, 0.5)

    # Compute Erlang mixture parameters:
    em1, em2 = (-b - d) / (2*a), (-b + d) / (2*a)
    p1 = (m1 / n - em2) / (em1 - em2)
    l1, l2 = 1 / em1, 1 / em2
    return l1, l2, p1


def _optimize_stability(
        m1: float,
        m2: float,
        m3: float,
        shape_base: int,
        max_shape_inc: int) -> Tuple[int, float, float, float]:
    """
    Optimize stability of the resulting hyper-Erlang distribution.

    Try to slightly increase shape of Erlang distributions to make
    ratio between Erlang parameters (`r = l2 / l1` if `l2 > l1`) less,
    as well as to increase the minimum probability.

    Parameters
    ----------
    m1: float
    m2: float
    m3: float
    shape_base: int
    max_shape_inc: int

    Returns
    -------
    shape: int
    l1: float
    l2: float
    p: float
    """
    shape = shape_base
    l1, l2, p = get_mern2_props(m1, m2, m3, shape)

    def get_ratio(l1_, l2_):
        """
        Helper to get ratio between Erlang rates. Always return value >= 1.
        """
        if l2_ >= l1_ > 0:
            return l2_ / l1_
        return l1_ / l2_ if l1_ > l2_ > 0 else np.inf

    def get_min_prob(p_):
        return p_ if p_ < 0.5 else 1 - p_

    r_max_prev = get_ratio(l1, l2)
    p_min_prev = get_min_prob(p)
    inc = 1
    while inc < max_shape_inc:
        shape = shape_base + inc
        l1_new, l2_new, p_new = get_mern2_props(m1, m2, m3, shape)
        r_max_curr = get_ratio(l1_new, l2_new)
        p_min_curr = get_min_prob(p_new)
        # If shape increase doesn't provide sufficient improvement,
        # stop iteration and abondon changes:
        if r_max_prev / r_max_curr < 1.1 and p_min_curr / p_min_prev < 1.1:
            shape = shape - 1
            break
        # Otherwise remember current values of L1, L2, P and go to the
        # next iteration:
        l1, l2, p = l1_new, l2_new, p_new
        inc += 1
    return shape, l1, l2, p
