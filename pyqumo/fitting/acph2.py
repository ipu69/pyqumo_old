"""
This module contain implementation of ACPH(2) moments matching fitting
algorithm defined in [1].

It defines three routines for fitting and bounds checking:

- fit_acph2()
- get_acph2_m2_min()
- get_acph2_m3_bounds()

[1] Telek, Miklós & Heindl, Armin. (2003). Matching Moments For Acyclic 
Discrete And Continuous Phase-Type Distributions Of Second Order. 
International Journal of Simulation Systems, Science & Technology. 3.
"""
from typing import Sequence, Tuple
import numpy as np

from pyqumo.errors import BoundsError
from pyqumo.random import PhaseType


def fit_acph2(
    moments: Sequence[float],
    strict: bool = True
) -> Tuple[PhaseType, np.ndarray]:
    """
    Fit ACPH(2) distribution matching three first moments.

    Algorithm and moments values bounds are provided in [1]. See Table 1 for
    boundaries on M2 and M3, Figure 1 for graphical display of the boundaries
    and Table 3 for formulas to compute PH distribution parameters from
    valid M1, M2 and M3.

    If algorithm fails to fit due to values M1, M2 and M3 laying out of bounds,
    raise `BoundsError` exception when `strict = True`. Otherwise, try to
    find the closest ACPH. In the latter case, select moments with these rules:

    1. If `cv^2 := (m1^2 / m2 - 1) <= 0.5`, then `m2` and `m3` are set equal to
        `m2 = 1.5 * m1^2` and `m3 = 3 * m1^3` respectively (as for Erlang-2).
    
    2. If `0.5 < cv^2 < 1.0`, then `m2` is OK. However, if `m3` is out of 
        bounds (region BII or BIII, see Figure 2 in [1]), then `m3` is selected
        to be `m3 = 0.5 * (m3_lower + m3_upper)`, where `m3_lower` and 
        `m3_upper` are boundaries of BII and BIII, see Table 1 [1] and
        `get_acph2_m3_bounds()`.
    
    3. If `cv == 1`, then m3` is set as for exponential distribution: 
        `m3 = 6 * m1^3`
    
    4. If `cv > 1`, then `m3` is set as `10/9 * m3_min`, where `m3_min` value
        is defined as the boundary of BI (see Figure 2 in [1]).

    The same rules for selecting moments `m2` and `m3` apply if less then
    three moments were provided and `strict = False`.

    If more then three moments are provided to the algorithm, 4-th and higher
    order moments are not used in estimation. However, the alogirhtm computes
    relative errors for these moments from the fitted ACPH(2).
    
    If any moment is less or equal to zero, or if pow(CV, 2) <= 0,
    the `ValueError` exception is raised.

    Parameters
    ----------
    moments : sequence of floats
        First moments. If strict = True, then this sequence MUST contain at
        least three moments. If strict = False, then missing moments will be
        selected by the algorithm.
    strict : bool, optional
        Flag indicating whether raise an error if ACPH(2) can not be found
        due to moment values laying out of bounds.
    
    Raises
    ------
    ValueError
        raise this when moments are less or equal to 0, or if pow(CV, 2) <= 0
    BoundsError
        raise if strict = True and moments values
    
    Returns
    -------
    ph : PhaseType
        ACPH(2) distribution
    errors : np.ndarray
        tuple containing relative errors for moments of the distribution found.
        The number of errors is equal to the number of moments passed: if 
        more then three moments were given, errors will be estimated for all
        of them. If strict = False and one or two moments were passed,
        then errors will be computed only for these one or two moments.
    """
    # First of all, check either three moments are provided, or strict = False:
    # - if strict = True and len(moments) < 3, raise ValueError
    # - if strict = False, however, try to find some m2 and m3 values
    #   to use in estimation.
    # 
    # If CV2 falls into region between (BII, BIII), use M3 value in the medium.
    # If CV2 > 1, use M3 from a line that is as 10/9 m3 lower bound.
    #
    # If in non-strict mode only M1 is provided, treat as exponential 
    # distribution and set M2 and M3 accordingly.
    if (n := len(moments)) < 3:
        if strict:
            raise ValueError(f"Expected three moments, but {n} found")
        if n == 0:
            raise ValueError(f"At least one moment needed, but 0 found")
        
        # If not strict and at least one moment provided, try to find m2, m3.
        m1 = moments[0]
        m2 = 2 * m1**2 if n < 2 else moments[1]
        
        # M3 selection depends on pow(CV, 2)
        cv2 = m2 / m1**2 - 1
        if cv2 <= 0.5:  # tread as Erlang: M3 = k(k+1)(k+2) / pow(k/m1, 3), k=2
            m3 = 3 * m1**3
        elif 0.5 < cv2 <= 1.0:
            m3 = sum(get_acph2_m3_bounds(m1, m2)) / 2
        else:
            m3 = 5/3 * m1**3 * (1 + cv2)**2  # to step from boundary
    else:
        m1 = moments[0]
        m2 = moments[1]
        m3 = moments[2]
    
    # Validate mandatory moments relations:
    # - each moment must be positive real value
    # - pow(CV, 2) must be positive
    for i, m in enumerate(moments):
        if m <= 0:
            raise ValueError(f"Expected m{i+1} > 0, but m{i+1} = {m}")
    cv2 = m2 / m1**2 - 1
    if cv2 <= 0:
        raise ValueError(f"Expected pow(CV, 2) > 0, but pow(CV, 2) = {cv2}")

    # If strict = True, validate moments and raise error is out of bounds:
    if strict:
        if (m2_min := get_acph2_m2_min(m1)) > m2:
            raise BoundsError(
                f"m2 = {m2} is out of bounds for m1 = {m1}\n"
                f"\tpow(CV, 2) = {cv2}\n"
                f"\tmin. pow(CV, 2) = 0.5\n"
                f"\tmin. M2 = {m2_min}"
            )
        m3_min, m3_max = get_acph2_m3_bounds(m1, m2)
        if not (m3_min <= m3 <= m3_max):
            raise BoundsError(
                f"m3 = {m3} is out of bounds for m1 = {m1}, m2 = {m2}\n"
                f"\tpow(CV, 2) = {cv2}\n"
                f"\tmin. M3 = {m3_min}\n"
                f"\tmax. M3 = {m3_max}"
            )
    
    # If strict = False, tune moments to put them into the valid bounds:
    if not strict:
        # If pow(CV, 2) < 0.5, then set M2 and M3 as for Erlang-2 distribution:
        if cv2 < 0.5:
            m2 = 1.5 * m1**2
            m3 = 3 * m1**3
        elif cv2 < 1.0:
            m3_min, m3_max = get_acph2_m3_bounds(m1, m2)
            if not (m3_min <= m3 <= m3_max):
                m3 = 0.5 * (m3_min + m3_max)
        elif cv2 == 1.0:
            m3 = 6 * m1**3
        elif cv2 > 1.0:
            m3_min = get_acph2_m3_bounds(m1, m2)[0]
            m3 = m3_min * 10/9

    # Define auxiliary variables
    d = 2 * m1**2 - m2
    c = 3 * m2**2 - 2 * m1 * m3
    b = 3 * m1 * m2 - m3
    a = (b**2 - 6 * c * d) ** 0.5  # in paper no **0.5, but this is useful
    
    # Define subgenerator and probabilities vector elements
    if c > 0:
        p = (-b + 6 * m1 * d + a) / (b + a)
        l1 = (b - a) / c
        l2 = (b + a) / c
    elif c < 0:
        p = (b - 6 * m1 * d + a) / (-b + a)
        l1 = (b + a) / c
        l2 = (b - a) / c
    else:
        p = 0
        l1 = 1 / m1
        l2 = 1 / m1
    
    # Build the distribution and compute estimation errors:
    ph = PhaseType(
        sub=np.asarray([[-l1, l1], [0.0, -l2]]), 
        p=np.asarray([p, 1 - p]))
    errors = [abs(m - ph.moment(i+1)) / m for i, m in enumerate(moments)]
    return ph, np.asarray(errors)


def get_acph2_m2_min(m1: float) -> float:
    """
    Get minimum value of m2 (second moment) for ACPH(2) fitting.

    According to [1], M2 has only lower bound since pow(CV, 2) should be 
    greater or equal to 0.5.

    If m1 < 0, then `ValueError` is raised.

    Parameters
    ----------
    m1 : float

    Returns
    -------
    m2_min : float
        Minimum eligble value of the second moment.
    """
    if m1 < 0:
        raise ValueError(f"Expected m1 > 0, but m1 = {m1}")
    return 1.5 * m1**2


def get_acph2_m3_bounds(m1: float, m2: float) -> Tuple[float, float]:
    """
    Get minimum and maximum possible values of the 3-rd moment for ACPH(2).

    Bounds are specified in Table 1 and Figure 2 in [1]. When CV > 1,
    only lowest bound exist, while for 0.5 < pow(CV, 2) < 1 both lower and 
    upper bounds are defined, and they are very tight. 
    When CV = 1, M3 is fixed for exponential distribution (singular point), 
    so both bounds are equal.

    If arguments are such that CV**2 < 0.5 (i.e. m2 < 1.5 * m1**2), then
    `ValueError` is raised.

    Parameters
    ----------
    m1 : float
    m2 : float

    Returns
    -------
    lower : float
    upper : float
    """
    if m1 <= 0:
        raise ValueError(f"Expected m1 > 0, but m1 = {m1}")
    if m2 <= 0:
        raise ValueError(f"Expected m2 > 0, but m2 = {m2}")

    # Find square of coefficient of variation (CV**2):
    cv2 = m2 / m1**2 - 1

    # If CV > 1, then only lower bound exists:
    if cv2 > 1:
        return 3/2 * m1**3 * (1 + cv2)**2, np.inf
    
    # If CV**2 >= 0.5, but <= 1, both bounds exist:
    if 0.5 <= cv2 <= 1:
        return (
            3 * m1**3 * (3 * cv2 - 1 + 2**0.5 * (1 - cv2)**1.5),
            6 * m1**3 * cv2
        )
    
    # If CV**2 < 0.5, M3 is undefined:
    raise ValueError(
        f"Expected CV >= sqrt(0.5), but CV = {cv2**0.5} "
        "(CV = coef. of variation)")
