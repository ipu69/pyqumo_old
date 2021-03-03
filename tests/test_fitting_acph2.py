import pytest
from numpy.testing import assert_allclose
import numpy as np

from pyqumo.errors import BoundsError
from pyqumo.fitting import fit_acph2
from pyqumo.fitting.acph2 import get_acph2_m2_min, get_acph2_m3_bounds


@pytest.mark.parametrize('m1, cv2, m3', [
    (4/3, 1.1, 18),
    (4/3, 0.8, 11),
    (1, 1.21, 7.4),
    (1, 1, 6)
])
def test_fit_acph2__with_good_data(m1, cv2, m3):
    """
    Test fit_acph2 with good data obtained from Telek and Heindl paper.

    Data was retrieved from Figure 2 of paper [1]. First point is from
    region with CV > 1, second point - from tight stripe when CV < 1, and
    the last point relates to singular point representing exponential
    distribution.

    [1] Telek, Miklós & Heindl, Armin. (2003). Matching Moments For Acyclic 
    Discrete And Continuous Phase-Type Distributions Of Second Order. 
    International Journal of Simulation Systems, Science & Technology. 3.
    """
    cv = cv2**0.5
    m2 = (cv2 + 1) * m1**2

    ph, errors = fit_acph2([m1, m2, m3])

    assert_allclose(errors, (0, 0, 0), rtol=1e-8, atol=1e-8)
    assert_allclose(ph.mean, m1)
    assert_allclose(ph.var, m2 - m1**2)
    assert_allclose(ph.cv, cv)
    assert_allclose(ph.moment(3), m3)


def test_acph2_m2_min():
    """Test M2 lower bound (m2 >= 1.5 * m1**2).
    """
    assert_allclose(get_acph2_m2_min(1.0), 1.5)
    assert_allclose(get_acph2_m2_min(2.0), 6.0)
    with pytest.raises(ValueError) as err:
        get_acph2_m2_min(-1)
    assert str(err.value) == "Expected m1 > 0, but m1 = -1"


def test_acph2_m3_bounds():
    """
    Test M3 lower and upper bounds as defined in [1].
    """
    # When pow(CV, 2) >= 0.5, both bounds exist:
    assert_allclose(
        get_acph2_m3_bounds(1, 1.64), 
        (3.67641, 3.840), rtol=1e-5)
    assert_allclose(
        get_acph2_m3_bounds(2.5, 11.890625),
        (82.05726, 84.609375), rtol=1e-5)
    
    # When CV=1, both bounds are equal each other:
    assert_allclose(get_acph2_m3_bounds(1, 2), (6.0, 6.0))

    # When CV > 1, only lower bound exists:
    assert_allclose(get_acph2_m3_bounds(1.0, 5.0), (37.5, np.inf))
    assert_allclose(get_acph2_m3_bounds(4.0, 35.36), (468.8736, np.inf))

    # When pow(CV, 2) < 0.5, ValueError is expected:
    with pytest.raises(ValueError) as err:
        get_acph2_m3_bounds(1.0, 1.49)
    assert str(err.value) == \
        "Expected CV >= sqrt(0.5), but CV = 0.7 (CV = coef. of variation)"

    # When either M1 or M2 is non-positive, ValueErrors should be raised:
    with pytest.raises(ValueError) as err:
        get_acph2_m3_bounds(-2, 4)
    assert str(err.value) == "Expected m1 > 0, but m1 = -2"

    with pytest.raises(ValueError) as err:
        get_acph2_m3_bounds(1, -4)
    assert str(err.value) == "Expected m2 > 0, but m2 = -4"


@pytest.mark.parametrize('moments, err_str', [
    ([1, 1.64, 3.0], "m3 = 3.0 is out of bounds for m1 = 1, m2 = 1.64"),
    ([3.4, 18.2, 225], "m3 = 225 is out of bounds for m1 = 3.4, m2 = 18.2"),
    ([4, 35.36, 460], "m3 = 460 is out of bounds for m1 = 4, m2 = 35.36"),
    ([2, 5, 10], "m2 = 5 is out of bounds for m1 = 2")
])
def test_fit_acph2__srtict__raise_error_for_bad_values(moments, err_str):
    with pytest.raises(BoundsError) as err:
        fit_acph2(moments, strict=True)
    assert str(err.value).startswith(err_str)


@pytest.mark.parametrize('moments, err_str', [
    ([-1, 2, 3], "Expected m1 > 0, but m1 = -1"),
    ([1, -2, 3], "Expected m2 > 0, but m2 = -2"),
    ([1, 2, -3], "Expected m3 > 0, but m3 = -3"),
    ([2, 3, 10], "Expected pow(CV, 2) > 0, but pow(CV, 2) = -0.25")
])
def test_fit_acph2__non_strict__raise_error_for_bad_moments(moments, err_str):
    """
    No matter of strict = True, if any moment is non-positive or CV <= 0,
    raise ValueError indicating failed moment (or CV).
    """
    with pytest.raises(ValueError) as err:
        fit_acph2(moments, strict=False)  # even non-strict mode!
    assert str(err.value) == err_str


def test_fit_acph2__strict__raise_error_if_less_then_three_moments_given():
    with pytest.raises(ValueError) as err:
        fit_acph2([1, 2], strict=True)
    assert str(err.value) == "Expected three moments, but 2 found"


def test_fit_acph2__non_strict__finds_ph_for_m1():
    """If only M1 is provided, get exponential distribution in ACPH(2) form.
    """
    ph, _ = fit_acph2([1], strict=False)
    assert_allclose(ph.mean, 1)
    assert_allclose(ph.cv, 1)
    assert_allclose(ph.moment(3), 6)


def test_fit_acph2__non_strict__finds_ph_for_m1_and_m2():
    """If only M1 and M2 are provided, take some M3 and find ACPH(2)
    when strict = False.
    """
    # If 0.5 <= pow(CV, 2) <= 1.0, take M3 value between boundaries:
    ph1, _ = fit_acph2([1, 1.64], strict=False)
    assert_allclose(ph1.moment(1), 1)
    assert_allclose(ph1.moment(2), 1.64)
    assert_allclose(ph1.moment(3), 3.7582051942088834)

    # If 1 < CV, take M3 from boundary:
    ph2, _ = fit_acph2([1, 2.21], strict=False)
    assert_allclose(ph2.moment(1), 1)
    assert_allclose(ph2.moment(2), 2.21)
    assert_allclose(ph2.moment(3), 7.32615 * 10/9)

    # If 1 = CV, M3 equals 6 * pow(m1, 3):
    ph3, _ = fit_acph2([1, 2], strict=False)
    assert_allclose(ph3.moment(1), 1)
    assert_allclose(ph3.moment(2), 2)
    assert_allclose(ph3.moment(3), 6)


@pytest.mark.parametrize('moments, errors, comment', [
    ([1, 1.25], (0, 0.2), "too small CV2, no M3 - align M2"),
    ([1, 1.1, 2, 10], (0, 0.364, 0.5, 0.25), "too small CV2 with M3 and M4"),
    ([1.33, 3.01, 8.0], (0, 0, 0.208), "too small M3 in BII (CV2 < 1)"),
    ([1.33, 3.01, 15.0], (0, 0, 0.356), "too large M3 in BIII (CV2 < 1)"),
    ([1.33, 3.89, 15.0], (0, 0, 0.264), "too small M3 in BI (CV2 > 1)"),
    ([2, 8, 42], (0, 0, 0.143), "M3 set to Exponential distribution (CV = 1)"),
    ([2.00001, 8.00001, 34], (0, 0, 0.412), "M3 very close to exp. (left)"),
    ([1.99999, 8.00001, 34], (0, 0, 0.569), "M3 very close to exp. (right)"),
])
def test_fit_acph2__non_strict__finds_approx_solution(moments, errors, comment):
    """
    Validate that fit_acph2() with strict = False tries to fit moments even
    if they are out of the bounds, and put relative errors into the result.
    """
    ph, actual_errors = fit_acph2(moments, strict=False)

    assert_allclose(
        actual_errors, errors, rtol=1e-3, atol=1e-3, err_msg=comment)
    for i, m in enumerate(moments):
        rtol = max((errors[i]*1.01), 1e-5)  # to avoid zero rtol for error=0
        assert_allclose(ph.moment(i+1), m, rtol=rtol, err_msg=comment)


def test_fit_acph2__strict__compute_errors_for_all_moments_passed():
    """
    Validate that when more then three moments passed to fit_acph2() with
    strict = True, errors will be computed for all of them.
    """
    moments = (2, 7.2, 37.5, 50, 800)
    ph, errors = fit_acph2(moments, strict=True)
    
    # Check that first three moments were fitted properly:
    assert_allclose(errors[:3], (0, 0, 0), atol=1e-5)
    for i in range(3):
        assert_allclose(ph.moment(i+1), moments[i], rtol=1e-5)
    
    # Check that other errors for other moments were estimated in some way:
    assert len(errors) == len(moments)
    assert all(errors[3:] > 0)
