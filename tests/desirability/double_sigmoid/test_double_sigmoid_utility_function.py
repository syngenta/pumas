import math

import numpy as np
import pytest

from pumas.desirability.double_sigmoid import double_sigmoid


@pytest.fixture
def utility_function():
    return double_sigmoid


# Numerical stability tests
@pytest.mark.parametrize("x", [0.0, 100.0, -100.0, 1e6, 1e-6])
@pytest.mark.parametrize(
    "low, high, coef_div, coef_si, coef_se, base, invert, shift",
    [
        (0.0, 1.0, 1.0, 1.0, 1.0, 10.0, False, 0.0),
        (-1.0, 1.0, 1.0, 1.0, 1.0, 10.0, False, 0.0),
        (0.0, 1000.0, 0.1, 1.0, 1.0, 10.0, False, 0.0),
        (-1000.0, 1000.0, 0.01, 10.0, 10.0, 10.0, False, 0.0),
        (0.0, 1.0, 1000.0, 1.0, 1.0, 10.0, False, 0.0),
        (0.0, 1.0, 1.0, 1000.0, 1000.0, 10.0, False, 0.0),
        (0.0, 1.0, 1.0, 1.0, 1.0, 10.0, False, 0.5),
        (0.0, 1.0, 1.0, 1.0, 1.0, 2.0, False, 0.0),
        (0.0, 1e6, 1.0, 1.0, 1.0, 10.0, False, 0.0),
        (-1e6, 1e6, 1e-6, 1e6, 1e6, 10.0, False, 0.0),
    ],
)
def test_double_sigmoid_numerical_stability(
    utility_function, x, low, high, coef_div, coef_si, coef_se, base, invert, shift
):
    """
    Test the numerical stability of the double sigmoid function.

    Hypothesis:
    The function should remain stable and return valid results for a wide range of input parameters,
    including extreme values and edge cases.

    This test verifies that:
    1. The function always returns a non-None result.
    2. The result is neither NaN nor infinity.
    3. The result is always within the valid range [0, 1].
    """  # noqa E501
    result = utility_function(
        x, low, high, coef_div, coef_si, coef_se, base, invert, shift
    )
    assert result is not None, "Result is None"
    assert not math.isnan(result), "Result is NaN"
    assert not math.isinf(result), "Result is infinity"
    assert 0 <= result <= 1, f"Result {result} is outside the range [0, 1]"


# Functional tests
def test_double_sigmoid_center(utility_function):
    """
    Test the behavior of the double sigmoid function at its center point.

    Hypothesis:
    1. The function should reach its maximum value at the center point.
    2. The maximum value should be very close to 1.0, but may not be exactly 1.0 due to
       floating-point precision limitations.

    This test verifies these properties by:
    1. Checking that the value at the center is very close to 1.0 (within a relaxed tolerance).
    2. Verifying that the center value is the maximum compared to points slightly to its left and right.
    """  # noqa E501
    low, high = 0, 10
    center = (low + high) / 2
    result_center = utility_function(center, low, high, 1, 1, 1)
    result_left = utility_function(center - 0.1, low, high, 1, 1, 1)
    result_right = utility_function(center + 0.1, low, high, 1, 1, 1)

    assert math.isclose(result_center, 1.0, rel_tol=1e-5), (
        f"Function should be very close to 1.0 at the center. "
        f"Actual value: {result_center}"
    )
    assert result_center > result_left and result_center > result_right, (
        f"Center value ({result_center}) should be greater than "
        f"left ({result_left}) and right ({result_right}) values"
    )


def test_double_sigmoid_boundaries(utility_function):
    """
    Test the behavior of the double sigmoid function at its boundaries.

    Hypothesis:
    The function should return 0.5 at both the lower and upper boundaries.

    This test verifies that the function returns 0.5 (within a small tolerance) when x equals low or high.
    """  # noqa E501
    low, high = 0, 10
    result_low = utility_function(low, low, high, 1, 1, 1)
    result_high = utility_function(high, low, high, 1, 1, 1)
    assert math.isclose(
        result_low, 0.5, rel_tol=1e-9
    ), "Function should be 0.5 at the lower boundary"
    assert math.isclose(
        result_high, 0.5, rel_tol=1e-9
    ), "Function should be 0.5 at the upper boundary"


def test_double_sigmoid_symmetry(utility_function):
    """
    Test the symmetry of the double sigmoid function.

    Hypothesis:
    The function should be symmetric around its center point when all coefficients are equal.

    This test verifies that for any point x, f(center - x) = f(center + x) within a small tolerance.
    """  # noqa E501
    low, high = 0, 10
    center = (low + high) / 2
    for x in np.linspace(low, high, 100):
        y1 = utility_function(center - x, low, high, 1, 1, 1)
        y2 = utility_function(center + x, low, high, 1, 1, 1)
        assert math.isclose(y1, y2, rel_tol=1e-9), f"Function not symmetric for x={x}"


def test_double_sigmoid_invert(utility_function):
    """
    Test the invert parameter of the double sigmoid function.

    Hypothesis:
    When the invert parameter is True, the function should return 1 - f(x) where f(x) is the non-inverted function.

    This test verifies that for any input x, f(x, invert=False) + f(x, invert=True) = 1.
    """  # noqa E501
    x, low, high = 5, 0, 10
    normal = utility_function(x, low, high, 1, 1, 1, invert=False)
    inverted = utility_function(x, low, high, 1, 1, 1, invert=True)
    assert math.isclose(
        normal + inverted, 1.0, rel_tol=1e-9
    ), "Invert not working correctly"


def test_double_sigmoid_shift(utility_function):
    """
    Test the shift parameter of the double sigmoid function.

    Hypothesis:
    The shift parameter should move the entire curve up by the specified amount.

    This test verifies that for any input x, f(x, shift=s) > f(x, shift=0) when s > 0.
    """  # noqa E501
    x, low, high = 0, 0, 10
    no_shift = utility_function(x, low, high, 1, 1, 1, shift=0.0)
    shifted = utility_function(x, low, high, 1, 1, 1, shift=0.2)
    assert shifted > no_shift, "Shift not working correctly"


@pytest.mark.parametrize("coef_div", [0.1, 1, 10])
def test_double_sigmoid_coef_div_impact(utility_function, coef_div):
    """
    Test the impact of coef_div on the steepness of the double sigmoid curve.

    Hypothesis:
    coef_div should affect the overall steepness of the curve, with higher values resulting in a steeper curve.

    This test verifies that the steepness of the curve is consistent on both sides of the center point for different coef_div values.
    """  # noqa E501
    low, high = 0, 10
    center = (low + high) / 2
    y1 = utility_function(center - 1, low, high, coef_div, 1, 1)
    y2 = utility_function(center + 1, low, high, coef_div, 1, 1)
    assert math.isclose(
        y1, y2
    ), f"coef_div={coef_div} not affecting curve steepness correctly"


@pytest.mark.parametrize("coef_si, coef_se", [(0.1, 10), (1, 1), (10, 0.1)])
def test_double_sigmoid_coef_si_se_impact(utility_function, coef_si, coef_se):
    """
    Test the impact of coef_si and coef_se on the double sigmoid curve's shape.

    Hypothesis:
    1. coef_si affects the steepness of the left side of the curve (for x < center).
    2. coef_se affects the steepness of the right side of the curve (for x >= center).
    3. A higher coefficient value results in a steeper curve on its respective side.

    This test verifies these properties by comparing the steepness of the curve
    near the center point on both sides and checking for symmetry or asymmetry as appropriate.
    """  # noqa E501
    low, high = 0, 10
    center = (low + high) / 2
    y_left = utility_function(low, low, high, 1, coef_si, coef_se)
    y_center_left = utility_function(center - 0.1, low, high, 1, coef_si, coef_se)
    y_center_right = utility_function(center + 0.1, low, high, 1, coef_si, coef_se)
    y_right = utility_function(high, low, high, 1, coef_si, coef_se)

    left_steepness = abs(y_center_left - y_left)
    right_steepness = abs(y_center_right - y_right)

    if coef_si == coef_se:
        assert math.isclose(
            left_steepness, right_steepness, rel_tol=1e-9
        ), f"Curve should be symmetric when coef_si={coef_si} and coef_se={coef_se}"
    else:
        assert (left_steepness > right_steepness) == (coef_si > coef_se), (
            f"coef_si={coef_si} and coef_se={coef_se} "
            f"not affecting curve asymmetry correctly"
        )

    expected = 0.5
    assert math.isclose(
        y_left, expected, rel_tol=1e-9
    ), f"Function should be {expected} at the lower boundary"
    assert math.isclose(
        y_right, expected, rel_tol=1e-9
    ), f"Function should be {expected} at the upper boundary"


def test_double_sigmoid_zero_coef_div_impact(utility_function):
    """
    Test the double sigmoid function when coef_div (k) is zero.

    Hypothesis:
    1. The function should use a hard sigmoid when coef_div is zero.
    2. The function should behave consistently for x < x_center and x >= x_center.
    3. No errors should be raised for a zero coef_div.

    This test verifies these properties by checking the function's behavior
    at points on both sides of the center and at the center itself.
    """  # noqa E501
    low, high = 0, 10
    center = (low + high) / 2
    coef_div = 0
    coef_si = 1
    coef_se = 1

    result_left = utility_function(center - 1, low, high, coef_div, coef_si, coef_se)
    result_right = utility_function(center + 1, low, high, coef_div, coef_si, coef_se)
    result_center = utility_function(center, low, high, coef_div, coef_si, coef_se)

    assert 0 <= result_left <= 1, f"Result should be between 0 and 1, got {result_left}"
    assert (
        0 <= result_right <= 1
    ), f"Result should be between 0 and 1, got {result_right}"
    assert (
        0 <= result_center <= 1
    ), f"Result should be between 0 and 1, got {result_center}"
    assert math.isclose(
        result_left, result_right
    ), "Function should behave consistently on both sides of the center"


@pytest.mark.parametrize("base", [2, 10, 100])
def test_double_sigmoid_base_impact(utility_function, base):
    """
    Test the impact of the base parameter on the double sigmoid curve's shape.

    Hypothesis:
    A higher base value should result in a steeper curve near the center.

    This test verifies that increasing the base value leads to a steeper curve by comparing
    function values at a fixed distance from the center for different base values.
    """  # noqa E501
    low, high = 0, 10
    center = (low + high) / 2
    y_lower_base = utility_function(center - 1, low, high, 1, 1, 1, base=base)
    y_higher_base = utility_function(center - 1, low, high, 1, 1, 1, base=base * 10)
    assert y_lower_base < y_higher_base, (
        f"Base={base} " f"not affecting curve steepness correctly"
    )


# Edge case tests
def test_double_sigmoid_zero_coef_div(utility_function):
    """
    Test the double sigmoid function with zero coef_div.

    Hypothesis:
    The function should handle a zero coef_div gracefully and return a valid result within the [0, 1] range.

    This test verifies that:
    1. The function does not raise an error when coef_div is zero.
    2. The returned value is within the valid range of [0, 1].
    """  # noqa E501
    low, high = 0, 10
    result = utility_function(5, low, high, 0, 1, 1)
    assert 0 <= result <= 1, "Function should handle zero coef_div gracefully"


def test_double_sigmoid_equal_low_high(utility_function):
    """
    Test the double sigmoid function when low and high are equal.

    Hypothesis:
    The function should handle the case where low and high are equal without errors
    and return a valid result within the [0, 1] range.

    This test verifies that:
    1. The function does not raise an error when low and high are equal.
    2. The returned value is within the valid range of [0, 1].

    Note: The expected behavior in this case might need to be defined more precisely
    based on the intended use of the function.
    """  # noqa E501
    low = high = 5
    result = utility_function(5, low, high, 1, 1, 1)
    assert 0 <= result <= 1, "Function should handle equal low and high gracefully"


def test_double_sigmoid_extreme_x_values(utility_function):
    """
    Test the double sigmoid function with extreme x values (positive and negative infinity).

    Hypothesis:
    1. As x approaches negative infinity, the function should approach 0.
    2. As x approaches positive infinity, the function should approach 0.

    This test verifies that:
    1. The function returns a value very close to 0 for extremely low x values.
    2. The function returns a value very close to 0 for extremely high x values.
    3. The function handles these extreme inputs without raising errors.

    Note: The exact behavior at infinity might depend on the specific implementation
    and could be subject to floating-point arithmetic limitations.
    """  # noqa E501
    low, high = 0, 10
    result_low = utility_function(float("-inf"), low, high, 1, 1, 1)
    result_high = utility_function(float("inf"), low, high, 1, 1, 1)
    assert math.isclose(
        result_low, 0.0, abs_tol=1e-9
    ), "Function should approach 0 for extreme low x values"
    assert math.isclose(
        result_high, 0.0, abs_tol=1e-9
    ), "Function should approach 0 for extreme high x values"
