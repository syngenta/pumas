import math
import sys

import numpy as np
import pytest

from pumas.desirability import desirability_catalogue
from pumas.desirability.bell import (
    get_bell_inflection_points,
    get_bell_slope_pivot_points,
)


@pytest.fixture
def utility_function():
    desirability_class = desirability_catalogue.get("bell")
    desirability_instance = desirability_class()
    return desirability_instance.utility_function


@pytest.mark.parametrize("x", [0.0, 100.0, -100, 1e6, 1e-6])
@pytest.mark.parametrize(
    "width, slope, center, invert, shift",
    [
        (1, 1, 0, False, 0),
        (10, 2, 50, False, 0),
        (0.1, 3, -10, True, 0.1),
        (1e3, 0.5, 0, False, 0.5),
        (1e-3, 5, 1e6, True, 0.9),
        (1, 1e23, 0, False, 0),
    ],
)
def test_bell_numerical_stability(
    utility_function, x, width, slope, center, invert, shift
):
    """
    Test the numerical stability of the bell utility function.

    Hypothesis:
    The function should return valid results (non-None, non-NaN, non-infinite, and within [0, 1])
    for a wide range of input parameters, including extreme values.

    This test verifies that the function remains stable across various input combinations.
    """  # noqa E501
    result = utility_function(x, width, slope, center, invert, shift)
    assert result is not None, "Result is None"
    assert not math.isnan(result), "Result is NaN"
    assert not math.isinf(result), "Result is infinity"
    assert 0.0 <= result <= 1.0, f"Result {result} is outside the range [0, 1]"


@pytest.mark.parametrize("x", [float(x) for x in range(-20, 21, 5)])
@pytest.mark.parametrize(
    "width, slope, center",
    [
        (1.0, 1.0, 0.0),
        (2.0, 2.0, 5.0),
        (0.5, 3.0, -5.0),
        (5.0, 0.5, 10.0),
    ],
)
def test_bell_symmetry(utility_function, x, width, slope, center):
    """
    Test the symmetry of the bell utility function.

    Hypothesis:
    The function should be symmetric around its center point.

    This test verifies that f(center - x) = f(center + x) for various input combinations.
    """  # noqa E501
    y1 = utility_function(center - x, width, slope, center)
    y2 = utility_function(center + x, width, slope, center)
    assert math.isclose(y1, y2, rel_tol=1e-9), f"Function not symmetric for x={x}"


def test_bell_peak_at_center(utility_function):
    """
    Test that the bell utility function peaks at its center.

    Hypothesis:
    The function should reach its maximum value (1.0) at the center point.

    This test verifies that f(center) = 1.0 for a specific set of parameters.
    """  # noqa E501
    width, slope, center = 1.0, 2.0, 0.5
    peak_value = utility_function(center, width, slope, center)
    assert math.isclose(
        peak_value, 1.0, rel_tol=1e-9
    ), "Peak not at 1.0 for center value"


def test_bell_invert(utility_function):
    """
    Test the invert parameter of the bell utility function.

    Hypothesis:
    When inverted, the function should return 1 - f(x) where f(x) is the non-inverted function.

    This test verifies that f(x, invert=False) + f(x, invert=True) = 1.0 for a specific set of parameters.
    """  # noqa E501
    x, width, slope, center = 0.5, 1.0, 2.0, 0.5
    normal_value = utility_function(x, width, slope, center, invert=False)
    inverted_value = utility_function(x, width, slope, center, invert=True)
    assert math.isclose(
        normal_value + inverted_value, 1.0, rel_tol=1e-9
    ), "Invert not working correctly"


def test_bell_shift(utility_function):
    """
    Test the shift parameter of the bell utility function.

    Hypothesis:
    1. In the unshifted case, there should be values below the shift value.
    2. In the shifted case, all values should be:
       a) Higher than or equal to their corresponding unshifted values.
       b) Higher than or equal to the shift value.

    This test calculates the function over a range of x values to verify these properties.
    """  # noqa E501
    width, slope, center = 1.0, 2.0, 0.5
    shift_value = 0.2
    x_range = np.linspace(center - 3 * width, center + 3 * width, 100)

    # Calculate reference values
    reference_values = [
        utility_function(x, width, slope, center, shift=0.0) for x in x_range
    ]

    # Verify that there are unshifted values below the shift value
    assert any(
        v < shift_value for v in reference_values
    ), "No unshifted values below shift value"

    # Calculate shifted values
    shifted_values = [
        utility_function(x, width, slope, center, shift=shift_value) for x in x_range
    ]

    # Verify properties of shifted values
    for unshifted, shifted in zip(reference_values, shifted_values):
        assert (
            shifted >= unshifted
        ), "Shifted value not higher than or equal to unshifted value"
        assert (
            shifted >= shift_value
        ), "Shifted value not higher than or equal to shift value"

    # Verify that all shifted values are above or at the shift value
    assert all(
        v >= shift_value for v in shifted_values
    ), "No unshifted values below shift value"


def test_bell_shift_on_center(utility_function):
    """
    Test the effect of shift on the center point of the bell utility function.

    Hypothesis:
    The shift parameter should not affect the value at the center point.

    This test verifies that f(center, shift=0) = f(center, shift=s) for s > 0.
    """  # noqa E501
    x, width, slope, center = 0.5, 1.0, 2.0, 0.5
    reference = utility_function(x, width, slope, center, shift=0.0)
    shifted = utility_function(x, width, slope, center, shift=0.2)
    assert math.isclose(shifted, reference), "Shift not working correctly"


def test_bell_width_effect(utility_function):
    """
    Test the effect of the width parameter on the bell utility function.

    Hypothesis:
    A larger width should result in a wider bell curve.

    This test verifies that f(x, width=w1) < f(x, width=w2) when w1 < w2 and x is not at the center.
    """  # noqa E501
    x, slope, center = 1.0, 2.0, 0.5
    narrow = utility_function(x, width=0.5, slope=slope, center=center)
    wide = utility_function(x, width=2.0, slope=slope, center=center)
    assert narrow < wide, "Width parameter not affecting curve width correctly"


def test_width_close_to_zero(utility_function):
    """
    Test that the function raises a ValueError when width is too close to zero.

    Hypothesis:
    The function should raise a ValueError to prevent numerical instability when width is extremely small.
    """  # noqa E501
    with pytest.raises(
        ValueError,
        match="Width is too close to zero, which may cause numerical instability.",
    ):
        utility_function(
            x=0.5, center=0.5, width=sys.float_info.epsilon / 2, slope=1, shift=0
        )


@pytest.mark.parametrize(
    "x, center, width, slope, shift, error_type, error_msg",
    [
        (
            0.5,
            0.5,
            sys.float_info.epsilon / 2,
            1,
            0,
            ValueError,
            "Width is too close to zero",
        ),
        (
            1.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            ValueError,
            "The 'width' parameter must be greater than 0",
        ),
        (
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            ValueError,
            "The 'width' parameter must be greater than 0",
        ),
        (
            1.0,
            0.0,
            2.0,
            1.0,
            2.0,
            ValueError,
            "The 'shift' parameter must be between 0 and 1, inclusive.",
        ),
    ],
)
def test_utility_function_raises_error(
    utility_function, x, center, width, slope, shift, error_type, error_msg
):
    """
    Test that the function raises appropriate errors for invalid inputs.

    Hypothesis:
    The function should raise specific errors with appropriate error messages for various invalid input combinations.
    """  # noqa E501
    with pytest.raises(error_type, match=error_msg):
        utility_function(x=x, center=center, width=width, slope=slope, shift=shift)


@pytest.mark.parametrize("slope", [1.0, 0.5])
def test_bell_slope_sign_effect(utility_function, slope):
    """
    Test the effect of the slope sign on the bell utility function.

    Hypothesis:
    The sign of the slope parameter should not affect the shape of the curve.

    This test verifies that f(x, slope=s) = f(x, slope=-s) for various slope values.
    """  # noqa E501
    x, width, center = 0.5, 1.0, 0.5
    p = utility_function(x=x, width=width, slope=slope, center=center)
    n = utility_function(x=x, width=width, slope=-1 * slope, center=center)
    assert math.isclose(p, n), "Slope sign not affecting curve direction correctly"


@pytest.mark.parametrize("x", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
@pytest.mark.parametrize("slope", [1.0, 0.5])
@pytest.mark.parametrize("width", [0.2, 0.3])
@pytest.mark.parametrize("center", [0.3, 0.5, 0.7])
def test_bell_slope_effect(utility_function, x, slope, width, center):
    """
    Test the effect of the slope parameter on the bell utility function.

    Hypothesis:
    1. At pivot points, the slope parameter should not affect the function value.
    2. Outside pivot points, steeper slopes should result in lower values.
    3. Inside pivot points (but not at center), steeper slopes should result in higher values.
    4. At the center, the function value should be independent of the slope.

    This test verifies these properties for various input combinations.
    """  # noqa E501
    a, b = get_bell_slope_pivot_points(center=center, width=width, slope=slope)

    ref = utility_function(x, width=width, slope=slope, center=center)
    steeper = utility_function(x, width=width, slope=slope + slope * 0.2, center=center)
    flatter = utility_function(x, width=width, slope=slope - slope * 0.2, center=center)

    if math.isclose(x, a) or math.isclose(x, b):
        assert math.isclose(
            steeper, ref
        ), "Slope parameter not affecting curve steepness correctly"
        assert math.isclose(
            flatter, ref
        ), "Slope parameter not affecting curve steepness correctly"
    elif x < a or x > b:
        assert steeper < ref, "Slope parameter not affecting curve steepness correctly"
        assert flatter > ref, "Slope parameter not affecting curve steepness correctly"
    elif a < x < b and not math.isclose(x, center):
        assert steeper > ref, "Slope parameter not affecting curve steepness correctly"
        assert flatter < ref, "Slope parameter not affecting curve steepness correctly"
    elif a < x < b and math.isclose(x, center):
        assert math.isclose(
            steeper, ref
        ), "Slope parameter not affecting curve steepness correctly"
        assert math.isclose(
            flatter, ref
        ), "Slope parameter not affecting curve steepness correctly"


def test_bell_invalid_width(utility_function):
    """
    Test that the function raises a ValueError for invalid width values.

    Hypothesis:
    The function should raise a ValueError for non-positive width values.
    """  # noqa E501
    with pytest.raises(ValueError):
        utility_function(x=0.5, width=0, slope=1.0, center=0.5)
    with pytest.raises(ValueError):
        utility_function(x=0.5, width=-1.0, slope=1.0, center=0.5)


def test_bell_invalid_shift(utility_function):
    """
    Test that the function raises a ValueError for invalid shift values.

    Hypothesis:
    The function should raise a ValueError for shift values outside the [0, 1] range.
    """
    with pytest.raises(ValueError):
        utility_function(x=0.5, width=1.0, slope=1.0, center=0.5, shift=-0.1)
    with pytest.raises(ValueError):
        utility_function(x=0.5, width=1.0, slope=1.0, center=0.5, shift=1.1)


def test_bell_extreme_values(utility_function):
    """
    Test the behavior of the bell utility function for extreme x values.

    Hypothesis:
    The function should approach 0 for x values far from the center.

    This test verifies that f(x) ≈ 0 when x is very large compared to the center and width.
    """  # noqa E501
    x, width, slope, center = 1e6, 1.0, 2.0, 0.0
    result = utility_function(x, width, slope, center)
    assert math.isclose(
        result, 0.0, abs_tol=1e-9
    ), "Function not approaching 0 for extreme x values"


@pytest.mark.parametrize(
    "center, width, slope",
    [(0.5, 1.0, 1.0), (0.5, 1.0, 2.0), (0.3, 0.5, 1.5), (0.7, 0.8, 2.5)],
)
def test_bell_inflection_points_symmetry(utility_function, center, width, slope):
    """
    Test the symmetry of the bell utility function at its inflection points.

    Hypothesis:
    The function values at the two inflection points should be equal.

    This test verifies that f(xa) = f(xb) where xa and xb are the inflection points.
    """  # noqa E501
    xa, xb = get_bell_inflection_points(center, width, slope)
    ya = utility_function(xa, width, slope, center)
    yb = utility_function(xb, width, slope, center)
    assert math.isclose(ya, yb), "Function values at inflection points are not equal"


@pytest.mark.parametrize("center, width", [(0.5, 1.0), (0.3, 0.5), (0.7, 0.8)])
@pytest.mark.parametrize("slope", [0.5, 1.0, 2.0, 5.0])
def test_bell_slope_pivot_points_symmetry(utility_function, center, width, slope):
    """
    Test the symmetry of the bell utility function at its slope pivot points.

    Hypothesis:
    The function values at the two slope pivot points should be equal and always 0.5.

    This test verifies that f(xa) = f(xb) = 0.5 where xa and xb are the slope pivot points.
    """  # noqa E501
    xa, xb = get_bell_slope_pivot_points(center, width, slope)

    ya = utility_function(xa, width, slope, center)
    yb = utility_function(xb, width, slope, center)

    assert math.isclose(ya, yb), "Function values at inflection points are not equal"
    assert math.isclose(ya, 0.5), "Function value at pivot point is not 0.5"


@pytest.mark.parametrize("center, width", [(0.5, 1.0), (0.3, 0.5), (0.7, 0.8)])
def test_bell_slope_pivot_points_slope_invariance(utility_function, center, width):
    """
    Test the slope invariance property of the bell utility function at its slope pivot points.

    Hypothesis:
    The function values at the slope pivot points should be invariant to changes in the slope parameter.

    This test verifies that:
    1. For any given slope, the function values at both pivot points are identical.
    2. These function values remain constant across different slope values.

    This property is important as it demonstrates that the slope parameter affects the shape of the curve
    without changing its fundamental characteristics at these specific points.
    """  # noqa E501
    slopes = [0.5, 1.0, 2.0]

    results = [
        (utility_function(xa, width, s, center), utility_function(xb, width, s, center))
        for s in slopes
        for xa, xb in [get_bell_slope_pivot_points(center=center, width=width, slope=s)]
    ]

    # First, check that both values in each tuple are identical
    assert all(
        math.isclose(ya, yb) for ya, yb in results
    ), "Not all pivot point pairs have identical values"

    # Then, check that all tuples are identical to each other
    assert all(
        math.isclose(results[0][0], result[0])
        and math.isclose(results[0][1], result[1])
        for result in results[1:]
    ), "Not all result tuples are identical"


@pytest.mark.parametrize(
    "center, width, slope", [(0.5, 1.0, 1.0), (0.3, 0.5, 1.5), (0.7, 0.8, 2.0)]
)
def test_invalid_inputs(center, width, slope):
    """
    Test that the bell utility function's helper functions properly handle invalid inputs.

    Hypothesis:
    The get_bell_inflection_points and get_bell_slope_pivot_points functions should raise ValueError
    for negative width or slope values.

    This test verifies that these helper functions properly validate their inputs and raise
    appropriate exceptions for invalid parameter combinations.
    """  # noqa E501
    with pytest.raises(ValueError):
        get_bell_inflection_points(center, -width, slope)
    with pytest.raises(ValueError):
        get_bell_inflection_points(center, width, -slope)
    with pytest.raises(ValueError):
        get_bell_slope_pivot_points(center, -width, slope)
    with pytest.raises(ValueError):
        get_bell_slope_pivot_points(center, width, -slope)
