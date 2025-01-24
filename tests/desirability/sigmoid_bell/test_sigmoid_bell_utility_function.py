import math

import numpy as np
import pytest

from pumas.desirability import desirability_catalogue


@pytest.fixture
def utility_function():
    desirability_class = desirability_catalogue.get("sigmoid_bell")
    desirability_instance = desirability_class()
    return desirability_instance.utility_function


@pytest.mark.parametrize("x", [0.0, 100.0, -100, 1e6, 1e-6])
@pytest.mark.parametrize(
    "x1, x2, x3, x4, k, base, invert, shift",
    [
        (0, 1, 2, 3, 1, 10, False, 0),
        (-10, 0, 10, 20, 2, 5, False, 0),
        (-100, -50, 50, 100, 3, 20, True, 0.1),
        (0, 1e3, 2e3, 3e3, 0.5, 2, False, 0.5),
        (-1e6, -1e3, 1e3, 1e6, 5, 100, True, 0.9),
        (0, 1, 2, 3, 1e23, 1.1, False, 0),
    ],
)
def test_sigmoid_bell_numerical_stability(
    utility_function, x, x1, x2, x3, x4, k, base, invert, shift
):
    """
    Test the numerical stability of the sigmoid_bell utility function.

    Hypothesis:
    The function should return valid results (non-None, non-NaN, non-infinite, and within [0, 1])
    for a wide range of input parameters, including extreme values.
    """  # noqa E501
    result = utility_function(x, x1, x2, x3, x4, k, base, invert, shift)
    assert result is not None, "Result is None"
    assert not math.isnan(result), "Result is NaN"
    assert not math.isinf(result), "Result is infinity"
    assert 0.0 <= result <= 1.0, f"Result {result} is outside the range [0, 1]"


def test_sigmoid_bell_peak(utility_function):
    """
    Test that the sigmoid_bell utility function peaks between x2 and x3.

    Hypothesis:
    The function should reach its maximum value (close to 1.0) between x2 and x3.
    """
    x1, x2, x3, x4, k, base = 0, 1, 2, 3, 2, 10
    peak_value = max(
        utility_function(x, x1, x2, x3, x4, k, base) for x in np.linspace(x2, x3, 100)
    )
    assert math.isclose(
        peak_value, 1.0, rel_tol=1e-2
    ), "Peak not close to 1.0 between x2 and x3"


def test_sigmoid_bell_invert(utility_function):
    """
    Test the invert parameter of the sigmoid_bell utility function.

    Hypothesis:
    When inverted, the function should return 1 - f(x) where f(x) is the non-inverted function.
    """  # noqa E501
    x, x1, x2, x3, x4, k, base = 1.5, 0, 1, 2, 3, 2, 10
    normal_value = utility_function(x, x1, x2, x3, x4, k, base, invert=False)
    inverted_value = utility_function(x, x1, x2, x3, x4, k, base, invert=True)
    assert math.isclose(
        normal_value + inverted_value, 1.0, rel_tol=1e-9
    ), "Invert not working correctly"


def test_sigmoid_bell_shift(utility_function):
    """
    Test the shift parameter of the sigmoid_bell utility function.

    Hypothesis:
    1. In the unshifted case, there should be values below the shift value.
    2. In the shifted case, all values should be:
       a) Higher than or equal to their corresponding unshifted values.
       b) Higher than or equal to the shift value.

    This test calculates the function over a range of x values to verify these properties.
    """  # noqa E501
    x1, x2, x3, x4, k, base = 0, 1, 2, 3, 2, 10
    shift_value = 0.2
    x_range = np.linspace(x1 - 1, x4 + 1, 100)  # Extend range slightly beyond x1 and x4

    # Calculate reference values
    reference_values = [
        utility_function(x, x1, x2, x3, x4, k, base, shift=0.0) for x in x_range
    ]

    # Verify that there are unshifted values below the shift value
    assert any(
        v < shift_value for v in reference_values
    ), "No unshifted values below shift value"

    # Calculate shifted values
    shifted_values = [
        utility_function(x, x1, x2, x3, x4, k, base, shift=shift_value) for x in x_range
    ]

    # Verify properties of shifted values
    for unshifted, shifted in zip(reference_values, shifted_values):
        assert (
            shifted >= unshifted
        ), "Shifted value not higher than or equal to unshifted value"
        assert (
            shifted >= shift_value
        ), "Shifted value not higher than or equal to shift value"


def test_sigmoid_bell_k_effect(utility_function):
    """
    Test the effect of the k parameter on the sigmoid_bell utility function.

    Hypothesis:
    A larger k should result in a steeper transition between low and high values.
    """
    x, x1, x2, x3, x4, base = 1.5, 0, 1, 2, 3, 10
    shallow = utility_function(x, x1, x2, x3, x4, k=1, base=base)
    steep = utility_function(x, x1, x2, x3, x4, k=5, base=base)
    assert shallow != steep, "k parameter not affecting curve steepness"


def test_sigmoid_bell_base_effect(utility_function):
    """
    Test the effect of the base parameter on the sigmoid_bell utility function.

    Hypothesis:
    A larger base should result in a more gradual transition between low and high values.
    """  # noqa E501
    x, x1, x2, x3, x4, k = 1.5, 0, 1, 2, 3, 2
    gradual = utility_function(x, x1, x2, x3, x4, k, base=2)
    sharp = utility_function(x, x1, x2, x3, x4, k, base=10)
    assert gradual != sharp, "base parameter not affecting curve shape"


def test_sigmoid_bell_invalid_parameters(utility_function):
    """
    Test that the function raises ValueError for invalid parameter combinations.

    Hypothesis:
    The function should raise ValueError when x1, x2, x3, x4 are
    not in ascending order or when base <= 1.
    """
    with pytest.raises(ValueError):
        utility_function(1, 2, 1, 3, 4, 1, 10)  # x2 < x1
    with pytest.raises(ValueError):
        utility_function(1, 1, 2, 4, 3, 1, 10)  # x4 < x3
    with pytest.raises(ValueError):
        utility_function(1, 1, 2, 3, 4, 1, 1)  # base <= 1


def test_sigmoid_bell_extreme_values(utility_function):
    """
    Test the behavior of the sigmoid_bell utility function for extreme x values.

    Hypothesis:
    The function should approach 0 for x values far below x1 and far above x4.
    """
    x1, x2, x3, x4, k, base = 0, 1, 2, 3, 2, 10
    assert math.isclose(
        utility_function(-1e6, x1, x2, x3, x4, k, base), 0, abs_tol=1e-6
    )
    assert math.isclose(utility_function(1e6, x1, x2, x3, x4, k, base), 0, abs_tol=1e-6)


@pytest.mark.parametrize(
    "x1, x2, x3, x4",
    [
        (0, 1, 2, 3),
        (-10, -5, 5, 10),
        (0, 10, 20, 30),
    ],
)
def test_sigmoid_bell_monotonicity(utility_function, x1, x2, x3, x4):
    """
    Test the monotonicity of the sigmoid_bell utility function in different regions.

    Hypothesis:
    The function should be:
    1. Increasing from x1 to (x2+x3)/2
    2. Decreasing from (x2+x3)/2 to x4
    """
    k, base = 2, 10
    center = (x2 + x3) / 2
    x_values = np.linspace(x1, center, 100)
    y_values = [utility_function(x, x1, x2, x3, x4, k, base) for x in x_values]
    assert all(
        y1 <= y2 for y1, y2 in zip(y_values, y_values[1:])
    ), "Not monotonically increasing from x1 to center"

    x_values = np.linspace(center, x4, 100)
    y_values = [utility_function(x, x1, x2, x3, x4, k, base) for x in x_values]
    assert all(
        y1 >= y2 for y1, y2 in zip(y_values, y_values[1:])
    ), "Not monotonically decreasing from center to x4"
