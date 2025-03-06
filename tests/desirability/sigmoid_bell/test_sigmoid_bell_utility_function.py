import math

import numpy as np
import pytest

from pumas.desirability.sigmoid_bell import sigmoid_bell


@pytest.fixture
def desirability_utility_function():
    return sigmoid_bell


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
    desirability_utility_function, x, x1, x2, x3, x4, k, base, invert, shift
):
    """
    Test the numerical stability of the sigmoid_bell utility function.

    Hypothesis:
    The function should return valid results (non-None, non-NaN, non-infinite, and within [0, 1])
    for a wide range of input parameters, including extreme values.
    """  # noqa E501

    params = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "k": k, "base": base}

    result = desirability_utility_function(x=x, **params)

    assert result is not None, "Result is None"
    assert not math.isnan(result), "Result is NaN"
    assert not math.isinf(result), "Result is infinity"
    assert 0.0 <= result <= 1.0, f"Result {result} is outside the range [0, 1]"


def test_sigmoid_bell_peak(desirability_utility_function):
    """
    Test that the sigmoid_bell utility function peaks between x2 and x3.

    Hypothesis:
    The function should reach its maximum value (close to 1.0) between x2 and x3.
    """
    x1, x2, x3, x4, k, base = 0, 1, 2, 3, 2, 10
    params = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "k": k, "base": base}
    peak_value = max(
        desirability_utility_function(x=x, **params) for x in np.linspace(x2, x3, 100)
    )
    assert math.isclose(
        peak_value, 1.0, rel_tol=1e-2
    ), "Peak not close to 1.0 between x2 and x3"


def test_sigmoid_bell_invert(desirability_utility_function):
    """
    Test the invert parameter of the sigmoid_bell utility function.

    Hypothesis:
    When inverted, the function should return 1 - f(x) where f(x) is the non-inverted function.
    """  # noqa E501
    x, x1, x2, x3, x4, k, base = 1.5, 0, 1, 2, 3, 2, 10
    params_normal = {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "k": k,
        "base": base,
        "invert": False,
    }
    params_inverted = {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "k": k,
        "base": base,
        "invert": True,
    }
    normal_value = desirability_utility_function(x=x, **params_normal)
    inverted_value = desirability_utility_function(x=x, **params_inverted)
    assert math.isclose(
        normal_value + inverted_value, 1.0, rel_tol=1e-9
    ), "Invert not working correctly"


def test_sigmoid_bell_shift(desirability_utility_function):
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

    params_normal = {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "k": k,
        "base": base,
        "shift": 0.0,
    }
    params_shifted = {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "k": k,
        "base": base,
        "shift": shift_value,
    }
    x_range = np.linspace(x1 - 1, x4 + 1, 100)  # Extend range slightly beyond x1 and x4

    # Calculate reference values
    reference_values = [
        desirability_utility_function(x, **params_normal) for x in x_range
    ]

    # Verify that there are unshifted values below the shift value
    assert any(
        v < shift_value for v in reference_values
    ), "No unshifted values below shift value"

    # Calculate shifted values
    shifted_values = [
        desirability_utility_function(x=x, **params_shifted) for x in x_range
    ]

    # Verify properties of shifted values
    for unshifted, shifted in zip(reference_values, shifted_values):
        assert (
            shifted >= unshifted
        ), "Shifted value not higher than or equal to unshifted value"
        assert (
            shifted >= shift_value
        ), "Shifted value not higher than or equal to shift value"


def test_sigmoid_bell_k_effect(desirability_utility_function):
    """
    Test the effect of the k parameter on the sigmoid_bell utility function.

    Hypothesis:
    A larger k should result in a steeper transition between low and high values.
    """
    x, x1, x2, x3, x4, base = 1.5, 0, 1, 2, 3, 10
    params_shallow = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "k": 1.0, "base": base}
    params_steep = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "k": 5.0, "base": base}

    shallow = desirability_utility_function(x=x, **params_shallow)
    steep = desirability_utility_function(x=x, **params_steep)
    assert shallow != steep, "k parameter not affecting curve steepness"


def test_sigmoid_bell_base_effect(desirability_utility_function):
    """
    Test the effect of the base parameter on the sigmoid_bell utility function.

    Hypothesis:
    A larger base should result in a more gradual transition between low and high values.
    """  # noqa E501
    x, x1, x2, x3, x4, k = 1.5, 0, 1, 2, 3, 2
    params_gradual = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "k": k, "base": 2}
    params_sharp = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "k": k, "base": 10}

    gradual = desirability_utility_function(x=x, **params_gradual)
    sharp = desirability_utility_function(x=x, **params_sharp)
    assert gradual != sharp, "base parameter not affecting curve shape"


def test_sigmoid_bell_extreme_values(desirability_utility_function):
    """
    Test the behavior of the sigmoid_bell utility function for extreme x values.

    Hypothesis:
    The function should approach 0 for x values far below x1 and far above x4.
    """
    x1, x2, x3, x4, k, base = 0, 1, 2, 3, 2, 10
    params = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "k": k, "base": base}
    assert math.isclose(
        desirability_utility_function(x=-1e6, **params), 0, abs_tol=1e-6
    )
    assert math.isclose(desirability_utility_function(x=1e6, **params), 0, abs_tol=1e-6)


@pytest.mark.parametrize(
    "x1, x2, x3, x4",
    [
        (0, 1, 2, 3),
        (-10, -5, 5, 10),
        (0, 10, 20, 30),
    ],
)
def test_sigmoid_bell_monotonicity(desirability_utility_function, x1, x2, x3, x4):
    """
    Test the monotonicity of the sigmoid_bell utility function in different regions.

    Hypothesis:
    The function should be:
    1. Increasing from x1 to (x2+x3)/2
    2. Decreasing from (x2+x3)/2 to x4
    """
    k, base = 2, 10
    params = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "k": k, "base": base}
    center = (x2 + x3) / 2
    x_values = np.linspace(x1, center, 100)
    y_values = [desirability_utility_function(x=x, **params) for x in x_values]
    assert all(
        y1 <= y2 for y1, y2 in zip(y_values, y_values[1:])
    ), "Not monotonically increasing from x1 to center"

    x_values = np.linspace(center, x4, 100)
    y_values = [desirability_utility_function(x=x, **params) for x in x_values]
    assert all(
        y1 >= y2 for y1, y2 in zip(y_values, y_values[1:])
    ), "Not monotonically decreasing from center to x4"
