import math

import numpy as np
import pytest

from pumas.desirability import desirability_catalogue


@pytest.fixture
def utility_function():
    desirability_class = desirability_catalogue.get("sigmoid")
    desirability_instance = desirability_class()
    return desirability_instance.utility_function


@pytest.mark.parametrize("x", [0.0, 100.0, -100, 1e6, 1e-6])
@pytest.mark.parametrize(
    "low, high, k, shift, base",
    [
        (0, 1, 1, 0, 10),
        (0, 1, 1000, 0, 10),
        (-1000, 1000, 0.1, 0, 10),
        (0, 1e6, 1, 0, 10),
        (0, 1, 1e6, 0, 10),
    ],
)
def test_sigmoid_numerical_stability(utility_function, x, low, high, k, shift, base):
    """
    Test the numerical stability of the sigmoid utility function.

    Hypothesis:
    The function should return valid results (non-None, non-NaN, non-infinite, and within [0, 1])
    for a wide range of input parameters, including extreme values.

    This test verifies that the function remains stable across various input combinations.
    """  # noqa E501
    result = utility_function(x, low, high, k, shift, base)
    assert result is not None, "Result is None"
    assert not math.isnan(result), "Result is NaN"
    assert not math.isinf(result), "Result is infinity"
    assert 0.0 <= result <= 1.0, f"Result {result} is outside the range [0, 1]"


@pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("k", [-1.0, 1.0])
def test_hard_sigmoid(utility_function, x, k):
    """
    Test the sigmoid function's behavior as a hard sigmoid when low and high are both zero.

    Hypothesis:
    When low = high = 0, the function should behave as a hard sigmoid:
    - Return 0 when k*x <= 0
    - Return 1 when k*x > 0

    This test verifies the hard sigmoid behavior for different x and k values.
    """  # noqa E501
    result = utility_function(x, 0.0, 0.0, k, shift=0.0, base=10.0)
    expected = float(k * x > 0)
    assert result == expected, f"Hard sigmoid failed for x={x}, k={k}"


@pytest.mark.parametrize("base", [2.0, np.e, 10.0])
def test_sigmoid_different_bases(utility_function, base):
    """
    Test the sigmoid function with different base values.

    Hypothesis:
    The function should produce valid results (between 0 and 1) for different base values.

    This test verifies that the function works correctly with various common bases (2, e, 10).
    """  # noqa E501
    x, low, high, k = 0.5, 0.0, 1.0, 1.0
    result = utility_function(x, low, high, k, shift=0.0, base=base)
    assert 0 < result < 1, f"Sigmoid with base {base} failed"


@pytest.mark.parametrize("shift", [0.0, 0.1, 0.5])
def test_sigmoid_shift(utility_function, shift):
    """
    Test the shift parameter of the sigmoid function.

    Hypothesis:
    The shift parameter should move the entire sigmoid curve up by the specified amount.

    This test verifies that the function's output is always greater than or equal to the shift value
    and less than or equal to 1.0.
    """  # noqa E501
    x, low, high, k = 0.5, 0.0, 1.0, 1.0
    result = utility_function(x, low, high, k, shift=shift, base=10.0)
    assert shift <= result <= 1.0, f"Shift {shift} not applied correctly"


@pytest.mark.parametrize("x", [-1e6, 1e6])
def test_sigmoid_extreme_values(utility_function, x):
    """
    Test the sigmoid function's behavior for extreme input values.

    Hypothesis:
    For very large negative x, the function should approach 0.
    For very large positive x, the function should approach 1.

    This test verifies the asymptotic behavior of the sigmoid function.
    """  # noqa E501
    low, high, k = -1.0, 1.0, 1.0
    result = utility_function(x, low, high, k, shift=0.0, base=10.0)
    expected = 0.0 if x < 0 else 1.0
    assert math.isclose(
        result, expected, abs_tol=1e-6
    ), f"Extreme value test failed for x={x}"


@pytest.mark.parametrize(
    "x, low, high, k, shift, base, error_type, error_msg",
    [
        (0.5, 0.0, 1.0, 1.0, 0.0, 1.0, ValueError, "Base must be greater than 1"),
        (0.5, 0.0, 1.0, 1.0, 0.0, 0.5, ValueError, "Base must be greater than 1"),
        (0.5, 0.0, 1.0, 1.0, -0.1, 10.0, ValueError, "Shift must be between 0 and 1"),
        (0.5, 0.0, 1.0, 1.0, 1.1, 10.0, ValueError, "Shift must be between 0 and 1"),
        (
            0.5,
            1.0,
            0.0,
            1.0,
            0.0,
            10.0,
            ValueError,
            "High must be greater than or equal to low",
        ),
    ],
)
def test_sigmoid_raises_error(
    utility_function, x, low, high, k, shift, base, error_type, error_msg
):
    """
    Test that the sigmoid function raises appropriate errors for invalid inputs.

    Hypothesis:
    The function should raise ValueError with appropriate error messages for:
    - Base values less than or equal to 1
    - Shift values outside the range [0, 1]
    - High values less than low values

    This test verifies that the function properly validates its inputs and raises
    appropriate exceptions for invalid parameter combinations.
    """  # noqa E501
    with pytest.raises(error_type, match=error_msg):
        utility_function(x=x, low=low, high=high, k=k, shift=shift, base=base)
