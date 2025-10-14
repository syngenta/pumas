import math

import numpy as np
import pytest

from pumas.desirability.exponential_decay import exponential_decay


@pytest.fixture
def utility_function():
    return exponential_decay


# Numerical stability tests
@pytest.mark.parametrize("x", [0.0, 100.0, -100.0, 1e6, 1e-6, -1e6])
@pytest.mark.parametrize(
    "k, shift",
    [
        (1.0, 0.0),
        (0.1, 0.0),
        (10.0, 0.0),
        (1.0, 0.0),
        (1.0, 0.5),
        (1.0, 0.5),
        (100.0, 0.0),
        (0.001, 0.0),
        (1.0, 0.9),
        (50.0, 0.1),
    ],
)
def test_exponential_decay_numerical_stability(utility_function, x, k, shift):
    """
    Test the numerical stability of the exponential decay function.

    Hypothesis:
    The function should remain stable and return valid results for a wide
    range of input parameters, including extreme values and edge cases.

    This test verifies that:
    1. The function always returns a non-None result.
    2. The result is neither NaN nor infinity.
    3. The result is always within the valid range [0, 1].
    """
    result = utility_function(x, k, shift)
    assert result is not None, "Result is None"
    assert not math.isnan(result), "Result is NaN"
    assert not math.isinf(result), "Result is infinity"
    assert 0 <= result <= 1, f"Result {result} is outside the range [0, 1]"


# Functional tests
def test_exponential_decay_at_zero(utility_function):
    """
    Test the behavior of the exponential decay function at x = 0.

    Hypothesis:
    The function should return 1.0 at x = 0 (full desirability).

    This test verifies that exp(-k * 0) = exp(0) = 1.0.
    """
    result = utility_function(0.0, k=1.0, shift=0.0)
    assert math.isclose(result, 1.0, rel_tol=1e-9), "Function should be 1.0 at x = 0"


def test_exponential_decay_at_one(utility_function):
    """
    Test the behavior of the exponential decay function at x = 1 with k = 1.

    Hypothesis:
    The function should return e^(-1) ≈ 0.3679 at x = 1 when k = 1.

    This test verifies the basic exponential decay behavior.
    """
    result = utility_function(1.0, k=1.0, shift=0.0)
    expected = math.exp(-1.0)
    assert math.isclose(
        result, expected, rel_tol=1e-9
    ), f"Function should be {expected} at x = 1 with k = 1"


def test_exponential_decay_negative_x_clamped(utility_function):
    """
    Test the behavior of the exponential decay function for negative x
    values.

    Hypothesis:
    The function should return 1.0 (clamped) for all x < 0.

    This test verifies the clamping behavior for negative inputs.
    """
    for x in [-1.0, -10.0, -100.0, -1e6]:
        result = utility_function(x, k=1.0, shift=0.0)
        assert math.isclose(
            result, 1.0, rel_tol=1e-9
        ), f"Function should be clamped to 1.0 for x = {x}"


def test_exponential_decay_monotonic_decrease(utility_function):
    """
    Test that the exponential decay function is monotonically decreasing
    for x >= 0.

    Hypothesis:
    For x >= 0, as x increases, the function value should decrease.

    This test verifies that f(x1) > f(x2) when x1 < x2 for x >= 0.
    """
    k = 1.0
    x_values = np.linspace(0, 10, 100)
    results = [utility_function(x, k, shift=0.0) for x in x_values]

    for i in range(len(results) - 1):
        assert results[i] >= results[i + 1], (
            f"Function should be monotonically decreasing, "
            f"but f({x_values[i]}) = {results[i]} < "
            f"f({x_values[i+1]}) = {results[i+1]}"
        )


def test_exponential_decay_shift(utility_function):
    """
    Test the shift parameter of the exponential decay function.

    Hypothesis:
    1. In the unshifted case, there should be values below the shift value.
    2. In the shifted case, all values should be:
       a) Higher than or equal to their corresponding unshifted values.
       b) Higher than or equal to the shift value.

    This test calculates the function over a range of x values to verify
    these properties.
    """
    k = 1.0
    shift_value = 0.2
    x_range = np.linspace(0, 5, 100)

    # Calculate reference values
    reference_values = [utility_function(x, k, shift=0.0) for x in x_range]

    # Verify that there are unshifted values below the shift value
    assert any(
        v < shift_value for v in reference_values
    ), "No unshifted values below shift value"

    # Calculate shifted values
    shifted_values = [utility_function(x, k, shift=shift_value) for x in x_range]

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
    ), "Not all shifted values are above or at shift value"


def test_exponential_decay_shift_at_zero(utility_function):
    """
    Test the effect of shift on the function value at x = 0.

    Hypothesis:
    At x = 0, the function value should always be 1.0 regardless of shift,
    because f(0) = 1.0 and shift is applied as: result * (1 - shift) +
    shift = 1.0 * (1 - shift) + shift = 1.0.

    This test verifies that f(0, shift=0) = f(0, shift=s) = 1.0 for s > 0.
    """
    k = 1.0
    reference = utility_function(0.0, k=k, shift=0.0)
    shifted = utility_function(0.0, k=k, shift=0.2)
    assert math.isclose(
        reference, 1.0, rel_tol=1e-9
    ), "Reference value at x=0 should be 1.0"
    assert math.isclose(
        shifted, 1.0, rel_tol=1e-9
    ), "Shifted value at x=0 should be 1.0"
    assert math.isclose(shifted, reference), "Shift should not affect value at x=0"


@pytest.mark.parametrize("k", [0.1, 1.0, 5.0, 10.0])
def test_exponential_decay_k_impact(utility_function, k):
    """
    Test the impact of the k parameter on the decay rate.

    Hypothesis:
    A higher k value should result in faster decay (steeper curve).

    This test verifies that for a fixed x > 0, f(x, k1) > f(x, k2) when
    k1 < k2.
    """
    x = 1.0
    result = utility_function(x, k, shift=0.0)
    expected = math.exp(-k * x)
    assert math.isclose(
        result, expected, rel_tol=1e-9
    ), f"Function value should be exp(-{k} * {x}) = {expected}"


def test_exponential_decay_k_comparison(utility_function):
    """
    Test that higher k values result in faster decay.

    Hypothesis:
    For x > 0, f(x, k=k1) > f(x, k=k2) when k1 < k2.

    This test verifies the relationship between k and decay rate.
    """
    x = 2.0
    k_small = 0.5
    k_large = 2.0

    result_small_k = utility_function(x, k_small, shift=0.0)
    result_large_k = utility_function(x, k_large, shift=0.0)

    assert result_small_k > result_large_k, (
        f"Smaller k should result in slower decay: "
        f"f({x}, k={k_small}) = {result_small_k} should be > "
        f"f({x}, k={k_large}) = {result_large_k}"
    )


# Edge case tests
def test_exponential_decay_very_large_x(utility_function):
    """
    Test the exponential decay function with very large x values.

    Hypothesis:
    As x approaches infinity, the function should approach 0.

    This test verifies that the function returns a value very close to 0
    for extremely large x values.
    """
    k = 1.0
    result = utility_function(1000.0, k, shift=0.0)
    assert math.isclose(
        result, 0.0, abs_tol=1e-9
    ), "Function should approach 0 for very large x values"


def test_exponential_decay_very_small_k(utility_function):
    """
    Test the exponential decay function with very small k values.

    Hypothesis:
    With very small k, the function should decay very slowly and remain
    close to 1.0 for moderate x values.

    This test verifies that the function behaves correctly with small
    decay rates.
    """
    x = 1.0
    k = 1e-6
    result = utility_function(x, k, shift=0.0)
    expected = math.exp(-k * x)
    assert math.isclose(
        result, expected, rel_tol=1e-9
    ), f"Function should be approximately {expected} for very small k"
    assert (
        result > 0.999
    ), "Function should remain close to 1.0 for small k and moderate x"


def test_exponential_decay_very_large_k(utility_function):
    """
    Test the exponential decay function with very large k values.

    Hypothesis:
    With very large k, the function should decay very rapidly and approach
    0 for even small x values.

    This test verifies that the function behaves correctly with large
    decay rates.
    """
    x = 0.1
    k = 100.0
    result = utility_function(x, k, shift=0.0)
    expected = math.exp(-k * x)
    assert math.isclose(
        result, expected, rel_tol=1e-9
    ), f"Function should be approximately {expected} for very large k"
    assert result < 0.001, "Function should approach 0 rapidly for large k"


def test_exponential_decay_negative_x_with_shift(utility_function):
    """
    Test the exponential decay function for negative x with shift applied.

    Hypothesis:
    For x < 0, the function is clamped to 1.0, so with shift, it should
    still return 1.0 (since 1.0 * (1 - shift) + shift = 1.0).

    This test verifies the interaction between clamping and shift.
    """
    x = -5.0
    shift = 0.3
    result = utility_function(x, k=1.0, shift=shift)
    assert math.isclose(result, 1.0, rel_tol=1e-9), (
        f"Function should be 1.0 for negative x regardless of shift, " f"got {result}"
    )
