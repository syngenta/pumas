import math

import numpy as np
import pytest

from pumas.desirability.multistep import multistep


@pytest.fixture
def desirability_utility_function():
    return multistep


@pytest.mark.parametrize(
    "coordinates, x, expected",
    [
        # Two coordinates, sorted
        [[(1.0, 0.3), (3.0, 0.7)], 0.5, 0.3],  # before first coordinate
        [[(2.0, 0.6), (4.0, 0.1)], 2.5, 0.475],  # interpolated value
        [[(1.5, 0.8), (3.5, 0.4)], 4.0, 0.4],  # after last coordinate
        # Two coordinates, unsorted
        [[(3.0, 0.7), (1.0, 0.3)], 0.5, 0.3],  # before first coordinate
        [[(4.0, 0.1), (2.0, 0.6)], 2.5, 0.475],  # interpolated value
        [[(3.5, 0.4), (1.5, 0.8)], 4.0, 0.4],  # after last coordinate
        # Three coordinates, sorted
        [[(1.0, 0.2), (3.0, 0.6), (5.0, 0.9)], 0.5, 0.2],  # before first coordinate
        [[(2.0, 0.4), (4.0, 0.3), (6.0, 0.7)], 2.5, 0.375],  # interpolated value
        [[(1.5, 0.8), (3.5, 0.1), (5.5, 0.5)], 4.0, 0.2],  # interpolated value
        [[(1.0, 0.3), (3.0, 0.7), (5.0, 0.9)], 6.0, 0.9],  # after last coordinate
        # Three coordinates, unsorted
        [[(3.0, 0.6), (5.0, 0.9), (1.0, 0.2)], 0.5, 0.2],  # before first coordinate
        [[(4.0, 0.3), (6.0, 0.7), (2.0, 0.4)], 2.5, 0.375],  # interpolated value
        [[(3.5, 0.1), (5.5, 0.5), (1.5, 0.8)], 4.0, 0.2],  # interpolated value
        [[(3.0, 0.7), (5.0, 0.9), (1.0, 0.3)], 6.0, 0.9],  # after last coordinate
        # Edge cases
        [[(1.0, 0.3), (3.0, 0.7)], 1.0, 0.3],  # x exactly at first coordinate
        [[(1.0, 0.3), (3.0, 0.7)], 3.0, 0.7],  # x exactly at last coordinate
        [
            [(1.0, 0.3), (2.0, 0.5), (3.0, 0.7)],
            2.0,
            0.5,
        ],  # x exactly at middle coordinate
    ],
)
def test_multi_step(desirability_utility_function, coordinates, x, expected):
    params = {"coordinates": coordinates, "shift": 0.0}
    result = desirability_utility_function(x=x, **params)
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-9)


@pytest.mark.parametrize(
    "x, coordinates, expected",
    [
        (2.0, [(1.0, 0.0), (3.0, 1.0)], 0.5),  # Basic interpolation
        (1.5, [(1.0, 0.0), (2.0, 1.0)], 0.5),  # Midpoint
        (1.0, [(1.0, 0.5), (2.0, 1.0)], 0.5),  # x equals first coordinate
        (2.0, [(1.0, 0.5), (2.0, 1.0)], 1.0),  # x equals second coordinate
        (1.25, [(1.0, 0.0), (2.0, 1.0)], 0.25),  # Quarter point
        (1.75, [(1.0, 0.0), (2.0, 1.0)], 0.75),  # Three-quarter point
    ],
)
def test_multistep_basic(desirability_utility_function, x, coordinates, expected):
    """
    Test basic multistep scenarios.

    This test covers various common interpolation cases, including
    midpoint, endpoints, and fractional points between two given coordinates.
    """  # noqa E501
    params = {"coordinates": coordinates, "shift": 0.0}
    result = desirability_utility_function(x=x, **params)
    assert math.isclose(result, expected, rel_tol=1e-9)


@pytest.mark.parametrize(
    "coordinates, error_msg",
    [
        (
            [(1.0, 0.5)],
            "At least two coordinates are required to form a valid multistep",
        ),  # Single point
    ],
)
def test_multistep_single_point(desirability_utility_function, coordinates, error_msg):
    """
    Test multistep behavior for a single point.

    Hypothesis:
    The function should raise a ValueError when given only one coordinate,
    as at least two points are required to define a multistep function.

    This test verifies that the function correctly identifies and rejects this invalid input.
    """  # noqa E501
    params = {"coordinates": coordinates, "shift": 0.0}
    with pytest.raises(ValueError, match=error_msg):
        desirability_utility_function(x=1.0, **params)


@pytest.mark.parametrize(
    "x, coordinates",
    [
        (1.5, [(1.0, 0.0), (2.0, float("inf"))]),
        (1.5, [(1.0, float("-inf")), (2.0, 1.0)]),
        (1.5, [(1.0, float("nan")), (2.0, 1.0)]),
    ],
)
def test_multistep_invalid_values(desirability_utility_function, x, coordinates):
    """
    Test multistep with invalid y-coordinate values.

    Hypothesis:
    The function should handle infinite or NaN y-coordinates gracefully,
    either by raising an appropriate exception or returning a special value.

    This test verifies the function's behavior with invalid input values.
    """  # noqa E501
    params = {"coordinates": coordinates, "shift": 0.0}
    with pytest.raises(ValueError):
        desirability_utility_function(x=x, **params)


@pytest.mark.parametrize(
    "x, coordinates, expected",
    [
        (1.5, [(1.0, 0.3), (2.0, 0.7)], 0.5),
        (0.5, [(0.0, 1.0), (1.0, 0.0)], 0.5),
        (2.5, [(0.0, 0.0), (5.0, 1.0)], 0.5),
    ],
)
def test_multistep_linear_relationship(
    desirability_utility_function, x, coordinates, expected
):
    """
    Test that multistep maintains a linear relationship between points.

    Hypothesis:
    The interpolated value should maintain a linear relationship between the two points,
    regardless of the absolute values of the coordinates.

    This test verifies that the function correctly implements linear interpolation.
    """  # noqa E501
    params = {"coordinates": coordinates, "shift": 0.0}
    result = desirability_utility_function(x=x, **params)
    assert math.isclose(result, expected, rel_tol=1e-9)


@pytest.mark.parametrize(
    "x, coordinates",
    [
        (1.5, [(2.0, 0.0), (1.0, 1.0)]),  # Decreasing x
        (1.5, [(-1.0, 0.0), (-2.0, 1.0)]),  # Negative x values
    ],
)
def test_multistep_reverse_order(desirability_utility_function, x, coordinates):
    """
    Test multistep with points in reverse order or with negative x-coordinates.

    Hypothesis:
    The function should work correctly regardless of the order of input points
    or the sign of x-coordinates.

    This test verifies that the function handles various orderings of input coordinates.
    """  # noqa E501
    params = {"coordinates": coordinates, "shift": 0.0}
    result = desirability_utility_function(x=x, **params)
    assert 0 <= result <= 1, f"Multistep result {result} is out of bounds"


@pytest.mark.parametrize(
    "x, coordinates, expected",
    [
        (0.5, [(1.0, 0.0), (2.0, 1.0)], 0.0),  # x below range
        (2.5, [(1.0, 0.0), (2.0, 1.0)], 1.0),  # x above range
        (1.0, [(1.0, 0.5), (2.0, 0.7), (3.0, 0.9)], 0.5),  # x at first point
        (3.0, [(1.0, 0.5), (2.0, 0.7), (3.0, 0.9)], 0.9),  # x at last point
        (2.0, [(1.0, 0.5), (2.0, 0.7), (3.0, 0.9)], 0.7),  # x at middle point
    ],
)
def test_multistep_edge_cases(desirability_utility_function, x, coordinates, expected):
    """
    Test multistep function for various edge cases.

    This test covers scenarios where:
    - x is below the range of given coordinates
    - x is above the range of given coordinates
    - x is exactly at one of the given coordinates
    - There are more than two coordinates

    Hypothesis:
    The function should handle these edge cases correctly, returning the appropriate
    y-value for out-of-range x values and exact matches, and interpolating correctly
    for in-between values.
    """  # noqa E501
    params = {"coordinates": coordinates, "shift": 0.0}
    result = desirability_utility_function(x=x, **params)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_multistep_many_coordinates(desirability_utility_function):
    """
    Test multistep function with a large number of coordinates.

    Hypothesis:
    The function should handle a large number of coordinates efficiently and accurately.

    This test verifies that the function can process a significant number of coordinates
    without loss of precision or performance issues.
    """  # noqa E501
    coordinates = [(i, i / 1000) for i in range(1000)]
    x = 500.5
    expected = 0.5005
    params = {"coordinates": coordinates, "shift": 0.0}
    result = desirability_utility_function(x=x, **params)
    assert math.isclose(result, expected, rel_tol=1e-9)


@pytest.mark.parametrize(
    "coordinates",
    [
        [(49.5, 0.0), (50.5, 1.0)],
        [(49.5, 1.0), (50.5, 0.0)],
        [(30.0, 0.0), (40.0, 0.8), (50.0, 1.0)],
        [(30.0, 1.0), (40.0, 0.8), (50.0, 0.0)],
        [(25.0, 0.0), (40.0, 1.0), (60.0, 1.0), (80.0, 0.0)],
        [(25.0, 0.0), (40.0, 0.8), (60.0, 0.8), (80.0, 0.0)],
        [(25.0, 1.0), (40.0, 0.0), (60.0, 0.0), (80.0, 1.0)],
    ],
)
def test_multistep_shift_impact(desirability_utility_function, coordinates):
    """
    Test the impact of the shift parameter on the multistep desirability function.

    Hypothesis:
    1. When shift is 0, the function should behave as normal.
    2. When shift is 0.2:
       a) All output values should be >= 0.2
       b) Values that were < 0.2 with no shift should now be exactly 0.2
       c) Values that were >= 0.2 with no shift should be scaled towards 1

    This test verifies that the shift parameter correctly adjusts the output range
    of the desirability function, effectively setting a minimum desirability value.
    """
    shift_value = 0.2
    x_range = np.linspace(
        min(c[0] for c in coordinates), max(c[0] for c in coordinates), 100
    )

    # Calculate reference values
    reference_values = [
        desirability_utility_function(x, coordinates, shift=0.0) for x in x_range
    ]

    # Verify that there are unshifted values below the shift value
    assert any(
        v < shift_value for v in reference_values
    ), "No unshifted values below shift value"

    # Calculate shifted values
    shifted_values = [
        desirability_utility_function(x, coordinates, shift=shift_value)
        for x in x_range
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
