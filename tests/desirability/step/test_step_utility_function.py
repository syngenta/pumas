import pytest

from pumas.desirability.step import (
    compute_numeric_left_step,
    compute_numeric_right_step,
    compute_numeric_step,
)


@pytest.fixture
def desirability_utility_function_leftstep():
    return compute_numeric_left_step


@pytest.fixture
def desirability_utility_function_rightstep():
    return compute_numeric_right_step


@pytest.fixture
def desirability_utility_function_step():
    return compute_numeric_step


@pytest.mark.parametrize(
    "name, x, low, high, expected",
    [
        ("step", 1.5, 1.0, 2.0, 1.0),
        ("step", 0.5, 1.0, 2.0, 0.0),
        ("step", 2.5, 1.0, 2.0, 0.0),
    ],
)
def test_central_step_functions_basic(
    desirability_utility_function_step,
    name,
    x,
    low,
    high,
    expected,
):
    """
    Test basic functionality of all step functions.

    Hypothesis:
    - Middle step should return 1 when low <= x <= high, 0 otherwise.
    """  # noqa: E501
    utility_functions = {
        "step": desirability_utility_function_step,
    }
    assert utility_functions[name](
        x, low=low, high=high, invert=False
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "name, x, low, high, expected",
    [
        ("leftstep", 0.5, 1.0, 2.0, 1.0),
        ("leftstep", 1.5, 1.0, 2.0, 0.0),
        ("rightstep", 2.5, 1.0, 2.0, 1.0),
        ("rightstep", 1.5, 1.0, 2.0, 0.0),
    ],
)
def test_r_l_step_functions_basic(
    desirability_utility_function_leftstep,
    desirability_utility_function_rightstep,
    name,
    x,
    low,
    high,
    expected,
):
    """
    Test basic functionality of all step functions.

    Hypothesis:
    - Left step should return 1 when x <= low, 0 otherwise.
    - Right step should return 1 when x >= high, 0 otherwise.
    - Middle step should return 1 when low <= x <= high, 0 otherwise.
    """  # noqa: E501
    utility_functions = {
        "leftstep": desirability_utility_function_leftstep,
        "rightstep": desirability_utility_function_rightstep,
    }
    assert utility_functions[name](x, low=low, high=high) == pytest.approx(expected)


@pytest.mark.parametrize(
    "name, param_to_change",
    [
        ("leftstep", "high"),
        ("rightstep", "low"),
    ],
)
def test_step_functions_unused_param(
    desirability_utility_function_leftstep,
    desirability_utility_function_rightstep,
    name,
    param_to_change,
):
    """
    Test that changing unused parameters doesn't affect the result.

    Hypothesis:
    - Changing 'high' for left step should not affect the result.
    - Changing 'low' for right step should not affect the result.
    """  # noqa: E501
    utility_functions = {
        "leftstep": desirability_utility_function_leftstep,
        "rightstep": desirability_utility_function_rightstep,
    }
    x, low, high = 1.5, 1.0, 2.0
    original_result = utility_functions[name](x, low=low, high=high)

    new_params = {"low": low, "high": high, param_to_change: 10.0}

    new_result = utility_functions[name](x, **new_params)
    assert original_result == new_result


@pytest.mark.parametrize(
    "name, x_values",
    [
        ("step", [0.5, 1.5, 2.5]),
    ],
)
def test_central_step_functions_shift_impact(
    desirability_utility_function_step,
    name,
    x_values,
):
    """
    Test the impact of the shift parameter on step functions.

    Hypothesis:
    1. When shift is 0, the function should behave normally.
    2. When shift is applied:
       a) All output values should be >= shift value
       b) Values that were 0 with no shift should now be exactly the shift value
       c) Values that were 1 with no shift should be scaled towards 1
    """  # noqa: E501
    utility_functions = {
        "step": desirability_utility_function_step,
    }
    low, high = 1.0, 2.0
    shift_value = 0.2

    for x in x_values:
        unshifted = utility_functions[name](
            x, low=low, high=high, invert=False, shift=0.0
        )
        shifted = utility_functions[name](
            x, low=low, high=high, invert=False, shift=shift_value
        )

        assert (
            shifted >= shift_value
        ), f"Shifted value {shifted} not >= shift value {shift_value}"
        assert (
            shifted >= unshifted
        ), f"Shifted value {shifted} not >= unshifted value {unshifted}"


@pytest.mark.parametrize(
    "name, x_values",
    [
        ("leftstep", [0.5, 1.0, 1.5]),
        ("rightstep", [1.5, 2.0, 2.5]),
    ],
)
def test_r_l_step_functions_shift_impact(
    desirability_utility_function_leftstep,
    desirability_utility_function_rightstep,
    desirability_utility_function_step,
    name,
    x_values,
):
    """
    Test the impact of the shift parameter on step functions.

    Hypothesis:
    1. When shift is 0, the function should behave normally.
    2. When shift is applied:
       a) All output values should be >= shift value
       b) Values that were 0 with no shift should now be exactly the shift value
       c) Values that were 1 with no shift should be scaled towards 1
    """  # noqa: E501
    utility_functions = {
        "leftstep": desirability_utility_function_leftstep,
        "rightstep": desirability_utility_function_rightstep,
    }
    low, high = 1.0, 2.0
    shift_value = 0.2

    for x in x_values:
        unshifted = utility_functions[name](x, low=low, high=high, shift=0.0)
        shifted = utility_functions[name](x, low=low, high=high, shift=shift_value)

        assert (
            shifted >= shift_value
        ), f"Shifted value {shifted} not >= shift value {shift_value}"
        assert (
            shifted >= unshifted
        ), f"Shifted value {shifted} not >= unshifted value {unshifted}"

        if unshifted == 0:
            assert shifted == pytest.approx(
                shift_value
            ), f"Expected {shift_value}, got {shifted}"
        elif unshifted == 1:
            assert shifted == pytest.approx(1.0), f"Expected 1.0, got {shifted}"
