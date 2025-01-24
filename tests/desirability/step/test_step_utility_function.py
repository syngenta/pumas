import pytest

from pumas.desirability import desirability_catalogue


@pytest.fixture
def desirability_leftstep():
    desirability_class = desirability_catalogue.get("leftstep")
    desirability_instance = desirability_class()
    return desirability_instance


@pytest.fixture
def desirability_rightstep():
    desirability_class = desirability_catalogue.get("rightstep")
    desirability_instance = desirability_class()
    return desirability_instance


@pytest.fixture
def desirability_step():
    desirability_class = desirability_catalogue.get("step")
    desirability_instance = desirability_class()
    return desirability_instance


@pytest.fixture
def utility_function_leftstep(desirability_leftstep):
    return desirability_leftstep.utility_function


@pytest.fixture
def utility_function_rightstep(desirability_rightstep):
    return desirability_rightstep.utility_function


@pytest.fixture
def utility_function_step(desirability_step):
    return desirability_step.utility_function


@pytest.mark.parametrize(
    "shift, error_type, error_msg",
    [
        (-0.1, ValueError, "Shift must be between 0 and 1"),
        (1.1, ValueError, "Shift must be between 0 and 1"),
    ],
)
def test_leftstep_wrong_shift(utility_function_leftstep, shift, error_type, error_msg):
    x = 50.0
    low = 0.0
    high = 100.0

    with pytest.raises(error_type, match=error_msg):
        utility_function_leftstep(x, low=low, high=high, shift=shift)


@pytest.mark.parametrize(
    "shift, error_type, error_msg",
    [
        (-0.1, ValueError, "Shift must be between 0 and 1"),
        (1.1, ValueError, "Shift must be between 0 and 1"),
    ],
)
def test_rightstep_wrong_shift(
    utility_function_rightstep, shift, error_type, error_msg
):
    x = 50.0
    low = 0.0
    high = 100.0

    with pytest.raises(error_type, match=error_msg):
        utility_function_rightstep(x, low=low, high=high, shift=shift)


@pytest.mark.parametrize(
    "shift, error_type, error_msg",
    [
        (-0.1, ValueError, "Shift must be between 0 and 1"),
        (1.1, ValueError, "Shift must be between 0 and 1"),
    ],
)
def test_step_wrong_shift(utility_function_step, shift, error_type, error_msg):
    x = 50.0
    low = 0.0
    high = 100.0

    with pytest.raises(error_type, match=error_msg):
        utility_function_step(x, low=low, high=high, shift=shift)


@pytest.mark.parametrize(
    "utility_function, x, low, high, expected",
    [
        ("leftstep", 0.5, 1.0, 2.0, 1.0),
        ("leftstep", 1.5, 1.0, 2.0, 0.0),
        ("rightstep", 2.5, 1.0, 2.0, 1.0),
        ("rightstep", 1.5, 1.0, 2.0, 0.0),
        ("step", 1.5, 1.0, 2.0, 1.0),
        ("step", 0.5, 1.0, 2.0, 0.0),
        ("step", 2.5, 1.0, 2.0, 0.0),
    ],
)
def test_step_functions_basic(
    desirability_leftstep,
    desirability_rightstep,
    desirability_step,
    utility_function,
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
        "leftstep": desirability_leftstep.utility_function,
        "rightstep": desirability_rightstep.utility_function,
        "step": desirability_step.utility_function,
    }
    assert utility_functions[utility_function](x, low=low, high=high) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "utility_function, param_to_change",
    [
        ("leftstep", "high"),
        ("rightstep", "low"),
    ],
)
def test_step_functions_unused_param(
    desirability_leftstep, desirability_rightstep, utility_function, param_to_change
):
    """
    Test that changing unused parameters doesn't affect the result.

    Hypothesis:
    - Changing 'high' for left step should not affect the result.
    - Changing 'low' for right step should not affect the result.
    """  # noqa: E501
    utility_functions = {
        "leftstep": desirability_leftstep.utility_function,
        "rightstep": desirability_rightstep.utility_function,
    }
    x, low, high = 1.5, 1.0, 2.0
    original_result = utility_functions[utility_function](x, low=low, high=high)

    new_params = {"low": low, "high": high, param_to_change: 10.0}

    new_result = utility_functions[utility_function](x, **new_params)
    assert original_result == new_result


@pytest.mark.parametrize(
    "utility_function, x_values",
    [
        ("leftstep", [0.5, 1.0, 1.5]),
        ("rightstep", [1.5, 2.0, 2.5]),
        ("step", [0.5, 1.5, 2.5]),
    ],
)
def test_step_functions_shift_impact(
    desirability_leftstep,
    desirability_rightstep,
    desirability_step,
    utility_function,
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
        "leftstep": desirability_leftstep.utility_function,
        "rightstep": desirability_rightstep.utility_function,
        "step": desirability_step.utility_function,
    }
    low, high = 1.0, 2.0
    shift_value = 0.2

    for x in x_values:
        unshifted = utility_functions[utility_function](
            x, low=low, high=high, shift=0.0
        )
        shifted = utility_functions[utility_function](
            x, low=low, high=high, shift=shift_value
        )

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
