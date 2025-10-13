# type: ignore
import sys

import pytest

from pumas.architecture.exceptions import InvalidBoundaryError, InvalidInputTypeError
from pumas.desirability import desirability_catalogue


def test_desirability_is_in_catalogue():
    assert "exponential_decay" in desirability_catalogue.list_items()


def test_retrieved_desirability_is_not_a_global():
    """from the catalog we retrieve a class, and we
    instantiate it when used so
    each desirability function does not become a global"""

    # Retrieve the class, instantiate and set parameters
    desirability_class_1 = desirability_catalogue.get("exponential_decay")
    desirability_class_2 = desirability_catalogue.get("exponential_decay")

    desirability_1 = desirability_class_1()

    desirability_2 = desirability_class_2()

    assert id(desirability_1) != id(desirability_2)


@pytest.fixture
def desirability_class():
    desirability_class = desirability_catalogue.get("exponential_decay")
    return desirability_class


def test_exponential_decay_parameters_defaults(desirability_class):
    desirability = desirability_class()
    assert desirability.get_parameters_values() == {
        "k": 1.0,
        "shift": 0.0,
    }


def test_exponential_decay_parameters_after_setting(desirability_class):
    params = {"k": 2.0, "shift": 0.1}
    desirability = desirability_class(params=params)

    assert desirability.get_parameters_values() == {
        "k": 2.0,
        "shift": 0.1,
    }


def test_exponential_decay_computes_with_default_parameters(
    desirability_class,
):
    """Test that the function can compute with default parameters (k=1.0)"""
    desirability = desirability_class()
    result = desirability.compute_numeric(x=0.0)
    assert result == pytest.approx(1.0)


def test_k_close_to_zero(desirability_class):
    """
    Test that the function raises a ValueError when k is too close to zero.

    Hypothesis:
    The function should raise a ValueError to prevent invalid decay rates
    when k is extremely small or zero.
    """
    params = {
        "k": sys.float_info.epsilon / 2.0,
        "shift": 0.0,
    }
    with pytest.raises(
        InvalidBoundaryError,
    ):
        _ = desirability_class(params=params)


@pytest.mark.parametrize(
    "x, k, shift, error_type",
    [
        (
            0.5,
            sys.float_info.epsilon / 2,
            0.0,
            InvalidBoundaryError,
        ),  # k too close to zero
        (
            1.0,
            -1.0,
            0.0,
            InvalidBoundaryError,
        ),  # k < 0
        (
            1.0,
            0.0,
            0.0,
            InvalidBoundaryError,
        ),  # k = 0
        (
            1.0,
            1.0,
            -0.5,
            InvalidBoundaryError,
        ),  # shift < 0
        (
            1.0,
            1.0,
            2.0,
            InvalidBoundaryError,
        ),  # shift > 1
    ],
)
def test_utility_function_raises_error(desirability_class, x, k, shift, error_type):
    """
    Test that the function raises appropriate errors for invalid inputs.

    Hypothesis:
    The function should raise specific errors with appropriate error
    messages for various invalid input combinations.
    """
    params = {"k": k, "shift": shift}
    with pytest.raises(
        error_type,
    ):
        desirability = desirability_class(params=params)
        _ = desirability.compute_numeric(x=x)


@pytest.mark.parametrize("x", [0.5, -0.5, 1, 0])
def test_exponential_decay_compute_numeric_input_type_success(desirability_class, x):
    params = {"k": 1.0, "shift": 0.1}
    desirability = desirability_class(params=params)
    desirability.compute_numeric(x=x)


@pytest.mark.parametrize("x, error_type", [("0.5", InvalidInputTypeError)])
def test_exponential_decay_compute_numeric_input_type_fail(
    desirability_class, x, error_type
):
    params = {"k": 1.0, "shift": 0.1}
    desirability = desirability_class(params=params)
    with pytest.raises(
        error_type,
    ):
        desirability.compute_numeric(x=x)


@pytest.mark.parametrize(
    "x, k, expected",
    [
        (0.0, 1.0, 1.0),  # At x=0, exp(-k*0) = 1
        (1.0, 1.0, 0.36787944117144233),  # exp(-1) ≈ 0.368
        (-1.0, 1.0, 1.0),  # Clamped to 1.0 for x < 0
        (-5.0, 2.0, 1.0),  # Clamped to 1.0 for x < 0
        (2.0, 0.5, 0.36787944117144233),  # exp(-0.5*2) = exp(-1) ≈ 0.368
    ],
)
def test_exponential_decay_values(desirability_class, x, k, expected):
    """
    Test that the exponential decay function returns expected values.

    Hypothesis:
    The function should return exp(-k*x) for x >= 0 and 1.0 for x < 0.
    """
    params = {"k": k, "shift": 0.0}
    desirability = desirability_class(params=params)
    result = desirability.compute_numeric(x=x)
    assert result == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "x, k, shift, expected",
    [
        (0.0, 1.0, 0.2, 1.0),  # 1.0 * (1 - 0.2) + 0.2 = 1.0
        (
            1.0,
            1.0,
            0.2,
            0.49430355293715387,
        ),  # exp(-1) * 0.8 + 0.2 ≈ 0.494
        (
            1.0,
            1.0,
            0.5,
            0.6839397205857117,
        ),  # exp(-1) * 0.5 + 0.5 ≈ 0.684
    ],
)
def test_exponential_decay_shift(desirability_class, x, k, shift, expected):
    """
    Test that the shift parameter correctly shifts the function vertically.
    """
    params = {"k": k, "shift": shift}
    desirability = desirability_class(params=params)
    result = desirability.compute_numeric(x=x)
    assert result == pytest.approx(expected, rel=1e-6)
