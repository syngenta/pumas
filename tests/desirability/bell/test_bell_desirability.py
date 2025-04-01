# type: ignore
import sys

import pytest

from pumas.architecture.exceptions import (
    InvalidBoundaryError,
    InvalidInputTypeError,
    ParameterValueNotSet,
)
from pumas.desirability import desirability_catalogue


def test_desirability_is_in_catalogue():
    assert "bell" in desirability_catalogue.list_items()


def test_retrieved_desirability_is_not_a_global():
    """from the catalog we retrieve a class, and we
    instantiate it when used so
    each desirability function does not become a global"""

    # Retrieve the class, instantiate and set parameters
    desirability_class_1 = desirability_catalogue.get("bell")
    desirability_class_2 = desirability_catalogue.get("bell")

    desirability_1 = desirability_class_1()

    desirability_2 = desirability_class_2()

    assert id(desirability_1) != id(desirability_2)


@pytest.fixture
def desirability_class():
    desirability_class = desirability_catalogue.get("bell")
    return desirability_class


def test_bell_parameters_defaults(desirability_class):
    desirability = desirability_class()
    assert desirability.get_parameters_values() == {
        "width": None,
        "slope": 1.0,
        "center": None,
        "invert": False,
        "shift": 0.0,
    }


def test_bell_parameters_after_setting(desirability_class):
    params = {"width": 1.0, "slope": 2.0, "center": 0.5, "invert": False, "shift": 0.1}
    desirability = desirability_class(params=params)

    assert desirability.get_parameters_values() == {
        "width": 1.0,
        "slope": 2.0,
        "center": 0.5,
        "invert": False,
        "shift": 0.1,
    }


def test_bell_fails_without_parameters(desirability_class):
    desirability = desirability_class()
    with pytest.raises(ParameterValueNotSet):
        desirability.compute_numeric(x=0.5)


def test_width_close_to_zero(desirability_class):
    """
    Test that the function raises a ValueError when width is too close to zero.

    Hypothesis:
    The function should raise a ValueError to prevent numerical instability when width is extremely small.
    """  # noqa E501
    params = {
        "center": 0.5,
        "width": sys.float_info.epsilon / 2.0,
        "slope": 1.0,
        "invert": True,
        "shift": 0.0,
    }
    with pytest.raises(
        InvalidBoundaryError,
    ):
        _ = desirability_class(params=params)


@pytest.mark.parametrize(
    "x, center, width, slope, shift, error_type",
    [
        (
            0.5,
            0.5,
            sys.float_info.epsilon / 2,
            1.0,
            0.0,
            InvalidBoundaryError,
        ),  # width too close to zero
        (
            1.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            InvalidBoundaryError,
        ),  # width < 0
        (
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            InvalidBoundaryError,
        ),  # width = 0
        (
            1.0,
            0.0,
            1.0,
            -1.0,
            0.0,
            InvalidBoundaryError,
        ),  # slope < 0
        (
            1.0,
            0.0,
            -2.0,
            1.0,
            2.0,
            InvalidBoundaryError,
        ),  # shift < -1
        (
            1.0,
            0.0,
            2.0,
            1.0,
            2.0,
            InvalidBoundaryError,
        ),  # shift > 1
    ],
)
def test_utility_function_raises_error(
    desirability_class, x, center, width, slope, shift, error_type
):
    """
    Test that the function raises appropriate errors for invalid inputs.

    Hypothesis:
    The function should raise specific errors with appropriate error messages for various invalid input combinations.
    """  # noqa E501
    params = {"center": center, "width": width, "slope": slope, "shift": shift}
    with pytest.raises(
        error_type,
    ):
        desirability = desirability_class(params=params)
        _ = desirability.compute_numeric(x=x)


@pytest.mark.parametrize("x", [0.5, -0.5, 1])
def test_bell_compute_numeric_input_type_success(desirability_class, x):
    params = {"width": 1.0, "slope": 2.0, "center": 0.5, "invert": False, "shift": 0.1}
    desirability = desirability_class(params=params)
    desirability.compute_numeric(x=x)


@pytest.mark.parametrize("x, error_type", [("0.5", InvalidInputTypeError)])
def test_bell_compute_numeric_input_type_fail(desirability_class, x, error_type):
    params = {"width": 1.0, "slope": 2.0, "center": 0.5, "invert": False, "shift": 0.1}
    desirability = desirability_class(params=params)
    with pytest.raises(
        error_type,
    ):
        desirability.compute_numeric(x=x)
