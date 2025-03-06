import pytest

from pumas.architecture.exceptions import InvalidBoundaryError, ParameterValueNotSet
from pumas.desirability import desirability_catalogue


def test_desirability_is_in_catalogue():
    assert "sigmoid_bell" in desirability_catalogue.list_items()


def test_retrieved_desirability_is_not_a_global():
    """from the catalog we retrieve a class, and we
    instantiate it when used so
    each desirability function does not become a global"""

    # Retrieve the class, instantiate and set parameters
    desirability_class_1 = desirability_catalogue.get("sigmoid_bell")
    desirability_class_2 = desirability_catalogue.get("sigmoid_bell")

    desirability_1 = desirability_class_1()

    desirability_2 = desirability_class_2()

    assert id(desirability_1) != id(desirability_2)


@pytest.fixture
def desirability_class():
    desirability_class = desirability_catalogue.get("sigmoid_bell")
    return desirability_class


def test_sigmoid_bell_parameters_defaults(desirability_class):
    desirability = desirability_class()
    assert desirability.get_parameters_values() == {
        "x1": None,
        "x4": None,
        "x2": None,
        "x3": None,
        "k": 1.0,
        "base": 10.0,
        "invert": False,
        "shift": 0.0,
    }


def test_sigmoid_bell_parameters_after_initialization(desirability_class):
    params = {
        "x1": 20.0,
        "x4": 80.0,
        "x2": 45.0,
        "x3": 60.0,
        "k": 1.0,
        "base": 10.0,
        "invert": True,
        "shift": 0.0,
    }
    desirability = desirability_class(params=params)

    assert desirability.get_parameters_values() == {
        "x1": 20.0,
        "x4": 80.0,
        "x2": 45.0,
        "x3": 60.0,
        "k": 1.0,
        "base": 10.0,
        "invert": True,
        "shift": 0.0,
    }


def test_sigmoid_bell_fails_without_parameters(desirability_class):
    desirability = desirability_class()
    with pytest.raises(ParameterValueNotSet):
        desirability.compute_numeric(x=0.5)


@pytest.mark.parametrize(
    "x, x1, x2, x3, x4, k, base, invert, shift, error_type",
    [
        # (50.0, 20.0, 45.0, 60.0, 80.0, 1.0, 10., False, 0.0, error_type),  # reference
        (
            50.0,
            20.0,
            15.0,
            60.0,
            80.0,
            1.0,
            10.0,
            False,
            0.0,
            InvalidBoundaryError,
        ),  # x2 < x1
        (
            50.0,
            20.0,
            45.0,
            90.0,
            80.0,
            1.0,
            10.0,
            False,
            0.0,
            InvalidBoundaryError,
        ),  # x3 > x4
        (
            50.0,
            20.0,
            45.0,
            60.0,
            80.0,
            0.5,
            10.0,
            False,
            0.0,
            InvalidBoundaryError,
        ),  # k < 1
        (
            50.0,
            20.0,
            45.0,
            60.0,
            80.0,
            1.0,
            1.0,
            False,
            0.0,
            InvalidBoundaryError,
        ),  # base < 1
        (
            50.0,
            20.0,
            45.0,
            60.0,
            80.0,
            1.0,
            10.0,
            False,
            -0.1,
            InvalidBoundaryError,
        ),  # shift < 0
        (
            50.0,
            20.0,
            45.0,
            60.0,
            80.0,
            1.0,
            10.0,
            False,
            2.0,
            InvalidBoundaryError,
        ),  # shift > 1
    ],
)
def test_sigmoid_bell_invalid_parameters(
    desirability_class, x, x1, x2, x3, x4, k, base, invert, shift, error_type
):
    """
    Test that the function raises ValueError for invalid parameter combinations.

    Hypothesis:
    The function should raise ValueError when x1, x2, x3, x4 are
    not in ascending order or when base <= 1.
    """
    params = {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "k": k,
        "base": base,
        "invert": invert,
        "shift": shift,
    }
    with pytest.raises(
        error_type,
    ):
        desirability = desirability_class(params=params)
        _ = desirability.compute_numeric(x=x)
