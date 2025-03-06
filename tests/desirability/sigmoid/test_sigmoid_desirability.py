import pytest

from pumas.architecture.exceptions import InvalidBoundaryError, ParameterValueNotSet
from pumas.desirability import desirability_catalogue


def test_desirability_is_in_catalogue():
    assert "sigmoid" in desirability_catalogue.list_items()


def test_retrieved_desirability_is_not_a_global():
    """from the catalog we retrieve a class, and we
    instantiate it when used so
    each desirability function does not become a global"""

    # Retrieve the class, instantiate and set parameters
    desirability_class_1 = desirability_catalogue.get("sigmoid")
    desirability_class_2 = desirability_catalogue.get("sigmoid")

    desirability_1 = desirability_class_1()

    desirability_2 = desirability_class_2()

    assert id(desirability_1) != id(desirability_2)


@pytest.fixture
def desirability_class():
    desirability_class = desirability_catalogue.get("sigmoid")
    return desirability_class


def test_sigmoid_parameters_defaults(desirability_class):
    desirability = desirability_class()
    assert desirability.get_parameters_values() == {
        "base": 10.0,
        "low": None,
        "shift": 0.0,
        "k": 0.5,
        "high": None,
    }


def test_sigmoid_parameters_after_initialization(desirability_class):
    params = {"low": 0.0, "high": 1.0, "k": 1.0, "shift": 0.1, "base": 10.0}
    desirability = desirability_class(params=params)

    assert desirability.get_parameters_values() == {
        "base": 10.0,
        "low": 0.0,
        "shift": 0.1,
        "k": 1.0,
        "high": 1.0,
    }


def test_sigmoid_fails_without_parameters(desirability_class):
    desirability = desirability_class()
    with pytest.raises(ParameterValueNotSet):
        desirability.compute_numeric(x=0.5)


@pytest.mark.parametrize(
    "x, low, high, k, shift, base, error_type",
    [
        (0.5, 1.0, 0.0, 1.0, 0.0, 10.0, InvalidBoundaryError),  # low > high
        (0.5, 0.0, 1.0, -2.0, 0.0, 10, InvalidBoundaryError),  # k < -1
        (0.5, 0.0, 1.0, 2.0, 0.0, 10, InvalidBoundaryError),  # k > 1
        (0.5, 0.0, 1.0, 1.0, 0.0, 1.0, InvalidBoundaryError),  # base = 1
        (0.5, 0.0, 1.0, 1.0, 0.0, 0.0, InvalidBoundaryError),  # base < 1
        (0.5, 0.0, 1.0, 1.0, -0.1, 10.0, InvalidBoundaryError),  # shift < 0
        (0.5, 0.0, 1.0, 1.0, 1.1, 10.0, InvalidBoundaryError),  # shift > 1
    ],
)
def test_sigmoid_raises_error(
    desirability_class,
    x,
    low,
    high,
    k,
    shift,
    base,
    error_type,
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
    We also verify that the error type raised by the utility function is the same
    as  the error type raised by the desirability class.
    """  # noqa E501
    params = {"low": low, "high": high, "k": k, "shift": shift, "base": base}
    with pytest.raises(
        error_type,
    ):
        desirability = desirability_class(params=params)
        _ = desirability.compute_numeric(x=x)
