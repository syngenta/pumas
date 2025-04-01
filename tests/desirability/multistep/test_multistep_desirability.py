import pytest

from pumas.architecture.exceptions import (
    InvalidBoundaryError,
    InvalidInputTypeError,
    ParameterValueNotSet,
)
from pumas.desirability import desirability_catalogue


def test_desirability_is_in_catalogue():
    assert "multistep" in desirability_catalogue.list_items()


def test_retrieved_desirability_is_not_a_global():
    """from the catalog we retrieve a class, and we
    instantiate it when used so
    each desirability function does not become a global"""

    # Retrieve the class, instantiate and set parameters
    desirability_class_1 = desirability_catalogue.get("multistep")
    desirability_class_2 = desirability_catalogue.get("multistep")

    desirability_1 = desirability_class_1()

    desirability_2 = desirability_class_2()

    assert id(desirability_1) != id(desirability_2)


@pytest.fixture
def desirability_class():
    desirability_class = desirability_catalogue.get("multistep")
    return desirability_class


def test_multistep_parameters_defaults(desirability_class):
    desirability = desirability_class()
    assert desirability.get_parameters_values() == {
        "coordinates": None,
        "shift": 0.0,
    }


def test_multistep_parameters_after_setting(desirability_class):
    params = {"coordinates": [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)], "shift": 0.1}
    desirability = desirability_class(params=params)

    assert desirability.get_parameters_values() == {
        "coordinates": [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
        "shift": 0.1,
    }


def test_multistep_fails_without_parameters(desirability_class):
    desirability = desirability_class()
    with pytest.raises(ParameterValueNotSet):
        desirability.compute_numeric(x=0.5)


@pytest.mark.parametrize(
    "x, coordinates, shift, error_type",
    [
        (50.0, [(49.5, 0.0), (50.5, 1.0)], -1.0, InvalidBoundaryError),  # shift < 0
        (50.0, [(49.5, 0.0), (50.5, 1.0)], 2.0, InvalidBoundaryError),  # shift > 1
    ],
)
def test_multistep_wrong_shift(desirability_class, x, coordinates, shift, error_type):
    params = {"coordinates": coordinates, "shift": shift}
    with pytest.raises(error_type):
        desirability = desirability_class(params=params)
        _ = desirability.compute_numeric(x=x)


@pytest.mark.parametrize("x", [0.5, -0.5, 1])
def test_multistep_compute_numeric_input_type_success(desirability_class, x):
    params = {
        "coordinates": [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
        "shift": 0.1,
    }
    desirability = desirability_class(params=params)
    desirability.compute_numeric(x=x)


@pytest.mark.parametrize("x, error_type", [("0.5", InvalidInputTypeError)])
def test_multistep_compute_numeric_input_type_fail(desirability_class, x, error_type):
    params = {
        "coordinates": [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
        "shift": 0.1,
    }
    desirability = desirability_class(params=params)
    with pytest.raises(
        error_type,
    ):
        desirability.compute_numeric(x=x)
