import pytest

from pumas.architecture.exceptions import InvalidBoundaryError, ParameterValueNotSet
from pumas.desirability import desirability_catalogue


@pytest.mark.parametrize("name", ["step", "leftstep", "rightstep"])
def test_desirability_is_in_catalogue(name):
    assert name in desirability_catalogue.list_items()


@pytest.mark.parametrize("name", ["step", "leftstep", "rightstep"])
def test_retrieved_desirability_is_not_a_global(name):
    """from the catalog we retrieve a class, and we
    instantiate it when used so
    each desirability function does not become a global"""

    # Retrieve the class, instantiate and set parameters
    desirability_class_1 = desirability_catalogue.get(name)
    desirability_class_2 = desirability_catalogue.get(name)

    desirability_1 = desirability_class_1()

    desirability_2 = desirability_class_2()

    assert id(desirability_1) != id(desirability_2)


@pytest.mark.parametrize("name", ["step", "leftstep", "rightstep"])
def test_step_parameters_after_initialization(name):
    desirability_class = desirability_catalogue.get(name)
    desirability_instance = desirability_class()
    desirability = desirability_instance

    desirability.set_parameters_values({"low": 1.0, "high": 2.0, "shift": 0.1})
    assert desirability.get_parameters_values() == {
        "low": 1.0,
        "high": 2.0,
        "shift": 0.1,
    }


@pytest.mark.parametrize("name", ["step", "leftstep", "rightstep"])
def test_multistep_fails_without_parameters(name):
    desirability_class = desirability_catalogue.get(name)
    desirability_instance = desirability_class()
    desirability = desirability_instance
    with pytest.raises(ParameterValueNotSet):
        _ = desirability.compute_numeric(x=0.5)


@pytest.mark.parametrize("name", ["step", "leftstep", "rightstep"])
@pytest.mark.parametrize(
    "shift, error_type",
    [
        (-0.1, InvalidBoundaryError),
        (1.1, InvalidBoundaryError),
    ],
)
def test_step_raises_error(name, shift, error_type):
    low = 0.0
    high = 100.0
    params = {"low": low, "high": high, "shift": shift}
    desirability_class = desirability_catalogue.get(name)
    with pytest.raises(error_type):
        _ = desirability_class(params=params)
