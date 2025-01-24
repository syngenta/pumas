import pytest

from pumas.architecture.parametrized_strategy import ParameterValueNotSet
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
    desirability_class = desirability_catalogue.get(name)

    desirability_1 = desirability_class()
    desirability_1.set_coefficient_parameters_values(
        {"low": 1.0, "high": 2.0, "shift": 0.0}
    )

    # Retrieve the class again
    desirability_class = desirability_catalogue.get("step")
    desirability_2 = desirability_class()
    # Retrieve the class, instantiate and look at parameters
    assert (
        desirability_1.coefficient_parameters_map
        != desirability_2.coefficient_parameters_map
    )


@pytest.mark.parametrize("name", ["step", "leftstep", "rightstep"])
def test_multistep_parameters_after_setting(name):
    desirability_class = desirability_catalogue.get(name)
    desirability_instance = desirability_class()
    desirability = desirability_instance

    desirability.set_coefficient_parameters_values(
        {"low": 1.0, "high": 2.0, "shift": 0.1}
    )
    assert desirability.get_coefficient_parameters_values() == {
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
        desirability.compute_score(x=0.5)
