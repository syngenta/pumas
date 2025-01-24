import pytest

from pumas.architecture.parametrized_strategy import ParameterValueNotSet
from pumas.desirability import desirability_catalogue


def test_desirability_is_in_catalogue():
    assert "multistep" in desirability_catalogue.list_items()


def test_retrieved_desirability_is_not_a_global():
    """from the catalog we retrieve a class, and we
    instantiate it when used so
    each desirability function does not become a global"""

    # Retrieve the class, instantiate and set parameters
    desirability_class = desirability_catalogue.get("multistep")

    desirability_1 = desirability_class()
    desirability_1.set_coefficient_parameters_values(
        {"coordinates": [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]}
    )

    # Retrieve the class again
    desirability_class = desirability_catalogue.get("multistep")
    desirability_2 = desirability_class()
    # Retrieve the class, instantiate and look at parameters
    assert (
        desirability_1.coefficient_parameters_map
        != desirability_2.coefficient_parameters_map
    )


@pytest.fixture
def desirability():
    desirability_class = desirability_catalogue.get("multistep")
    desirability_instance = desirability_class()
    return desirability_instance


def test_multistep_parameters_defaults(desirability):
    assert desirability.get_coefficient_parameters_values() == {
        "coordinates": None,
        "shift": 0.0,
    }


def test_multistep_parameters_after_setting(desirability):
    desirability.set_coefficient_parameters_values(
        {"coordinates": [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)], "shift": 0.1}
    )
    assert desirability.get_coefficient_parameters_values() == {
        "coordinates": [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
        "shift": 0.1,
    }


def test_multistep_fails_without_parameters(desirability):
    with pytest.raises(ParameterValueNotSet):
        desirability.compute_score(x=0.5)
