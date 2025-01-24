import pytest

from pumas.architecture.parametrized_strategy import ParameterValueNotSet
from pumas.desirability import desirability_catalogue


def test_desirability_is_in_catalogue():
    assert "sigmoid_bell" in desirability_catalogue.list_items()


def test_retrieved_desirability_is_not_a_global():
    """from the catalog we retrieve a class, and we
    instantiate it when used so
    each desirability function does not become a global"""

    # Retrieve the class, instantiate and set parameters
    desirability_class = desirability_catalogue.get("sigmoid_bell")

    desirability_1 = desirability_class()
    desirability_1.set_coefficient_parameters_values(
        {
            "x1": 20.0,
            "x4": 80.0,
            "x2": 45.0,
            "x3": 60.0,
            "k": 1.0,
            "base": 10.0,
            "invert": False,
            "shift": 0.0,
        }
    )

    # Retrieve the class again
    desirability_class = desirability_catalogue.get("sigmoid_bell")
    desirability_2 = desirability_class()
    # Retrieve the class, instantiate and look at parameters
    assert (
        desirability_1.coefficient_parameters_map
        != desirability_2.coefficient_parameters_map
    )


@pytest.fixture
def desirability():
    desirability_class = desirability_catalogue.get("sigmoid_bell")
    desirability_instance = desirability_class()
    return desirability_instance


def test_sigmoid_bell_parameters_defaults(desirability):
    assert desirability.get_coefficient_parameters_values() == {
        "x1": None,
        "x4": None,
        "x2": None,
        "x3": None,
        "k": 1.0,
        "base": 10.0,
        "invert": False,
        "shift": 0.0,
    }


def test_sigmoid_bell_parameters_after_setting(desirability):
    desirability.set_coefficient_parameters_values(
        {
            "x1": 20.0,
            "x4": 80.0,
            "x2": 45.0,
            "x3": 60.0,
            "k": 1.0,
            "base": 10.0,
            "invert": True,
            "shift": 0.0,
        }
    )
    assert desirability.get_coefficient_parameters_values() == {
        "x1": 20.0,
        "x4": 80.0,
        "x2": 45.0,
        "x3": 60.0,
        "k": 1.0,
        "base": 10.0,
        "invert": True,
        "shift": 0.0,
    }


def test_sigmoid_bell_fails_without_parameters(desirability):
    with pytest.raises(ParameterValueNotSet):
        desirability.compute_score(x=0.5)
