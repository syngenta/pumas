import pytest

from pumas.architecture.parametrized_strategy import ParameterValueNotSet
from pumas.desirability import desirability_catalogue


def test_desirability_is_in_catalogue():
    assert "double_sigmoid" in desirability_catalogue.list_items()


def test_retrieved_desirability_is_not_a_global():
    """from the catalog we retrieve a class, and we
    instantiate it when used so
    each desirability function does not become a global"""
    desirability_class = desirability_catalogue.get("double_sigmoid")

    desirability_1 = desirability_class()
    desirability_1.set_coefficient_parameters_values(
        {
            "low": 3.0,
            "high": 7.0,
            "coef_div": 1.0,
            "coef_si": 2.0,
            "coef_se": 2.0,
            "shift": 0.0,
            "invert": False,
            "base": 10.0,
        }
    )

    desirability_class = desirability_catalogue.get("double_sigmoid")
    desirability_2 = desirability_class()

    assert (
        desirability_1.coefficient_parameters_map
        != desirability_2.coefficient_parameters_map
    )


@pytest.fixture
def desirability():
    desirability_class = desirability_catalogue.get("double_sigmoid")
    desirability_instance = desirability_class()
    return desirability_instance


def test_double_sigmoid_parameters_defaults(desirability):
    assert desirability.get_coefficient_parameters_values() == {
        "low": None,
        "high": None,
        "coef_div": 1.0,
        "coef_si": 1.0,
        "coef_se": 1.0,
        "base": 10.0,
        "invert": False,
        "shift": 0.0,
    }


def test_double_sigmoid_parameters_after_setting(desirability):
    desirability.set_coefficient_parameters_values(
        {
            "low": 3.0,
            "high": 7.0,
            "coef_div": 1.0,
            "coef_si": 2.0,
            "coef_se": 2.0,
            "shift": 0.1,
            "invert": True,
            "base": 10.0,
        }
    )
    assert desirability.get_coefficient_parameters_values() == {
        "low": 3.0,
        "high": 7.0,
        "coef_div": 1.0,
        "coef_si": 2.0,
        "coef_se": 2.0,
        "base": 10.0,
        "invert": True,
        "shift": 0.1,
    }


def test_double_sigmoid_fails_without_parameters(desirability):
    with pytest.raises(ParameterValueNotSet):
        desirability.compute_score(x=5.0)
