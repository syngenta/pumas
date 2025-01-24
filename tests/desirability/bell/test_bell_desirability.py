import pytest

from pumas.architecture.parametrized_strategy import ParameterValueNotSet
from pumas.desirability import desirability_catalogue


def test_desirability_is_in_catalogue():
    assert "bell" in desirability_catalogue.list_items()


def test_retrieved_desirability_is_not_a_global():
    desirability_class = desirability_catalogue.get("bell")

    desirability_1 = desirability_class()
    desirability_1.set_coefficient_parameters_values(
        {"width": 1.0, "slope": 2.0, "center": 0.5, "invert": False, "shift": 0.0}
    )

    desirability_class = desirability_catalogue.get("bell")
    desirability_2 = desirability_class()

    assert (
        desirability_1.coefficient_parameters_map
        != desirability_2.coefficient_parameters_map
    )


@pytest.fixture
def desirability():
    desirability_class = desirability_catalogue.get("bell")
    desirability_instance = desirability_class()
    return desirability_instance


def test_bell_parameters_defaults(desirability):
    assert desirability.get_coefficient_parameters_values() == {
        "width": None,
        "slope": 1.0,
        "center": None,
        "invert": False,
        "shift": 0.0,
    }


def test_bell_parameters_after_setting(desirability):
    desirability.set_coefficient_parameters_values(
        {"width": 1.0, "slope": 2.0, "center": 0.5, "invert": False, "shift": 0.1}
    )
    assert desirability.get_coefficient_parameters_values() == {
        "width": 1.0,
        "slope": 2.0,
        "center": 0.5,
        "invert": False,
        "shift": 0.1,
    }


def test_bell_fails_without_parameters(desirability):
    with pytest.raises(ParameterValueNotSet):
        desirability.compute_score(x=0.5)
