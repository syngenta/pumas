import pytest

from pumas.architecture.exceptions import InvalidBoundaryError, ParameterValueNotSet
from pumas.desirability import desirability_catalogue


def test_desirability_is_in_catalogue():
    assert "value_mapping" in desirability_catalogue.list_items()


def test_retrieved_desirability_is_not_a_global():
    """from the catalog we retrieve a class, and we
    instantiate it when used so
    each desirability function does not become a global"""

    # Retrieve the class, instantiate and set parameters
    desirability_class_1 = desirability_catalogue.get("value_mapping")
    desirability_class_2 = desirability_catalogue.get("value_mapping")

    desirability_1 = desirability_class_1()

    desirability_2 = desirability_class_2()

    assert id(desirability_1) != id(desirability_2)


@pytest.fixture
def desirability_class():
    desirability_class = desirability_catalogue.get("value_mapping")
    return desirability_class


def test_value_mapping_parameters_defaults(desirability_class):
    desirability = desirability_class()
    assert desirability.get_parameters_values() == {
        "mapping": None,
        "shift": 0.0,
    }


def test_value_mapping_parameters_after_setting(desirability_class):
    params = {"mapping": {"Low": 0.2, "Medium": 0.5, "High": 0.8}, "shift": 0.1}
    desirability = desirability_class(params=params)

    assert desirability.get_parameters_values() == {
        "mapping": {"Low": 0.2, "Medium": 0.5, "High": 0.8},
        "shift": 0.1,
    }


def test_value_mapping_fails_without_parameters(desirability_class):
    desirability = desirability_class()
    with pytest.raises(ParameterValueNotSet):
        desirability.compute_string(x="Low")


@pytest.mark.parametrize(
    "shift, error_type",
    [
        (-0.1, InvalidBoundaryError),
        (1.1, InvalidBoundaryError),
    ],
)
def test_value_mapping_raises_error(desirability_class, shift, error_type):
    x = "Low"
    mapping = {"Low": 0.2, "Medium": 0.5, "High": 0.8}
    params = {"mapping": mapping, "shift": shift}
    with pytest.raises(
        error_type,
    ):
        desirability = desirability_class(params=params)
        _ = desirability.compute_numeric(x=x)
