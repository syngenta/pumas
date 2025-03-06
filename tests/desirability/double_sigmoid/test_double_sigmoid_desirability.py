import pytest

from pumas.architecture.exceptions import InvalidBoundaryError, ParameterValueNotSet
from pumas.desirability import desirability_catalogue


def test_desirability_is_in_catalogue():
    assert "double_sigmoid" in desirability_catalogue.list_items()


def test_retrieved_desirability_is_not_a_global():
    """from the catalog we retrieve a class, and we
    instantiate it when used so
    each desirability function does not become a global"""
    desirability_class_1 = desirability_catalogue.get("double_sigmoid")
    desirability_class_2 = desirability_catalogue.get("double_sigmoid")

    desirability_1 = desirability_class_1()
    desirability_2 = desirability_class_2()

    # Retrieve the class, instantiate and look at parameters
    desirability_2.set_parameters_values(
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

    assert desirability_1.get_parameters_values != desirability_2.get_parameters_values

    assert id(desirability_1) != id(desirability_2)


@pytest.fixture
def desirability_class():
    desirability_class = desirability_catalogue.get("double_sigmoid")
    return desirability_class


def test_double_sigmoid_parameters_defaults(desirability_class):

    assert desirability_class()


def test_double_sigmoid_parameters_after_setting(desirability_class):
    params = {
        "low": 3.0,
        "high": 7.0,
        "coef_div": 1.0,
        "coef_si": 2.0,
        "coef_se": 2.0,
        "shift": 0.1,
        "invert": True,
        "base": 10.0,
    }
    desirability = desirability_class(params=params)

    assert desirability.get_parameters_values() == {
        "low": 3.0,
        "high": 7.0,
        "coef_div": 1.0,
        "coef_si": 2.0,
        "coef_se": 2.0,
        "base": 10.0,
        "invert": True,
        "shift": 0.1,
    }


def test_double_sigmoid_fails_without_parameters(desirability_class):
    desirability = desirability_class()
    assert any(v is None for v in desirability.get_parameters_values().values())
    with pytest.raises(ParameterValueNotSet):
        desirability.compute_numeric(x=5.0)


@pytest.mark.parametrize(
    "low, high, coef_div, coef_si, coef_se, base, shift",
    [
        (0.0, 10.0, 1.0, -1.0, 1.0, 10.0, 0.0),
        (0.0, 10.0, 1.0, 1.0, -1.0, 10.0, 0.0),
        (0.0, 10.0, 1.0, 1.0, 1.0, 0.5, 0.0),
        (0.0, 10.0, 1.0, 1.0, 1.0, 10.0, -0.1),
        (0.0, 10.0, 1.0, 1.0, 1.0, 10.0, 1.1),
    ],
)
def test_double_sigmoid_raises_error(
    desirability_class,
    low,
    high,
    coef_div,
    coef_si,
    coef_se,
    base,
    shift,
):
    """
    Test that the double sigmoid function raises appropriate errors for invalid inputs.

    Hypothesis:
    The function should raise ValueError with specific error messages for various invalid input combinations.

    This test verifies that the function raises the expected errors with the correct error messages for each invalid input case.
    """  # noqa E501
    params = {
        "low": low,
        "high": high,
        "coef_div": coef_div,
        "coef_si": coef_si,
        "coef_se": coef_se,
        "base": base,
        "shift": shift,
    }

    with pytest.raises(
        InvalidBoundaryError,
    ):
        _ = desirability_class(params=params)
