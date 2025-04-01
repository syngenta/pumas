import pytest

from pumas.desirability import desirability_catalogue
from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import ufloat


@pytest.fixture
def desirability_class():
    desirability_class = desirability_catalogue.get("double_sigmoid")
    return desirability_class


def test_double_sigmoid_results_compute_score(desirability_class):
    params = {
        "low": 20.0,
        "high": 80.0,
        "coef_div": 5.0,
        "coef_si": 1.0,
        "coef_se": 1.0,
        "base": 10.0,
        "invert": False,
        "shift": 0.0,
    }
    desirability = desirability_class(params=params)
    result = desirability.compute_numeric(x=20.0)
    assert result == pytest.approx(expected=0.5)


def test_double_sigmoid_results_compute_uscore(desirability_class):
    params = {
        "low": 20.0,
        "high": 80.0,
        "coef_div": 5.0,
        "coef_si": 1.0,
        "coef_se": 1.0,
        "base": 10.0,
        "invert": False,
        "shift": 0.0,
    }
    desirability = desirability_class(params=params)
    result = desirability.compute_ufloat(x=ufloat(nominal_value=20.0, std_dev=0.0))
    assert result.nominal_value == pytest.approx(expected=0.5)
    assert result.std_dev == pytest.approx(expected=0.0)
    assert str(result) == "0.5+/-0"
