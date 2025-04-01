import pytest

from pumas.desirability import desirability_catalogue
from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import ufloat


@pytest.fixture
def desirability_class():
    desirability_class = desirability_catalogue.get("sigmoid_bell")
    return desirability_class


def test_sigmoid_bell_results_compute_numeric(desirability_class):
    params = {
        "x1": 20.0,
        "x4": 80.0,
        "x2": 45.0,
        "x3": 60.0,
        "k": 1.0,
        "base": 10.0,
        "invert": False,
        "shift": 0.0,
    }
    desirability = desirability_class(params=params)
    result = desirability.compute_numeric(x=50.0)
    assert result == pytest.approx(expected=1.0)


def test_sigmoid_bell_results_compute_uscore(desirability_class):
    params = {
        "x1": 20.0,
        "x4": 80.0,
        "x2": 45.0,
        "x3": 60.0,
        "k": 1.0,
        "base": 10.0,
        "invert": False,
        "shift": 0.0,
    }
    desirability = desirability_class(params=params)
    result = desirability.compute_ufloat(x=ufloat(nominal_value=50.0, std_dev=0.0))
    assert result.nominal_value == pytest.approx(expected=1.0)
    assert result.std_dev == pytest.approx(expected=0.0)
