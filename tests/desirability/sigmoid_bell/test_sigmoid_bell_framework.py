import pytest

from pumas.desirability import desirability_catalogue
from pumas.error_propagation.uncertainties import ufloat


@pytest.fixture
def desirability():
    desirability_class = desirability_catalogue.get("sigmoid_bell")
    desirability_instance = desirability_class()
    return desirability_instance


def test_sigmoid_bell_results_compute_score(desirability):
    desirability.set_coefficient_parameters_values(
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
    result = desirability.compute_score(x=50.0)
    assert result == pytest.approx(expected=1.0)


# let this test fail
@pytest.mark.xfail
def test_sigmoid_bell_results_compute_uscore(desirability):
    desirability.set_coefficient_parameters_values(
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
    result = desirability.compute_uscore(x=ufloat(nominal_value=1.0, std_dev=0.0))
    assert result == pytest.approx(expected=20.0)
