import pytest

from pumas.desirability import desirability_catalogue
from pumas.error_propagation.uncertainties import ufloat


@pytest.fixture
def desirability():
    desirability_class = desirability_catalogue.get("double_sigmoid")
    desirability_instance = desirability_class()
    return desirability_instance


def test_double_sigmoid_results_compute_score(desirability):
    desirability.set_coefficient_parameters_values(
        {
            "low": 20.0,
            "high": 80.0,
            "coef_div": 5.0,
            "coef_si": 1.0,
            "coef_se": 1.0,
            "base": 10.0,
            "invert": False,
            "shift": 0.0,
        }
    )
    result = desirability.compute_score(x=20.0)
    assert result == pytest.approx(expected=0.5)


# let this test fail
@pytest.mark.xfail
def test_double_sigmoid_results_compute_uscore(desirability):
    desirability.set_coefficient_parameters_values(
        {
            "low": 20.0,
            "high": 80.0,
            "coef_div": 5.0,
            "coef_si": 1.0,
            "coef_se": 1.0,
            "base": 10.0,
            "invert": False,
            "shift": 0.0,
        }
    )
    result = desirability.compute_uscore(x=ufloat(nominal_value=20.0, std_dev=0.0))
    assert result == pytest.approx(expected=20.0)
