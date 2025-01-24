import pytest

from pumas.desirability import desirability_catalogue
from pumas.error_propagation.uncertainties import ufloat


@pytest.fixture
def desirability():
    desirability_class = desirability_catalogue.get("sigmoid")
    desirability_instance = desirability_class()
    return desirability_instance


def test_sigmoid_results_compute_score(desirability):
    desirability.set_coefficient_parameters_values(
        {"low": 0.0, "high": 1.0, "k": 1.0, "shift": 0.0, "base": 10.0}
    )
    result = desirability.compute_score(x=0.5)
    assert result == pytest.approx(expected=0.5)


# let this test fail
@pytest.mark.xfail
def test_sigmoid_results_compute_uscore(desirability):
    desirability.set_coefficient_parameters_values(
        {"low": 0.0, "high": 1.0, "k": 1.0, "shift": 0.0, "base": 10.0}
    )
    result = desirability.compute_uscore(x=ufloat(nominal_value=0.5, std_dev=0.0))
    assert result == pytest.approx(expected=0.5)
