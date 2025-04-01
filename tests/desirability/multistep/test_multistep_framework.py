import pytest

from pumas.desirability import desirability_catalogue
from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import ufloat


@pytest.fixture
def desirability_class():
    desirability_class = desirability_catalogue.get("multistep")
    return desirability_class


def test_multistep_results_compute_score(desirability_class):
    params = {"coordinates": [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)], "shift": 0.0}
    desirability = desirability_class(params=params)
    result = desirability.compute_numeric(x=0.25)
    assert result == pytest.approx(expected=0.25)


def test_multistep_results_compute_uscore(desirability_class):
    params = {"coordinates": [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)], "shift": 0.0}
    desirability = desirability_class(params=params)
    result = desirability.compute_ufloat(x=ufloat(nominal_value=0.25, std_dev=0.0))
    assert result.std_dev == pytest.approx(expected=0.0)
    assert str(result) == "0.25+/-0"
