# type: ignore
import pytest

from pumas.desirability import desirability_catalogue
from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import ufloat


@pytest.fixture
def desirability():
    desirability_class = desirability_catalogue.get("bell")
    desirability_instance = desirability_class()
    return desirability_instance


def test_bell_results_compute_score(desirability):
    desirability.set_parameters_values(
        {"width": 1.0, "slope": 2.0, "center": 0.5, "invert": False, "shift": 0.0}
    )
    result = desirability.compute_numeric(x=0.5)
    assert result == pytest.approx(expected=1.0)


def test_bell_results_compute_uscore(desirability):
    desirability.set_parameters_values(
        {"width": 1.0, "slope": 2.0, "center": 5.0, "shift": 0.0}
    )
    result = desirability.compute_ufloat(x=ufloat(nominal_value=5.0, std_dev=0.0))
    assert result == pytest.approx(1.0)
