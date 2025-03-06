import pytest

from pumas.desirability import desirability_catalogue
from pumas.uncertainty.uncertainties_wrapper import ufloat


@pytest.fixture
def desirability_class():
    desirability_class = desirability_catalogue.get("sigmoid")
    return desirability_class


def test_sigmoid_results_compute_numeric(desirability_class):
    desirability = desirability_class(
        params={"low": 0.0, "high": 1.0, "k": 1.0, "shift": 0.0, "base": 10.0}
    )
    result = desirability.compute_numeric(x=0.5)
    assert result == pytest.approx(expected=0.5)

    result = desirability(x=0.5)
    assert result == pytest.approx(expected=0.5)


def test_sigmoid_results_compute_ufloat(desirability_class):
    desirability = desirability_class(
        params={"low": 0.0, "high": 1.0, "k": 1.0, "shift": 0.0, "base": 10.0}
    )
    result = desirability.compute_ufloat(x=ufloat(nominal_value=0.5, std_dev=0.0))
    assert result.nominal_value == pytest.approx(expected=0.5)
    assert result.std_dev == pytest.approx(expected=0.0)
    assert str(result) == "0.5+/-0"
