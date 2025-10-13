import pytest

from pumas.desirability import desirability_catalogue
from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import ufloat


@pytest.fixture
def desirability_class():
    desirability_class = desirability_catalogue.get("exponential_decay")
    return desirability_class


def test_exponential_decay_results_compute_score(desirability_class):
    params = {
        "k": 1.0,
        "shift": 0.0,
    }
    desirability = desirability_class(params=params)
    result = desirability.compute_numeric(x=0.0)
    assert result == pytest.approx(expected=1.0)


def test_exponential_decay_results_compute_score_at_one(desirability_class):
    params = {
        "k": 1.0,
        "shift": 0.0,
    }
    desirability = desirability_class(params=params)
    result = desirability.compute_numeric(x=1.0)
    assert result == pytest.approx(expected=0.36787944117144233)  # exp(-1)


def test_exponential_decay_results_compute_score_clamped(desirability_class):
    params = {
        "k": 1.0,
        "shift": 0.0,
    }
    desirability = desirability_class(params=params)
    result = desirability.compute_numeric(x=-5.0)
    assert result == pytest.approx(expected=1.0)  # Clamped for x < 0


def test_exponential_decay_results_compute_uscore(desirability_class):
    params = {
        "k": 1.0,
        "shift": 0.0,
    }
    desirability = desirability_class(params=params)
    result = desirability.compute_ufloat(x=ufloat(nominal_value=0.0, std_dev=0.0))
    assert result.nominal_value == pytest.approx(expected=1.0)
    assert result.std_dev == pytest.approx(expected=0.0)
    assert str(result) == "1.0+/-0"


def test_exponential_decay_results_compute_uscore_at_one(desirability_class):
    params = {
        "k": 1.0,
        "shift": 0.0,
    }
    desirability = desirability_class(params=params)
    result = desirability.compute_ufloat(x=ufloat(nominal_value=1.0, std_dev=0.0))
    assert result.nominal_value == pytest.approx(expected=0.36787944117144233)
    assert result.std_dev == pytest.approx(expected=0.0)


def test_exponential_decay_results_compute_uscore_with_uncertainty(desirability_class):
    params = {
        "k": 1.0,
        "shift": 0.0,
    }
    desirability = desirability_class(params=params)
    result = desirability.compute_ufloat(x=ufloat(nominal_value=1.0, std_dev=0.1))
    # exp(-1) ≈ 0.368, with uncertainty propagation
    assert result.nominal_value == pytest.approx(expected=0.36787944117144233, rel=1e-6)
    # Uncertainty should be propagated: d/dx[exp(-kx)] = -k*exp(-kx)
    # std_dev ≈ |derivative| * input_std_dev = |-1 * exp(-1)| * 0.1 ≈ 0.0368
    assert result.std_dev == pytest.approx(expected=0.036787944117144235, rel=1e-2)


def test_exponential_decay_results_compute_uscore_clamped(desirability_class):
    params = {
        "k": 1.0,
        "shift": 0.0,
    }
    desirability = desirability_class(params=params)
    result = desirability.compute_ufloat(x=ufloat(nominal_value=-5.0, std_dev=0.5))
    # For x < 0, clamped to 1.0 (returns plain float, not UFloat)
    assert result == pytest.approx(1.0)
    assert isinstance(result, float)


def test_exponential_decay_with_shift_compute_score(desirability_class):
    params = {
        "k": 1.0,
        "shift": 0.2,
    }
    desirability = desirability_class(params=params)
    result = desirability.compute_numeric(x=1.0)
    # exp(-1) * (1 - 0.2) + 0.2 ≈ 0.494
    assert result == pytest.approx(expected=0.49430355293715387)
