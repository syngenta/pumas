import pytest

from pumas.desirability import desirability_catalogue
from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import ufloat


@pytest.mark.parametrize(
    ["name", "low", "high", "shift", "x", "expected_y"],
    [
        ("leftstep", 1.0, 2.0, 0.0, 0.5, 1.0),
        ("rightstep", 1.0, 2.0, 0.0, 2.5, 1.0),
        ("step", 1.0, 2.0, 0.0, 2.5, 0.0),
        ("step", 1.0, 2.0, 0.0, 2.5, 0.0),
    ],
)
def test_step_results_compute_numeric(name, high, low, shift, x, expected_y):
    desirability_class = desirability_catalogue.get(name)
    desirability_instance = desirability_class()
    desirability = desirability_instance

    desirability.set_parameters_values({"low": low, "high": high, "shift": shift})
    result = desirability.compute_numeric(x=x)
    assert result == pytest.approx(expected=expected_y)


@pytest.mark.parametrize(
    ["name", "low", "high", "shift", "x", "expected_y"],
    [
        ("leftstep", 1.0, 2.0, 0.0, 0.5, 1.0),
        ("rightstep", 1.0, 2.0, 0.0, 2.5, 1.0),
        ("step", 1.0, 2.0, 0.0, 2.5, 0.0),
        ("step", 1.0, 2.0, 0.0, 2.5, 0.0),
    ],
)
def test_step_results_compute_ufloat(name, high, low, shift, x, expected_y):
    desirability_class = desirability_catalogue.get(name)
    desirability_instance = desirability_class()
    desirability = desirability_instance

    desirability.set_parameters_values({"low": low, "high": high, "shift": shift})
    result = desirability.compute_ufloat(x=ufloat(nominal_value=x, std_dev=0.0))
    assert result.nominal_value == pytest.approx(expected=expected_y)
