import pytest

from pumas.desirability import desirability_catalogue
from pumas.error_propagation.uncertainties import ufloat


@pytest.mark.parametrize(
    ["name", "low", "high", "shift", "x", "expected_y"],
    [
        ("leftstep", 1.0, 2.0, 0.0, 0.5, 1.0),
        ("rightstep", 1.0, 2.0, 0.0, 2.5, 1.0),
        ("step", 1.0, 2.0, 0.0, 1.5, 1.0),
    ],
)
def test_step_results_compute_score(name, high, low, shift, x, expected_y):
    desirability_class = desirability_catalogue.get(name)
    desirability_instance = desirability_class()
    desirability = desirability_instance

    desirability.set_coefficient_parameters_values(
        {"low": low, "high": high, "shift": shift}
    )
    result = desirability.compute_score(x=x)
    assert result == pytest.approx(expected=expected_y)


# let this test fail
@pytest.mark.xfail
@pytest.mark.parametrize(
    ["name", "low", "high", "shift", "x", "expected_y"],
    [
        # Test for left_step
        ("leftstep", 1.0, 2.0, 0.0, 0.5, 1.0),
        # Test for right_step
        ("rightstep", 1.0, 2.0, 0.0, 2.5, 1.0),
        # Test for step (middle step)
        ("step", 1.0, 2.0, 0.0, 1.5, 1.0),  # low <= x <= high, with shift
    ],
)
def test_step_results_compute_uscore(name, high, low, shift, x, expected_y):
    desirability_class = desirability_catalogue.get(name)
    desirability_instance = desirability_class()
    desirability = desirability_instance

    desirability.set_coefficient_parameters_values(
        {"low": low, "high": high, "shift": shift}
    )
    result = desirability.compute_uscore(
        x=ufloat(nominal_value=expected_y, std_dev=0.0)
    )
    assert result == pytest.approx(expected=expected_y)
