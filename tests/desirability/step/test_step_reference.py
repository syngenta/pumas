import numpy as np
import pytest

from pumas.desirability import desirability_catalogue
from tests.desirability.external_reference_implementation.steps import (
    LeftStep,
    Parameters,
    RightStep,
    Step,
)


@pytest.mark.parametrize("x", [float(x) for x in range(-20, 20, 5)])
@pytest.mark.parametrize("low, high", [(1.0, 2.0), (0.0, 1.0), (-1.0, 0.0)])
def test_leftstep_equivalence_to_func_reference(x, low, high):
    desirability_class = desirability_catalogue.get("leftstep")
    desirability_instance = desirability_class()
    desirability = desirability_instance
    utility_function = desirability.utility_function

    y = utility_function(x, low=low, high=high, shift=0.0)
    y = np.float32(y)
    parameters = Parameters(
        **{
            "type": "type",
            "low": low,
            "high": high,
        }
    )
    func_reference = LeftStep(params=parameters)
    y_ref = func_reference([x])

    assert y == (x <= low)
    assert y == pytest.approx(y_ref, abs=1e-6)


@pytest.mark.parametrize("x", [float(x) for x in range(-20, 20, 5)])
@pytest.mark.parametrize("low, high", [(1.0, 2.0), (0.0, 1.0), (-1.0, 0.0)])
def test_rightstep_equivalence_to_func_reference(x, low, high):
    desirability_class = desirability_catalogue.get("rightstep")
    desirability_instance = desirability_class()
    desirability = desirability_instance
    utility_function = desirability.utility_function

    y = utility_function(x, low=low, high=high, shift=0.0)
    y = np.float32(y)
    parameters = Parameters(
        **{
            "type": "type",
            "low": low,
            "high": high,
        }
    )
    func_reference = RightStep(params=parameters)
    y_ref = func_reference([x])
    assert y == (x >= high)
    assert y == pytest.approx(y_ref, abs=1e-6)


@pytest.mark.parametrize("x", [float(x) for x in range(-20, 20, 5)])
@pytest.mark.parametrize("low, high", [(1.0, 2.0), (0.0, 1.0), (-1.0, 0.0)])
def test_step_equivalence_to_func_reference(x, low, high):
    desirability_class = desirability_catalogue.get("step")
    desirability_instance = desirability_class()
    desirability = desirability_instance
    utility_function = desirability.utility_function

    y = utility_function(x, low=low, high=high, shift=0.0)
    y = np.float32(y)
    parameters = Parameters(
        **{
            "type": "type",
            "low": low,
            "high": high,
        }
    )
    func_reference = Step(params=parameters)
    y_ref = func_reference([x])
    assert y == ((x <= high) & (x >= low))
    assert y == pytest.approx(y_ref, abs=1e-6)
