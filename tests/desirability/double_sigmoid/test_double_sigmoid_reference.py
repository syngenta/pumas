import numpy as np
import pytest

from pumas.desirability import desirability_catalogue
from tests.desirability.external_reference_implementation.double_sigmoid import (
    DoubleSigmoid,
    Parameters,
)


@pytest.fixture
def desirability():
    desirability_class = desirability_catalogue.get("double_sigmoid")
    desirability_instance = desirability_class()
    return desirability_instance


@pytest.fixture
def utility_function(desirability):
    return desirability.utility_function


@pytest.mark.parametrize("x", [float(x) for x in range(-20, 20, 5)])
@pytest.mark.parametrize(
    "low, high, coef_div, coef_si, coef_se",
    [(-5.0, 5.0, 100.0, 150.0, 150.0), (4.0, 9.0, 35.0, 150.0, 150.0)],
)
def test_double_sigmoid_equivalence_to_func_reference(
    utility_function,
    x,
    low,
    high,
    coef_div,
    coef_si,
    coef_se,
):
    y = utility_function(
        x=x,
        low=low,
        high=high,
        coef_div=coef_div,
        coef_si=coef_si,
        coef_se=coef_se,
        shift=0.0,
        base=10.0,
    )
    y = np.float32(y)
    parameters = Parameters(
        **{
            "type": "type",
            "low": low,
            "high": high,
            "coef_div": coef_div,
            "coef_si": coef_si,
            "coef_se": coef_se,
        }
    )
    func_reference = DoubleSigmoid(params=parameters)
    y_ref = func_reference(x)

    # Use larger than default absolute tolerance for very small values (close to 0.0)
    assert y == pytest.approx(y_ref, abs=1e-6), (
        f"Results differ for x={x}, low={low}, high={high}, "
        f"coef_div={coef_div}, coef_si={coef_si}, coef_se={coef_se}: "
        f"Your result: {y}, reference result: {y_ref}"
    )
