import numpy as np
import pytest

from pumas.desirability.sigmoid import sigmoid
from tests.desirability.external_reference_implementation.sigmoids import (  # type: ignore # noqa: 501
    Parameters,
    Sigmoid,
)


@pytest.fixture
def desirability_utility_function():
    return sigmoid


@pytest.mark.parametrize("x", [float(x) for x in range(-20, 20, 5)])
@pytest.mark.parametrize(
    "low, high, k",
    [
        (-5.0, 5.0, 1.0),
        (0.0, 10.0, 2.0),
        (5.0, 10.0, 0.5),
        (-5.0, 5.0, -1.0),
        (0.0, 10.0, -2.0),
        (5.0, 10.0, -0.5),
    ],
)
def test_sigmoid_equivalence_to_func_reference(
    desirability_utility_function,
    x,
    low,
    high,
    k,
):
    params = {"low": low, "high": high, "k": k, "shift": 0.0, "base": 10.0}

    y = desirability_utility_function(x=x, **params)
    y = np.float32(y)

    parameters = Parameters(
        **{
            "type": "type",
            "low": low,
            "high": high,
            "k": k,
        }
    )
    func_reference = Sigmoid(params=parameters)
    y_ref = func_reference(x)

    assert y == pytest.approx(y_ref), (
        f"Results differ for x={x}, low={low}, high={high}, k={k}: "
        f"Your result: {y}, reference result: {y_ref}"
    )
