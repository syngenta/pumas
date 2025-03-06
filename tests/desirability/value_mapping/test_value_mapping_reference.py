import numpy as np
import pytest

from pumas.desirability.value_mapping import value_mapping
from tests.desirability.external_reference_implementation.value_mapping import (  # type: ignore # noqa: 501
    Parameters,
    ValueMapping,
)


@pytest.fixture
def desirability_utility_function():
    return value_mapping


@pytest.mark.parametrize(
    "mapping",
    [
        {0: 1, 1: 0.2, 2: 0.5},
    ],
)
def test_value_mapping_equivalence_to_func_reference(
    desirability_utility_function, mapping
):
    """This test is here only as a reference for the implementation
    of the value mapping function
    in reality the values should be strings rather than int."""
    values = [0, 0, 1, 2, 1, 2]
    params = {"mapping": mapping, "shift": 0.0}
    y_list = [desirability_utility_function(x=x, **params) for x in values]

    parameters = Parameters(
        **{
            "type": "type",
            "mapping": mapping,
        }
    )
    func_reference = ValueMapping(params=parameters)
    y_ref_list = func_reference(values)

    assert np.all(y_list == y_ref_list)
