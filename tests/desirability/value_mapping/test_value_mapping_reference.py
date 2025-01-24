import numpy as np
import pytest

from pumas.desirability import desirability_catalogue
from tests.desirability.external_reference_implementation.value_mapping import (
    Parameters,
    ValueMapping,
)


@pytest.fixture
def desirability():
    desirability_class = desirability_catalogue.get("value_mapping")
    desirability_instance = desirability_class()
    return desirability_instance


@pytest.fixture
def utility_function(desirability):
    return desirability.utility_function


@pytest.mark.parametrize("mapping", [{0: 1, 1: 0.2, 2: 0.5}])
def test_value_mapping_equivalence_to_func_reference(utility_function, mapping):
    values = [0, 0, 1, 2, 1, 2]

    y_list = [utility_function(x=x, mapping=mapping) for x in values]

    parameters = Parameters(
        **{
            "type": "type",
            "mapping": mapping,
        }
    )
    func_reference = ValueMapping(params=parameters)
    y_ref_list = func_reference(values)

    assert np.all(y_list == y_ref_list)
