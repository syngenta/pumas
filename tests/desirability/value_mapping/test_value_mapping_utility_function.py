import math

import pytest

from pumas.desirability.value_mapping import value_mapping


@pytest.fixture
def desirability_utility_function():
    return value_mapping


@pytest.mark.parametrize(
    "mapping, x, expected",
    [
        # Basic cases
        ({"Low": 0.2, "Medium": 0.5, "High": 0.8}, "Low", 0.2),
        ({"Low": 0.2, "Medium": 0.5, "High": 0.8}, "Medium", 0.5),
        ({"Low": 0.2, "Medium": 0.5, "High": 0.8}, "High", 0.8),
        # Case sensitivity
        ({"low": 0.2, "Low": 0.5}, "Low", 0.5),
        # Edge cases
        ({"Extremely Low": 0.2, "Extremely High": 0.5}, "Extremely Low", 0.2),
    ],
)
def test_value_mapping(desirability_utility_function, mapping, x, expected):
    params = {"mapping": mapping, "shift": 0.0}
    result = desirability_utility_function(x=x, **params)
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-9)


def test_value_mapping_invalid_value(desirability_utility_function):
    x = "Very High"
    mapping = {"Low": 0.2, "Medium": 0.5, "High": 0.8}
    params = {"mapping": mapping, "shift": 0.0}

    y = desirability_utility_function(x=x, **params)
    assert math.isnan(y), f"Expected NaN, but got {y}"


@pytest.mark.parametrize(
    "mapping",
    [
        {"Low": 0.2, "Medium": 0.5, "High": 1.1},
        {"Low": -0.1, "Medium": 0.5, "High": 1.0},
    ],
)
def test_invalid_mapping_values(desirability_utility_function, mapping):
    x = "Low"
    params = {"mapping": mapping, "shift": 0.0}
    with pytest.raises(ValueError, match="Mapping values should be between 0 and 1"):
        desirability_utility_function(x=x, **params)


# test the effect of shift on the utility function
@pytest.mark.parametrize("mapping", [{"Low": 0.1, "Medium": 0.5, "High": 0.8}])
def test_value_mapping_shift(desirability_utility_function, mapping):
    shift_value = 0.2
    x_range = list(mapping.keys())

    # Calculate reference values
    params = {"mapping": mapping, "shift": 0.0}
    reference_values = [desirability_utility_function(x=x, **params) for x in x_range]

    # Verify that there are unshifted values below the shift value
    assert any(
        v < shift_value for v in reference_values
    ), "No unshifted values below shift value"

    # Calculate shifted values
    params = {"mapping": mapping, "shift": shift_value}
    shifted_values = [desirability_utility_function(x=x, **params) for x in x_range]

    # Verify properties of shifted values
    for unshifted, shifted in zip(reference_values, shifted_values):
        assert (
            shifted >= unshifted
        ), "Shifted value not higher than or equal to unshifted value"
        assert (
            shifted >= shift_value
        ), "Shifted value not higher than or equal to shift value"

    # Verify that all shifted values are above or at the shift value
    assert all(
        v >= shift_value for v in shifted_values
    ), "No unshifted values below shift value"
