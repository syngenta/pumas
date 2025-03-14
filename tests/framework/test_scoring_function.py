# type: ignore
import pytest

from pumas.scoring_framework.scoring_function import ScoringFunction
from pumas.scoring_profile.scoring_profile import ScoringProfile


# Fixture for a valid DataFrame
@pytest.fixture()
def property_dataframe():
    """
    Provides a DataFrame with properties `quality`, `cost`, and `efficiency`.
    """
    data = [
        {"uid": "A", "quality": 1.0, "cost": 30.0, "efficiency": 0.7},
        {"uid": "B", "quality": 5.0, "cost": 40.0, "efficiency": 0.2},
        {"uid": "C", "quality": 3.0, "cost": 10.0, "efficiency": 0.9},
        {"uid": "D", "quality": 6.0, "cost": 60.0, "efficiency": 0.4},
        {"uid": "E", "quality": 2.0, "cost": 80.0, "efficiency": 0.5},
    ]
    return data


# Fixture for a valid scoring profile
@pytest.fixture()
def scoring_profile():
    """
    Provides a valid scoring profile corresponding to the valid_dataframe.
    """
    scoring_profile_dict = {
        "objectives": [
            {
                "name": "quality",
                "desirability_function": {
                    "name": "sigmoid",
                    "parameters": {
                        "low": 1.0,
                        "high": 10.0,
                        "k": 0.1,
                        "shift": 0.0,
                        "base": 10.0,
                    },
                },
                "weight": 1.0,
                "value_type": "float",
                "kind": "numerical",
            },
            {
                "name": "efficiency",
                "desirability_function": {
                    "name": "sigmoid",
                    "parameters": {
                        "low": 0.2,
                        "high": 0.8,
                        "k": 0.1,
                        "shift": 0.0,
                        "base": 10.0,
                    },
                },
                "weight": 2.0,
                "value_type": "float",
                "kind": "numerical",
            },
            {
                "name": "cost",
                "desirability_function": {
                    "name": "sigmoid",
                    "parameters": {
                        "low": 20.0,
                        "high": 80.0,
                        "k": -0.5,
                        "shift": 0.0,
                        "base": 10.0,
                    },
                },
                "weight": 3.0,
                "value_type": "float",
                "kind": "numerical",
            },
        ],
        "aggregation_function": {
            "name": "geometric_mean",
            "parameters": {},
        },
    }
    return ScoringProfile(**scoring_profile_dict)


def test_initialize_scoring_function(property_dataframe, scoring_profile):
    """
    Test the initialization of the ScoringFunction class.
    """
    scoring_function = ScoringFunction(
        profile=scoring_profile,
    )
    assert scoring_function is not None

    result = scoring_function.compute(data=property_dataframe[0])
    assert result is not None
