import pytest

from pumas.dataframes.dataframe import DataFrame
from pumas.framework import framework_catalogue
from pumas.framework.score_value import FrameworkScoreValue
from pumas.scoring_profile.models import Profile


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
    dataframe = DataFrame(row_data=data)
    dataframe.set_index_from_column(column_name="uid")
    return dataframe


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
    return Profile(**scoring_profile_dict)


@pytest.fixture()
def framework():
    return FrameworkScoreValue


def test_aggregation_catalogue():
    assert "score_value" in framework_catalogue.list_items()


def test_scorer_init(framework, property_dataframe, scoring_profile):
    """
    Test the initialization of a Scorer object.
    """
    scorer = framework(properties=property_dataframe, scoring_profile=scoring_profile)
    assert scorer is not None
    assert scorer._desirability_functions_map is not None
    scorer.compute()
