# conftest.py
import pytest

from pumas.scoring_profile.scoring_profile import ScoringProfile


@pytest.fixture
def sample_profile():
    profile_data = {
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
            },
        ],
        "aggregation_function": {"name": "geometric_mean", "parameters": {}},
    }
    return ScoringProfile.model_validate(profile_data)


@pytest.fixture
def sample_numeric_data():
    return {
        f"compound{i}": {"quality": 2.5, "efficiency": 7.8, "cost": 0.9}
        for i in range(10)
    }


@pytest.fixture
def sample_uncertain_data():
    return {
        f"compound{i}": {
            "quality": {"nominal_value": 2.5, "std_dev": 0.1},
            "efficiency": {"nominal_value": 7.8, "std_dev": 0.2},
            "cost": {"nominal_value": 0.9, "std_dev": 0.05},
        }
        for i in range(10)
    }
