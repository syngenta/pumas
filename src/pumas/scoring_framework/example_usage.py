# example_usage.py
from pumas.scoring_framework.factory import ScoringStrategyFactory, StrategyType
from pumas.scoring_profile.scoring_profile import ScoringProfile


def example_usage():
    # Create scoring profile
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

    profile = ScoringProfile.model_validate(profile_data)

    size = 10
    # Example numeric data
    numeric_data_i = {"quality": 2.5, "efficiency": 7.8, "cost": 0.9}

    numeric_data = {f"compound{str(i)}": numeric_data_i for i in range(size)}

    # Example UFloat data
    uncertain_data_i = {
        "quality": {"nominal_value": 2.5, "std_dev": 0.1},
        "efficiency": {"nominal_value": 7.8, "std_dev": 0.2},
        "cost": {"nominal_value": 0.9, "std_dev": 0.05},
    }

    uncertain_data = {f"compound{str(i)}": uncertain_data_i for i in range(size)}

    # Using numeric strategy
    numeric_input = ScoringStrategyFactory.create_input_data(
        StrategyType.NUMERIC, numeric_data
    )

    numeric_strategy = ScoringStrategyFactory.create_strategy(
        StrategyType.NUMERIC, profile
    )
    numeric_results = numeric_strategy.compute(numeric_input)
    compound_numeric = numeric_results.get_result_by_uid("compound1")
    print(compound_numeric)

    # Using uncertain strategy
    uncertain_input = ScoringStrategyFactory.create_input_data(
        StrategyType.FOERP, uncertain_data
    )
    uncertain_strategy = ScoringStrategyFactory.create_strategy(
        StrategyType.FOERP, profile
    )
    uncertain_results = uncertain_strategy.compute(uncertain_input)
    compound_ufloat = uncertain_results.get_result_by_uid("compound1")
    print(compound_ufloat)


if __name__ == "__main__":
    example_usage()
