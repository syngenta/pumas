# test_scoring_data_types.py
import pytest

from pumas.scoring_framework.factory import ScoringStrategyFactory, StrategyType


@pytest.mark.parametrize(
    "strategy_type, data_fixture",
    [
        (StrategyType.NUMERIC, "sample_numeric_data"),
        (StrategyType.FOERP, "sample_uncertain_data"),
    ],
)
def test_scoring_computation(strategy_type, data_fixture, sample_profile, request):
    data = request.getfixturevalue(data_fixture)
    strategy = ScoringStrategyFactory.create_strategy(strategy_type, sample_profile)
    input_data = ScoringStrategyFactory.create_input_data(strategy_type, data)

    results = strategy.compute(input_data)
    assert results is not None
    assert len(results.results) == 10

    for compound_id in data.keys():
        result = results.get_result_by_uid(compound_id)
        assert result is not None
        assert result.aggregated_score is not None
        assert len(result.desirability_scores) == 3
        for obj in sample_profile.objectives:
            assert obj.name in result.desirability_scores


def test_numeric_scoring(sample_profile, sample_numeric_data):
    strategy = ScoringStrategyFactory.create_strategy(
        StrategyType.NUMERIC, sample_profile
    )
    input_data = ScoringStrategyFactory.create_input_data(
        StrategyType.NUMERIC, sample_numeric_data
    )
    results = strategy.compute(input_data)

    compound_result = results.get_result_by_uid("compound0")
    assert isinstance(compound_result.aggregated_score, float)
    for score in compound_result.desirability_scores.values():
        assert isinstance(score, float)


def test_uncertain_scoring(sample_profile, sample_uncertain_data):
    strategy = ScoringStrategyFactory.create_strategy(
        StrategyType.FOERP, sample_profile
    )
    input_data = ScoringStrategyFactory.create_input_data(
        StrategyType.FOERP, sample_uncertain_data
    )
    results = strategy.compute(input_data)

    compound_result = results.get_result_by_uid("compound0")
    assert hasattr(compound_result.aggregated_score, "nominal_value")
    assert hasattr(compound_result.aggregated_score, "std_dev")
    for score in compound_result.desirability_scores.values():
        assert hasattr(score, "nominal_value")
        assert hasattr(score, "std_dev")


# Add more specific tests as needed
