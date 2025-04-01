# test_scoring_general.py
import pytest

from pumas.scoring_framework.factory import ScoringStrategyFactory, StrategyType


def test_profile_validation(sample_profile):
    assert sample_profile.objectives is not None
    assert len(sample_profile.objectives) == 3
    assert sample_profile.aggregation_function is not None


@pytest.mark.parametrize("strategy_type", [StrategyType.NUMERIC, StrategyType.FOERP])
def test_strategy_creation(strategy_type, sample_profile):
    strategy = ScoringStrategyFactory.create_strategy(strategy_type, sample_profile)
    assert strategy is not None
    assert strategy.profile == sample_profile


@pytest.mark.parametrize(
    "strategy_type, data_fixture",
    [
        (StrategyType.NUMERIC, "sample_numeric_data"),
        (StrategyType.FOERP, "sample_uncertain_data"),
    ],
)
def test_input_data_creation(strategy_type, data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    input_data = ScoringStrategyFactory.create_input_data(strategy_type, data)
    assert input_data is not None
    assert len(input_data.data) == 10
