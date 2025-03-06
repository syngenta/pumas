import pytest

from pumas.aggregation import aggregation_catalogue
from pumas.aggregation.weighted_deviation_index import WeightedDeviationIndexAggregation
from pumas.uncertainty.uncertainties_wrapper import UFloat, ufloat


@pytest.fixture
def aggregation(dataset_1):
    """This aggregation method requires the ideal value to be set.
    Here for testing purposes, the ideal value is set
    to the maximum value in the dataset values.
    """
    values, _ = dataset_1
    ideal_value = max(values)
    return WeightedDeviationIndexAggregation(params={"ideal_value": ideal_value})


@pytest.fixture
def expected_value_float():
    return -0.6936413589162196


@pytest.fixture
def expected_result_ufloat():
    return ufloat(nominal_value=-0.69, std_dev=0.08)


@pytest.fixture
def expected_result_float_mask_null_values():
    return -0.8569533817705186


@pytest.fixture
def expected_result_float_mask_null_weights():
    return -0.8569533817705186


@pytest.fixture
def expected_result_ufloat_mask_null_values():
    return ufloat(nominal_value=-0.86, std_dev=0.09)


def test_aggregation_catalogue():
    assert "deviation_index" in aggregation_catalogue.list_items()


def test_weighted_deviation_index_numeric(aggregation, dataset_1, expected_value_float):
    """Test weighted deviation ndex with float values."""
    values, weights = dataset_1
    result = aggregation.compute_numeric(values=values, weights=weights)
    assert isinstance(result, float)
    assert result == pytest.approx(expected_value_float)


def test_weighted_deviation_index_ufloat(
    aggregation, dataset_2, expected_value_float, expected_result_ufloat
):
    """Test weighted deviation ndex with ufloat values."""
    values, weights = dataset_2
    result = aggregation.compute_ufloat(values=values, weights=weights)
    assert isinstance(result, UFloat)
    assert result.nominal_value == pytest.approx(expected_value_float)  # type: ignore
    assert str(result) == str(expected_result_ufloat)


def test_weighted_deviation_index_numeric_masking_null_values(
    aggregation, dataset_8a, dataset_8b, expected_result_float_mask_null_values
):
    """Test weighted deviation ndex numeric masking None or nan values."""
    for dataset in [dataset_8a, dataset_8b]:
        values, weights = dataset
        result = aggregation.compute_numeric(values=values, weights=weights)
        assert isinstance(result, float)
        assert result == pytest.approx(expected_result_float_mask_null_values)


def test_weighted_deviation_index_ufloat_masking_null_values(
    aggregation,
    dataset_9a,
    dataset_9b,
    dataset_9c,
    dataset_9d,
    expected_result_float_mask_null_values,
    expected_result_ufloat_mask_null_values,
):
    """Test weighted deviation ndex ufloat masking None or nan values."""
    for dataset in [dataset_9a, dataset_9b, dataset_9c, dataset_9d]:
        values, weights = dataset
        result = aggregation.compute_ufloat(values=values, weights=weights)
        assert isinstance(result, UFloat)
        assert result.nominal_value == pytest.approx(  # type: ignore
            expected_result_float_mask_null_values
        )
        assert str(result) == str(expected_result_ufloat_mask_null_values)


def test_weighted_deviation_index_numeric_masking_null_weights(
    aggregation, dataset_10a, dataset_10b, expected_result_float_mask_null_weights
):
    """Test weighted deviation ndex numeric masking None or nan weights."""
    for dataset in [dataset_10a, dataset_10b]:
        values, weights = dataset
        result = aggregation.compute_numeric(values=values, weights=weights)
        assert isinstance(result, float)
        assert result == pytest.approx(expected_result_float_mask_null_weights)
