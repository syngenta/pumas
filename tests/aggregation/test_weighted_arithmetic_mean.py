import pytest

from pumas.aggregation import aggregation_catalogue
from pumas.aggregation.weighted_arithmetic_mean import WeightedArithmeticMeanAggregation
from pumas.uncertainty.uncertainties_wrapper import UFloat, ufloat


@pytest.fixture
def aggregation():
    return WeightedArithmeticMeanAggregation()


@pytest.fixture
def expected_value_float():
    return 2.2


@pytest.fixture
def expected_result_ufloat():
    return ufloat(nominal_value=2.2, std_dev=0.1)


@pytest.fixture
def expected_result_float_mask_null_values():
    return 2.0714285714285716


@pytest.fixture
def expected_result_float_mask_null_weights():
    return 2.0714285714285716


@pytest.fixture
def expected_result_ufloat_mask_null_values():
    return ufloat(nominal_value=2.07, std_dev=0.11)


def test_aggregation_catalogue():
    assert "arithmetic_mean" in aggregation_catalogue.list_items()


def test_weighted_arithmetic_mean_numeric(aggregation, dataset_1, expected_value_float):
    """Test weighted arithmetic mean with float values."""
    values, weights = dataset_1
    result = aggregation.compute_numeric(values=values, weights=weights)
    assert isinstance(result, float)
    assert result == pytest.approx(expected_value_float)


def test_weighted_arithmetic_mean_ufloat(
    aggregation, dataset_2, expected_value_float, expected_result_ufloat
):
    """Test weighted arithmetic mean with ufloat values."""
    values, weights = dataset_2
    result = aggregation.compute_ufloat(values=values, weights=weights)
    assert isinstance(result, UFloat)
    assert result.nominal_value == pytest.approx(expected_value_float)  # type: ignore
    assert str(result) == str(expected_result_ufloat)


def test_weighted_arithmetic_mean_numeric_masking_null_values(
    aggregation, dataset_8a, dataset_8b, expected_result_float_mask_null_values
):
    """Test weighted arithmetic mean numeric masking None or nan values."""
    for dataset in [dataset_8a, dataset_8b]:
        values, weights = dataset
        result = aggregation.compute_numeric(values=values, weights=weights)
        assert isinstance(result, float)
        assert result == pytest.approx(expected_result_float_mask_null_values)


def test_weighted_arithmetic_mean_ufloat_masking_null_values(
    aggregation,
    dataset_9a,
    dataset_9b,
    dataset_9c,
    dataset_9d,
    expected_result_float_mask_null_values,
    expected_result_ufloat_mask_null_values,
):
    """Test weighted arithmetic mean ufloat masking None or nan values."""
    for dataset in [dataset_9a, dataset_9b, dataset_9c, dataset_9d]:
        values, weights = dataset
        result = aggregation.compute_ufloat(values=values, weights=weights)
        assert isinstance(result, UFloat)
        assert result.nominal_value == pytest.approx(  # type: ignore
            expected_result_float_mask_null_values
        )
        assert str(result) == str(expected_result_ufloat_mask_null_values)


def test_weighted_arithmetic_mean_numeric_masking_null_weights(
    aggregation, dataset_10a, dataset_10b, expected_result_float_mask_null_weights
):
    """Test weighted arithmetic mean numeric masking None or nan weights."""
    for dataset in [dataset_10a, dataset_10b]:
        values, weights = dataset
        result = aggregation.compute_numeric(values=values, weights=weights)
        assert isinstance(result, float)
        assert result == pytest.approx(expected_result_float_mask_null_weights)
