import pytest

from pumas.aggregation.aggregation_utils import (
    check_length_match,
    check_negative_values,
    check_negative_weights,
    check_null_values,
    check_null_weights,
    filter_out_null_values_weights_pairs,
    report_null_values,
    report_null_weights,
)
from pumas.aggregation.exceptions import (
    AggregationNegativeValuesException,
    AggregationNegativeWeightsException,
    AggregationNullValuesException,
    AggregationNullValuesWarning,
    AggregationNullWeightsException,
    AggregationNullWeightsWarning,
    AggregationValuesToWeightLengthMismatchException,
)


def test_check_length_match_success(dataset_1):
    """Test for success if length of value and weights match."""
    values, weights = dataset_1
    assert check_length_match(values, weights) is None


def test_check_length_match_failure(dataset_5):
    """Test error raising if length do not match."""
    values, weights = dataset_5
    with pytest.raises(AggregationValuesToWeightLengthMismatchException):
        check_length_match(values, weights)


# You would replicate these test patterns for each dataset and helper function.
# Here are some more test examples:
def test_check_negative_values_success(dataset_1):
    """Test for success if all values are positive."""
    values, _ = dataset_1
    assert check_negative_values(values) is None


def test_check_negative_weights_failure(dataset_6):
    """Test error raising if any weight is negative."""
    _, weights = dataset_6
    with pytest.raises(AggregationNegativeWeightsException):
        check_negative_weights(weights)


def test_check_negative_values_failure(dataset_7):
    """Test error raising if any value is negative."""
    values, _ = dataset_7
    with pytest.raises(AggregationNegativeValuesException):
        check_negative_values(values)


def test_check_null_values_with_none(dataset_8):
    """Test error raising if any value is None."""
    values, _ = dataset_8
    with pytest.raises(AggregationNullValuesException):
        check_null_values(values)


def test_check_null_values_with_float_nan(dataset_9):
    """Test error raising if any value is float("nan")."""
    values, _ = dataset_9
    with pytest.raises(AggregationNullValuesException):
        check_null_values(values)


def test_check_null_values_with_ufloat_nv_nan(dataset_10):
    """Test error raising if any value has nominal_value nan."""
    values, _ = dataset_10
    with pytest.raises(AggregationNullValuesException):
        check_null_values(values)


def test_check_null_values_with_ufloat_std_dev_nan(dataset_11):
    """Test error raising if any value is ufloat(2.5, std_dev=nan)"""
    values, _ = dataset_11
    with pytest.raises(AggregationNullValuesException):
        check_null_values(values)


def test_report_null_values_logs_warning(dataset_8, dataset_9, dataset_10):
    for dataset in [dataset_8, dataset_9, dataset_10]:
        values, _ = dataset
        with pytest.warns(AggregationNullValuesWarning):
            report_null_values(values)


def test_check_null_weights_with_none(dataset_12):
    """Test error raising if any weight is None."""
    _, weights = dataset_12
    with pytest.raises(AggregationNullWeightsException):
        check_null_weights(weights)


def test_check_null_weights_with_float_nan(dataset_13):
    """Test error raising if any value is float("nan")."""
    _, weights = dataset_13
    with pytest.raises(AggregationNullWeightsException):
        check_null_weights(weights)


def test_check_null_weights_with_ufloat_nv_nan(dataset_14):
    """Test error raising if any weight is ufloat(nan, std_dev=x)."""
    _, weights = dataset_14
    with pytest.raises(AggregationNullWeightsException):
        check_null_weights(weights)


def test_check_null_weights_with_ufloat_std_dev_nan(dataset_15):
    """Test error raising if any weight is ufloat(x, std_dev=nan)."""
    _, weights = dataset_15
    with pytest.raises(AggregationNullWeightsException):
        check_null_weights(weights)


def test_report_null_weights_logs_warning(dataset_12, dataset_13, dataset_14):
    for dataset in [dataset_12, dataset_13, dataset_14]:
        _, weights = dataset
        with pytest.warns(AggregationNullWeightsWarning):
            report_null_weights(weights)


def test_filter_out_null_values_nan_handling(dataset_8, dataset_9, dataset_10):
    """Test that filter_out_null_values_weights_pairs() removes NaN values
    and the corresponding weight."""
    for dataset in [dataset_8, dataset_9, dataset_10]:
        values, weights = dataset
        selected_values, selected_weights = filter_out_null_values_weights_pairs(
            values, weights
        )
        assert len(selected_weights) == len(weights) - 1
        assert len(selected_values) == len(values) - 1


def test_filter_out_null_weights_nan_handling(dataset_12, dataset_13, dataset_14):
    """Test that filter_out_null_values_weights_pairs() removes NaN weights
    and the corresponding values."""
    for dataset in [dataset_12, dataset_13, dataset_14]:
        values, weights = dataset

        selected_values, selected_weights = filter_out_null_values_weights_pairs(
            values, weights
        )
        assert len(selected_weights) == len(weights) - 1
        assert len(selected_values) == len(values) - 1
