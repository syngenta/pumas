import pytest

from pumas.aggregation.aggregation_utils import (
    check_length_match,
    check_negative_values,
    check_negative_weights,
    check_null_values,
    check_null_weights,
    fill_weights,
    filter_out_null_values_weights_pairs,
    is_nan_none,
    report_null_values,
    report_null_weights,
    run_data_validation_pipeline,
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
    check_length_match(values, weights)


def test_check_length_match_failure(dataset_5a, dataset_5b, dataset_5c):
    """Test error raising if length do not match."""
    for dataset in [dataset_5a, dataset_5b, dataset_5b]:
        values, weights = dataset
        with pytest.raises(AggregationValuesToWeightLengthMismatchException):
            check_length_match(values, weights)


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


def test_check_null_values_with_none(dataset_8a):
    """Test error raising if any value is None."""
    values, _ = dataset_8a
    with pytest.raises(AggregationNullValuesException):
        check_null_values(values)


def test_check_null_values_with_float_nan(dataset_8b):
    """Test error raising if any value is float("nan")."""
    values, _ = dataset_8b
    with pytest.raises(AggregationNullValuesException):
        check_null_values(values)


def test_check_null_values_with_ufloat_none(dataset_9a):
    """Test error raising if any value is None."""
    values, _ = dataset_9a
    assert any(isinstance(v, type(None)) for v in values)
    assert any(is_nan_none(v) for v in values)
    with pytest.raises(AggregationNullValuesException):
        check_null_values(values)


def test_check_null_values_with_ufloat_nan(dataset_9b):
    """Test error raising if any value is nan."""
    values, _ = dataset_9b
    with pytest.raises(AggregationNullValuesException):
        check_null_values(values)


def test_check_null_values_with_ufloat_with_nan_as_nominal_value(dataset_9c):
    """Test error raising if any value has nominal_value nan."""
    values, _ = dataset_9c
    with pytest.raises(AggregationNullValuesException):
        check_null_values(values)


def test_check_null_values_with_ufloat_with_nan_as_std_dev(dataset_9d):
    """Test error raising if any value is ufloat(2.5, std_dev=nan)"""
    values, _ = dataset_9d
    with pytest.raises(AggregationNullValuesException):
        check_null_values(values)


def test_check_null_weights_with_none(dataset_10a):
    """Test error raising if any weight is None."""
    _, weights = dataset_10a
    with pytest.raises(AggregationNullWeightsException):
        check_null_weights(weights)


def test_check_null_weights_with_float_nan(dataset_10b):
    """Test error raising if any value is float("nan")."""
    _, weights = dataset_10b
    with pytest.raises(AggregationNullWeightsException):
        check_null_weights(weights)


def test_filter_out_null_values_float(dataset_8a, dataset_8b, dataset_11):
    """Test that filter_out_null_values_weights_pairs() removes
    value-weight pair when the value is None or nan."""

    reference_values, reference_weights = dataset_11
    for dataset in [dataset_8a, dataset_8b]:
        values, weights = dataset
        selected_values, selected_weights = filter_out_null_values_weights_pairs(
            values, weights
        )

        assert len(selected_values) == len(reference_values)
        assert len(selected_weights) == len(reference_weights)
        assert all(
            pytest.approx(sv) == rv for sv, rv in zip(selected_values, reference_values)
        )
        assert all(
            pytest.approx(sw) == rw
            for sw, rw in zip(selected_weights, reference_weights)
        )


def test_filter_out_null_values_ufloat(
    dataset_9a, dataset_9b, dataset_9c, dataset_9d, dataset_12
):
    """Test that filter_out_null_values_weights_pairs() removes
    value-weight pair when the ufloat value is None or nan
    or is defined with a nan either as a nominal value or std_dev."""

    reference_values, reference_weights = dataset_12
    for dataset in [dataset_9a, dataset_9b, dataset_9c, dataset_9d]:
        values, weights = dataset
        selected_values, selected_weights = filter_out_null_values_weights_pairs(
            values, weights
        )

        # equality on ufloat instances by comparing their string representations
        assert all(
            str(sv) == str(rv) for sv, rv in zip(selected_values, reference_values)
        )
        assert all(
            pytest.approx(sw) == rw
            for sw, rw in zip(selected_weights, reference_weights)
        )


def test_filter_out_null_weights(dataset_10a, dataset_10b, dataset_11):
    """Test that filter_out_null_values_weights_pairs() removes
    value-weight pair when the weight is None or nan."""

    reference_values, reference_weights = dataset_11
    for dataset in [dataset_10a, dataset_10b]:
        values, weights = dataset
        selected_values, selected_weights = filter_out_null_values_weights_pairs(
            values, weights
        )

        assert len(selected_values) == len(reference_values)
        assert len(selected_weights) == len(reference_weights)
        assert all(
            pytest.approx(sv) == rv for sv, rv in zip(selected_values, reference_values)
        )
        assert all(
            pytest.approx(sw) == rw
            for sw, rw in zip(selected_weights, reference_weights)
        )


def test_report_null_values_logs_warning(
    dataset_8a,
    dataset_8b,
    dataset_9a,
    dataset_9b,
    dataset_9c,
    dataset_9d,
):
    for dataset in [
        dataset_8a,
        dataset_8b,
        dataset_9a,
        dataset_9b,
        dataset_9c,
        dataset_9d,
    ]:
        values, _ = dataset
        with pytest.warns(AggregationNullValuesWarning):
            report_null_values(values)


def test_report_null_weights_logs_warning(dataset_10a, dataset_10b):
    for dataset in [dataset_10a, dataset_10b]:
        _, weights = dataset
        with pytest.warns(AggregationNullWeightsWarning):
            report_null_weights(weights)


def test_values_float_fill_none_weights_with_ones(dataset_1):
    """Test that fill_weights_with_ones() fills weights with ones."""
    values, _ = dataset_1
    weights = None
    filled_weights = fill_weights(values, weights)
    assert len(filled_weights) == len(values)
    assert all(pytest.approx(w) == 1.0 for w in filled_weights)


def test_values_ufloat_fill_none_weights_with_ones(dataset_2):
    """Test that fill_weights_with_ones() fills weights with ones."""
    values, _ = dataset_2
    weights = None
    filled_weights = fill_weights(values, weights)
    assert len(filled_weights) == len(values)
    assert all(pytest.approx(w) == 1.0 for w in filled_weights)


def test_data_validation_pipeline_mask_null_values_float(
    dataset_8a, dataset_8b, dataset_11
):
    """Test that run_data_validation_pipeline()
    removes value-weight pair when the value is None or nan.
    """
    reference_values, reference_weights = dataset_11
    for dataset in [dataset_8a, dataset_8b]:
        values, weights = dataset
        selected_values, selected_weights = run_data_validation_pipeline(
            values, weights
        )

        assert len(selected_values) == len(reference_values)
        assert len(selected_weights) == len(reference_weights)
        assert all(
            pytest.approx(sv) == rv for sv, rv in zip(selected_values, reference_values)
        )
        assert all(
            pytest.approx(sw) == rw
            for sw, rw in zip(selected_weights, reference_weights)
        )


def test_data_validation_pipeline_mask_null_values_ufloat(
    dataset_9a, dataset_9b, dataset_9c, dataset_9d, dataset_12
):
    """Test that run_data_validation_pipeline()
    removes value-weight pair when the ufloat value is None or nan"""
    reference_values, reference_weights = dataset_12
    for dataset in [dataset_9a, dataset_9b, dataset_9c, dataset_9d]:
        values, weights = dataset
        selected_values, selected_weights = run_data_validation_pipeline(
            values, weights
        )

        # equality on ufloat instances by comparing their string representations
        assert all(
            str(sv) == str(rv) for sv, rv in zip(selected_values, reference_values)
        )
        assert all(
            pytest.approx(sw) == rw
            for sw, rw in zip(selected_weights, reference_weights)
        )


def test_data_validation_pipeline_mask_null_weights(
    dataset_10a, dataset_10b, dataset_11
):
    """
    Test that run_data_validation_pipeline()
    removes value-weight pair when the weight is None or nan."""
    reference_values, reference_weights = dataset_11
    for dataset in [dataset_10a, dataset_10b]:
        values, weights = dataset
        selected_values, selected_weights = run_data_validation_pipeline(
            values, weights
        )

        assert len(selected_values) == len(reference_values)
        assert len(selected_weights) == len(reference_weights)
        assert all(
            pytest.approx(sv) == rv for sv, rv in zip(selected_values, reference_values)
        )
        assert all(
            pytest.approx(sw) == rw
            for sw, rw in zip(selected_weights, reference_weights)
        )


def test_data_validation_pipeline_negative_weights_failure(dataset_6):
    """Test error raising if any value is negative."""
    values, weights = dataset_6
    with pytest.raises(AggregationNegativeWeightsException):
        run_data_validation_pipeline(values, weights)


def test_data_validation_pipeline_negative_values_failure(dataset_7):
    """Test error raising if any value is negative."""
    values, weights = dataset_7
    with pytest.raises(AggregationNegativeValuesException):
        run_data_validation_pipeline(values, weights)
