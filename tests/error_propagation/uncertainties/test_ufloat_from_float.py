import pytest

from pumas.error_propagation.uncertainties import (
    UFloat,
    float_to_ufloat_conversion_catalogue,
    ufloat_from_float,
    ufloat_from_str,
)


# Test zero_uncertainty method
def test_zero_uncertainty():
    value = 5.0
    converter_class = float_to_ufloat_conversion_catalogue.get("zero_uncertainty")
    converter = converter_class()
    result = converter.convert(value)
    assert isinstance(result, UFloat)
    assert result.nominal_value == pytest.approx(value)
    assert result.std_dev == pytest.approx(0.0)


# Test fixed_value_uncertainty method
def test_fixed_value_uncertainty():
    nominal_value, std_dev = 5.0, 0.5
    converter_class = float_to_ufloat_conversion_catalogue.get("fixed_value")
    converter = converter_class(fixed_uncertainty=std_dev)
    result = converter.convert(nominal_value)
    assert isinstance(result, UFloat)
    assert result.nominal_value == pytest.approx(nominal_value)
    assert result.std_dev == pytest.approx(std_dev)


# Test percentage_uncertainty method with valid percentage
def test_percentage_uncertainty_valid():
    value, percentage = 5.0, 10
    expected_uncertainty = value * (percentage / 100)
    converter_class = float_to_ufloat_conversion_catalogue.get("percentage_of_value")
    converter = converter_class(percentage=percentage)
    result = converter.convert(value)
    assert isinstance(result, UFloat)
    assert result.nominal_value == pytest.approx(value)
    assert result.std_dev == pytest.approx(expected_uncertainty)


# Test invalid percentage in percentage_uncertainty method
def test_percentage_uncertainty_invalid():
    converter_class = float_to_ufloat_conversion_catalogue.get("percentage_of_value")

    with pytest.raises(ValueError):
        _ = converter_class(percentage=-5)

    with pytest.raises(ValueError):
        _ = converter_class(percentage=-150)


# Test multiplier_uncertainty method
def test_multiplier_uncertainty():
    value, multiplier = 5.0, 0.2
    expected_uncertainty = value * multiplier
    converter_class = float_to_ufloat_conversion_catalogue.get("multiplier")
    converter = converter_class(multiplier=multiplier)
    result = converter.convert(value)
    assert isinstance(result, UFloat)
    assert result.nominal_value == pytest.approx(value)
    assert result.std_dev == pytest.approx(expected_uncertainty)


# Test listing available methods
def test_list_available_methods():
    methods = float_to_ufloat_conversion_catalogue.list_items()
    assert isinstance(methods, list)
    assert set(methods) == {
        "zero_uncertainty",
        "fixed_value",
        "percentage_of_value",
        "multiplier",
    }


def test_ufloat_from_str():
    value = "5.0+/-0.5"
    result = ufloat_from_str(value)
    assert isinstance(result, UFloat)
    assert result.nominal_value == pytest.approx(5.0)
    assert result.std_dev == pytest.approx(0.5)


def test_ufloat_from_float():
    value = 5.0
    method = "zero_uncertainty"
    result = ufloat_from_float(value, method)
    assert isinstance(result, UFloat)
    assert result.nominal_value == pytest.approx(value)
    assert result.std_dev == pytest.approx(0.0)


def test_ufloat_from_float_invalid_method():
    with pytest.raises(ValueError):
        ufloat_from_float(value=5.0, method="invalid_method")
