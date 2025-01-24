import pytest
from uncertainties import ufloat

from pumas.architecture.exceptions import (
    InvalidAcceptedValuesError,
    InvalidBoundaryDefinitionError,
    InvalidBoundaryError,
    InvalidParameterNameError,
    InvalidParameterTypeError,
)
from pumas.architecture.parameters import (
    BoolParameter,
    FloatParameter,
    IntParameter,
    IterableParameter,
    MappingParameter,
    StrParameter,
    UFloatParameter,
)


# Tests for IntParameter
@pytest.mark.parametrize(
    "default,min_value,max_value,expected",
    [
        (5, 0, 10, 5),
        (0, 0, 10, 0),
        (10, 0, 10, 10),
    ],
)
def test_int_parameter_valid(default, min_value, max_value, expected):
    """Test creating an IntParameter with valid settings."""
    param = IntParameter(name="TestInt", default=default, min=min_value, max=max_value)

    assert param.value == expected


@pytest.mark.parametrize(
    "default,min_value,max_value",
    [
        (5, 5, 4),  # min is greater than max
    ],
)
def test_int_parameter_default_out_of_range(default, min_value, max_value):
    """Test IntParameter raises InvalidBoundaryError when default is out of range."""
    with pytest.raises(InvalidBoundaryDefinitionError):
        IntParameter(name="TestInt", default=default, min=min_value, max=max_value)


def test_int_parameter_set_value_out_of_range():
    """Test IntParameter raises InvalidBoundaryError when set value is out of range."""
    param = IntParameter(name="TestInt", default=5, min=0, max=10)
    with pytest.raises(InvalidBoundaryError):
        param.set_value(15)


@pytest.mark.parametrize("value", [11, 0])
def test_int_parameter_invalid_value_boundary(value):
    """Test IntParameter for various invalid scenarios."""
    param = IntParameter(name="TestInt", default=5, min=1, max=10)
    with pytest.raises(InvalidBoundaryError):
        param.set_value(value)


@pytest.mark.parametrize("value", [True, 1.0, "invalid type"])
def test_int_parameter_invalid_value_type(value):
    """Test IntParameter for various invalid scenarios."""
    param = IntParameter(name="TestInt", default=5, min=1, max=10)
    with pytest.raises(InvalidParameterTypeError):
        param.set_value(value)


def test_int_parameter_set_value_none():
    """Test IntParameter accepts None as a valid value if not restricted."""
    param = IntParameter(name="TestInt", default=None, min=0, max=10)
    assert param.value is None


@pytest.mark.parametrize(
    "default,min_value,max_value,expected",
    [
        (5.5, 1.0, 10.0, 5.5),
        (1.0, 1.0, 10.0, 1.0),
        (10.0, 1.0, 10.0, 10.0),
    ],
)
def test_float_parameter_valid(default, min_value, max_value, expected):
    """Test creating a FloatParameter with valid settings."""
    param = FloatParameter(
        name="TestFloat", default=default, min=min_value, max=max_value
    )
    assert param.value == expected


@pytest.mark.parametrize("value", [11.0, 0.5])
def test_float_parameter_invalid_value_boundary(value):
    """Test FloatParameter for various invalid scenarios."""
    param = FloatParameter(name="TestFloat", default=5.0, min=1.0, max=10.0)
    with pytest.raises(InvalidBoundaryError):
        param.set_value(value)


@pytest.mark.parametrize("value", [True, 1, "invalid type"])
def test_float_parameter_invalid_value_type(value):
    """Test FloatParameter for various invalid scenarios."""
    param = FloatParameter(name="TestFloat", default=5.0, min=1.0, max=10.0)
    with pytest.raises(InvalidParameterTypeError):
        param.set_value(value)


@pytest.mark.parametrize(
    "accepted_values,default,expected",
    [
        (["option1", "option2"], "option1", "option1"),
        (["option1", "option2"], "option2", "option2"),
        ([], "any_value", "any_value"),  # No accepted values defined
    ],
)
def test_str_parameter_valid(accepted_values, default, expected):
    """Test creating a StrParameter with valid settings."""
    param = StrParameter(
        name="TestStr", accepted_values=accepted_values, default=default
    )
    assert param.value == expected


def test_str_parameter_invalid_value():
    """Test StrParameter raises InvalidAcceptedValuesError
    when set value is not accepted."""
    param = StrParameter(name="TestStr", accepted_values=["option1", "option2"])
    with pytest.raises(InvalidAcceptedValuesError):
        param.set_value("invalid option")


@pytest.mark.parametrize("value", [True, 1, 1.5])
def test_str_parameter_invalid_value_type(value):
    """Test StrParameter for various invalid scenarios."""
    param = StrParameter(name="TestStr", accepted_values=["option1", "option2"])
    with pytest.raises(InvalidParameterTypeError):
        param.set_value(value)


@pytest.mark.parametrize(
    "default,expected",
    [
        (True, True),
        (False, False),
    ],
)
def test_bool_parameter_valid(default, expected):
    """Test creating a BoolParameter with valid settings."""
    param = BoolParameter(name="TestBool", default=default)
    assert param.value is expected


@pytest.mark.parametrize("value", ["not a bool", 0, 1.0])
def test_bool_parameter_set_value_wrong_type(value):
    """Test BoolParameter raises InvalidParameterType
    when set value is of wrong type."""
    param = BoolParameter(name="TestBool", default=True)
    with pytest.raises(InvalidParameterTypeError):
        param.set_value(value)


def test_str_parameter_none_in_accepted_values():
    """Test StrParameter handling None in accepted values."""
    with pytest.raises(InvalidBoundaryDefinitionError):
        StrParameter(name="TestStr", accepted_values=[None, "option1"])


@pytest.mark.parametrize(
    "param_class,default_value",
    [
        (IntParameter, 5),
        (FloatParameter, 5.0),
        (StrParameter, "option1"),
        (BoolParameter, True),
    ],
)
def test_parameter_default_none(param_class, default_value):
    """Test creating parameters with None as default value."""
    param = param_class(name="TestParam", default=None)
    assert param.value is None


def test_parameter_name_type_validation():
    """Test that providing a non-string name raises InvalidParameterType."""
    with pytest.raises(InvalidParameterNameError):
        IntParameter(name=123, default=5)


# Tests to ensure the name parameter validation
@pytest.mark.parametrize(
    "name",
    [
        (""),  # empty name!
        (" "),  # empty name!
        ("\n"),  # empty name!
        ("\t"),  # empty name!
        (None),
        (123),  # An invalid type (int instead of str)
        (True),  # An invalid type (bool instead of str)
    ],
)
def test_parameter_invalid_name(name):
    """Test creating a parameter with an invalid name."""
    with pytest.raises(InvalidParameterNameError):
        IntParameter(name=name, default=1)


def test_parameter_name_validation():
    with pytest.raises(InvalidParameterNameError):
        IntParameter(name=5, default=10)


# Test setting a FloatParameter within range
def test_float_parameter_set_value_in_range():
    param = FloatParameter(name="TestFloat", default=0.0, min=0.0, max=100.0)
    param.set_value(99.9)
    assert param.value == pytest.approx(99.9)


# Test setting a FloatParameter out of range
def test_float_parameter_set_value_out_of_range():
    param = FloatParameter(name="TestFloat", default=0.0, min=0.0, max=100.0)
    with pytest.raises(InvalidBoundaryError):
        param.set_value(101.0)


# Test setting an IntParameter to its minimum bound
def test_int_parameter_set_to_minimum():
    param = IntParameter(name="TestInt", min=0, max=10)
    param.set_value(0)
    assert param.value == 0


# Test setting an IntParameter to its maximum bound
def test_int_parameter_set_to_maximum():
    param = IntParameter(name="TestInt", min=0, max=10)
    param.set_value(10)
    assert param.value == 10


# Test creation of any parameter with defaults and without defining boundaries
@pytest.mark.parametrize(
    "param_class, default, expected_value",
    [
        (IntParameter, 5, 5),
        (FloatParameter, 5.5, 5.5),
        (StrParameter, "test", "test"),
        (BoolParameter, True, True),
        (IterableParameter, [1, 2], [1, 2]),
        (MappingParameter, {1: "one", 2: "two"}, {1: "one", 2: "two"}),
    ],
)
def test_parameter_creation_without_bounds(param_class, default, expected_value):
    param = param_class(name="TestParam", default=default)
    assert param.value == expected_value


# Test setting and unsetting the value for each parameter type
@pytest.mark.parametrize(
    "param_class, value",
    [
        (IntParameter, 10),
        (FloatParameter, 10.0),
        (StrParameter, "new_value"),
        (BoolParameter, False),
        (IterableParameter, [1, 2]),
        (MappingParameter, {1: "one", 2: "two"}),
    ],
)
def test_parameter_set_and_unset_value(param_class, value):
    param = param_class(name="TestParam", default=None)
    param.set_value(value)
    assert param.value == value
    param.set_value(None)
    assert param.value is None


# Test if invalid min/max bounds throw the right error for FloatParameter
@pytest.mark.parametrize(
    "min_value, max_value",
    [
        (5.0, 4.0),  # Invalid: min is greater than max
        ("low", "high"),  # Invalid type for min and max
        ("low", 10.0),  # Invalid type for  max
        (1.0, "high"),  # Invalid type for min
    ],
)
def test_float_parameter_invalid_bounds(min_value, max_value):
    with pytest.raises(InvalidBoundaryDefinitionError):
        FloatParameter(name="TestFloat", default=4.5, min=min_value, max=max_value)


# Test if invalid min/max bounds throw the right error for IntParameter
@pytest.mark.parametrize(
    "min_value, max_value",
    [
        (5, 3),  # Invalid: min is greater than max
        ("low", "high"),  # Invalid type for min and max
        ("low", 10),  # Invalid type for  max
        (1, "high"),  # Invalid type for min
    ],
)
def test_int_parameter_invalid_bounds(min_value, max_value):
    with pytest.raises(InvalidBoundaryDefinitionError):
        IntParameter(name="TestFloat", default=4, min=min_value, max=max_value)


# Test for StrParameter with special characters and empty string in accepted values
@pytest.mark.parametrize(
    "accepted_values, default",
    [
        (["option1", "$pec!@l"], "$pec!@l"),
        (["", "empty"], "empty"),
    ],
)
def test_str_parameter_special_and_empty_accepted_values(accepted_values, default):
    param = StrParameter(
        name="TestStr", accepted_values=accepted_values, default=default
    )
    assert param.value == default


# Test for BoolParameter accepting only boolean types
@pytest.mark.parametrize("default", [True, False, None])
def test_bool_parameter_only_accepts_booleans(default):
    # Should not raise if True, False or None
    param = BoolParameter(name="TestBool", default=default)
    assert param.value == default


# Test the UFloatParameter with valid settings and default values
@pytest.mark.parametrize(
    "default, min_value, max_value",
    [
        (ufloat(1, 0.1), ufloat(0, 0.1), ufloat(2, 0.1)),
        (ufloat(0, 0), ufloat(-1, 0.1), ufloat(1, 0.2)),
        (ufloat(2, 0.5), ufloat(1, 0.01), ufloat(3, 0.01)),
    ],
)
def test_ufloat_parameter_valid(default, min_value, max_value):
    """Test creating a UFloatParameter with valid settings."""
    param = UFloatParameter(
        name="TestUFloat", default=default, min=min_value, max=max_value
    )

    assert param.value == default
    assert param.min == min_value
    assert param.max == max_value


# Test UFloatParameter for invalid scenarios based on nominal and std_dev
@pytest.mark.parametrize(
    "default, min_value, max_value",
    [
        (
            ufloat(2.1, 0.1),
            ufloat(0, 0.1),
            ufloat(2, 0.1),
        ),  # Nominal value out of range
        (
            ufloat(0.5, 0.05),
            ufloat(1, 0.1),
            ufloat(2, 0.1),
        ),  # Nominal value less than min
        (
            ufloat(-0.1, 0.0),
            ufloat(0, 0.0),
            ufloat(1, 0.2),
        ),  # Nominal value less than min
    ],
)
def test_ufloat_parameter_invalid_range(default, min_value, max_value):
    """Test UFloatParameter raises InvalidBoundaryError when default is out of range."""
    with pytest.raises(InvalidBoundaryError):
        UFloatParameter(
            name="TestUFloat", default=default, min=min_value, max=max_value
        )


# Test UFloatParameter set_value method with a new valid UFloat
def test_ufloat_parameter_set_value():
    param = UFloatParameter(name="TestUFloat", min=ufloat(0, 0.01), max=ufloat(1, 0.01))
    new_value = ufloat(0.5, 0.05)
    param.set_value(new_value)
    assert param.value == new_value


# Test UFloatParameter setting None as a value
def test_ufloat_parameter_none_value():
    param = UFloatParameter(name="TestUFloat", default=ufloat(0.5, 0.05))
    param.set_value(None)
    assert param.value is None


# Test UFloatParameter with mismatching uncertainty in range and default
def test_ufloat_parameter_mismatched_uncertainty():
    default = ufloat(0.5, 0.05)
    min_value = ufloat(0, 0.1)
    max_value = ufloat(1, 0.1)
    param = UFloatParameter(
        name="TestUFloat", default=default, min=min_value, max=max_value
    )
    assert param.value == default


def test_iterable_parameter():
    param = IterableParameter(name="TestIterable")
    assert param


def test_mapping_parameter():
    param = MappingParameter(name="TestIterable")
    assert param
