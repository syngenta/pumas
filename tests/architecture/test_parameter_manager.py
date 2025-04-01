from typing import Any, Dict

import pytest

from pumas.architecture.exceptions import (
    InvalidAcceptedValuesError,
    InvalidBoundaryError,
    InvalidParameterAttributeError,
    InvalidParameterTypeError,
    ParameterDefinitionError,
    ParameterNotFoundError,
    ParameterUpdateAttributeWarning,
)
from pumas.architecture.parameters import (
    BoolParameter,
    FloatParameter,
    IntParameter,
    ParameterManager,
    StrParameter,
)


def test_parameter_manager_initialization():
    """Test that ParameterManager correctly initializes
    with valid parameter definitions."""
    param_defs: Dict[str, Dict[str, Any]] = {
        "int_param": {"type": "int", "default": 5, "min": 0, "max": 10},
        "float_param": {"type": "float", "default": 3.14, "min": 0.0, "max": 10.0},
        "str_param": {
            "type": "str",
            "default": "test",
            "accepted_values": ["test", "example"],
        },
        "bool_param": {"type": "bool", "default": True},
    }
    pm = ParameterManager(param_defs)

    assert isinstance(pm.parameters_map["int_param"], IntParameter)
    assert isinstance(pm.parameters_map["float_param"], FloatParameter)
    assert isinstance(pm.parameters_map["str_param"], StrParameter)
    assert isinstance(pm.parameters_map["bool_param"], BoolParameter)


def test_parameter_manager_invalid_type():
    """Test that ParameterManager raises a ParameterDefinitionError
    when given an invalid parameter type."""
    param_defs = {"invalid_param": {"type": "invalid_type"}}
    with pytest.raises(
        ParameterDefinitionError,
        match="No Parameter class registered for type 'invalid_type'",
    ):
        ParameterManager(param_defs)


def test_parameter_manager_missing_type():
    """Test that ParameterManager raises a ParameterDefinitionError
    when a parameter definition is missing the 'type'
    key."""
    param_defs = {"missing_type_param": {"default": 5}}
    with pytest.raises(
        ParameterDefinitionError,
        match="Parameter 'missing_type_param' is missing a type definition",
    ):
        ParameterManager(param_defs)


def test_set_parameter_value():
    """Test that set_parameter_value correctly updates the value of a parameter."""
    param_defs = {"int_param": {"type": "int", "default": 5, "min": 0, "max": 10}}
    pm = ParameterManager(param_defs)

    pm.set_parameter_value("int_param", 7)
    assert pm.parameters_map["int_param"].value == 7


def test_set_parameter_value_invalid():
    """Test that set_parameter_value raises an InvalidParameterAttributeError
    when setting an invalid value."""
    param_defs = {"int_param": {"type": "int", "default": 5, "min": 0, "max": 10}}
    pm = ParameterManager(param_defs)

    with pytest.raises(InvalidBoundaryError):
        pm.set_parameter_value("int_param", 15)


def test_set_parameter_value_nonexistent():
    """Test that set_parameter_value raises a ParameterNotFoundError
    when trying to set a value for a non-existent parameter."""
    param_defs = {"int_param": {"type": "int", "default": 5}}
    pm = ParameterManager(param_defs)

    with pytest.raises(
        ParameterNotFoundError, match="Parameter 'nonexistent_param' does not exist"
    ):
        pm.set_parameter_value("nonexistent_param", 10)


def test_set_parameter_attributes():
    """Test that set_parameter_attributes correctly updates
    the attributes of a parameter."""
    param_defs = {"int_param": {"type": "int", "default": 5, "min": 0, "max": 10}}
    pm = ParameterManager(param_defs)

    pm.set_parameter_attributes("int_param", {"min": -5, "max": 15})

    param = pm.parameters_map["int_param"]
    if isinstance(param, IntParameter):
        assert param.min == -5
        assert param.max == 15
    else:
        pytest.fail("Expected IntParameter")


def test_set_parameter_attributes_nonexistent():
    """Test that set_parameter_attributes raises a ParameterNotFoundError
    when trying to update attributes of a non-existent parameter."""
    param_defs = {"int_param": {"type": "int", "default": 5}}
    pm = ParameterManager(param_defs)

    with pytest.raises(
        ParameterNotFoundError, match="Parameter 'nonexistent_param' does not exist"
    ):
        pm.set_parameter_attributes(name="nonexistent_param", attributes={"min": 0})


def test_set_parameter_attributes_value_ignored():
    """Test that set_parameter_attributes ignores
    attempts to update the 'value' attribute and raises a warning."""
    param_defs = {"int_param": {"type": "int", "default": 5, "min": 0}}
    pm = ParameterManager(param_defs)

    with pytest.warns(ParameterUpdateAttributeWarning) as warning_info:
        pm.set_parameter_attributes("int_param", {"value": 10, "min": 0})

    assert len(warning_info) == 1
    assert (
        "Attempt to update 'value' attribute for parameter 'int_param' ignored"
        in str(warning_info[0].message)
    )
    param = pm.parameters_map["int_param"]
    if isinstance(param, IntParameter):
        assert param.value == 5  # Value should remain unchanged
        assert param.min == 0  # Other attributes should be updated
    else:
        pytest.fail("Expected IntParameter")


def test_get_parameters_values():
    """Test that get_parameters_values returns the correct
    dictionary of parameter values."""
    param_defs = {
        "int_param": {"type": "int", "default": 5},
        "float_param": {"type": "float", "default": 3.14},
    }
    pm = ParameterManager(param_defs)

    values = pm.get_parameters_values()
    assert values == {"int_param": 5, "float_param": 3.14}


def test_parameter_type_validation():
    """Test that parameters are correctly validated based on their type."""
    param_defs: Dict[str, Dict[str, Any]] = {
        "int_param": {"type": "int", "default": 5},
        "str_param": {
            "type": "str",
            "default": "test",
            "accepted_values": ["test", "example"],
        },
    }
    pm = ParameterManager(param_defs)

    with pytest.raises(InvalidParameterTypeError):
        pm.set_parameter_value("int_param", "not an int")

    with pytest.raises(InvalidAcceptedValuesError):
        pm.set_parameter_value("str_param", "invalid")


def test_invalid_parameter_attribute_error():
    """Test that InvalidParameterAttributeError is raised
    when setting an invalid attribute."""
    param_defs = {"int_param": {"type": "int", "default": 5}}
    pm = ParameterManager(param_defs)

    with pytest.raises(
        InvalidParameterAttributeError,
        match="Invalid attribute for parameter 'int_param'",
    ):
        pm.set_parameter_attributes("int_param", {"invalid_attr": "value"})


def test_get_parameters_values_after_update():
    """Test that get_parameters_values returns updated values after modifying parameters."""  # noqa: E501
    param_defs = {
        "int_param": {"type": "int", "default": 5},
        "float_param": {"type": "float", "default": 3.14},
    }
    pm = ParameterManager(param_defs)

    pm.set_parameter_value("int_param", 10)
    pm.set_parameter_value("float_param", 2.718)

    values = pm.get_parameters_values()
    assert values == {"int_param": 10, "float_param": 2.718}


def test_parameter_manager_no_parameters():
    """Test that ParameterManager correctly
    initializes with no parameter definitions."""
    pm = ParameterManager()
    assert len(pm.parameters_map) == 0
    assert pm.get_parameters_values() == {}


def test_parameter_manager_empty_dict():
    """Test that ParameterManager correctly initializes with an empty dictionary."""
    pm = ParameterManager({})
    assert len(pm.parameters_map) == 0
    assert pm.get_parameters_values() == {}


def test_set_parameter_value_no_parameters():
    """Test that set_parameter_value raises a
    ParameterNotFoundError when there are no parameters."""
    pm = ParameterManager()
    with pytest.raises(
        ParameterNotFoundError, match="Parameter 'test_param' does not exist"
    ):
        pm.set_parameter_value("test_param", 10)


def test_set_parameter_attributes_no_parameters():
    """Test that set_parameter_attributes raises a
    ParameterNotFoundError when there are no parameters."""
    pm = ParameterManager()
    with pytest.raises(
        ParameterNotFoundError, match="Parameter 'test_param' does not exist"
    ):
        pm.set_parameter_attributes("test_param", {"min": 0})
