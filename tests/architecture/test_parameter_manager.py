import pytest

from pumas.architecture.exceptions import ParameterUpdateAttributeWarning
from pumas.architecture.parameters import FloatParameter, IntParameter, ParameterManager


def test_initialization_with_valid_function():
    """Test that the ParameterManager correctly
    initializes with a well-annotated function."""

    def func(x: int, y: float):
        return x + y

    pm = ParameterManager(input_function=func)
    assert "x" in pm.parameters_map and isinstance(pm.parameters_map["x"], IntParameter)
    assert "y" in pm.parameters_map and isinstance(
        pm.parameters_map["y"], FloatParameter
    )


def test_check_input_function_with_non_callable():
    """Test that the ParameterManager raises a
    ValueError if the input_function is not callable."""
    with pytest.raises(ValueError):
        ParameterManager(input_function="not callable")  # noqa type mismatch on purpose


def test_get_type_names_with_valid_types():
    """Test that the _get_type_names method correctly
    maps parameter names to their type names."""

    def func(x: int, y: float):
        return x + y

    pm = ParameterManager(input_function=func)
    type_names = pm._get_type_names()
    assert type_names == {"x": "int", "y": "float"}


def test_validate_function_parameters_with_unregistered_type():
    """Test that ParameterManager raises
    a ValueError for parameters with unregistered types."""

    class UnknownType:
        pass

    def func(
        x: UnknownType,
    ):
        return x

    with pytest.raises(ValueError):
        ParameterManager(input_function=func)


def test_get_parameters_map_with_default_values():
    """Test that default values are correctly
    assigned to Parameter instances in the map."""

    def func(x: int = 42, y: float = 3.14):
        return x + y

    pm = ParameterManager(input_function=func)
    assert pm.parameters_map["x"].default == 42
    assert pm.parameters_map["y"].default == pytest.approx(3.14)


def test_set_parameter_attributes_valid_update():
    """Test updating parameter attributes while
    preserving the old ones that are not overridden."""

    def func(x: int = 42):
        return x

    pm = ParameterManager(input_function=func)
    pm.set_parameter_attributes("x", {"min": 1, "max": 100})
    assert pm.parameters_map["x"].min == 1
    assert pm.parameters_map["x"].max == 100
    assert pm.parameters_map["x"].default == 42  # Default should stay unchanged


def test_set_parameter_attributes_non_existing_parameter():
    """Test that a ValueError is raised when updating a non-existing parameter."""

    def func(x: int = 42):
        return x

    pm = ParameterManager(input_function=func)
    with pytest.raises(ValueError):
        pm.set_parameter_attributes("y", {"min": 0})


def test_set_parameter_attributes_with_value_update_attempt():
    """Test that updating the 'value'
    property raises a ParameterUpdateAttributeWarning."""

    def func(x: int = 42):
        return x

    pm = ParameterManager(input_function=func)
    with pytest.warns(ParameterUpdateAttributeWarning):
        pm.set_parameter_attributes("x", {"value": 100})


def test_set_parameter_value_valid_assignment():
    """Test that the set_parameter_value
    method correctly sets the value of a parameter."""

    def func(x: int = 42):
        return x

    pm = ParameterManager(input_function=func)
    pm.set_parameter_value("x", 24)
    assert (
        pm.parameters_map["x"].value == 24
    )  # The value attribute of x should now be 24


def test_set_parameter_value_non_existing_parameter():
    """Test that setting the value of a non-existing parameter raises a ValueError."""

    def func(x: int = 42):
        return x

    pm = ParameterManager(input_function=func)
    with pytest.raises(ValueError):
        pm.set_parameter_value("y", 24)


def test_validate_function_parameters_with_type_not_annotated():
    """Test that the _validate_function_parameters method
    raises ValueError for parameters with no type annotations."""

    # Here we define a function with one parameter
    # that lacks a type annotation, i.e., 'y'.
    def func(x: int, y):
        return x + y

    with pytest.raises(ValueError) as excinfo:
        pm = ParameterManager(input_function=func)
        pm._validate_function_parameters()

    assert "Parameter 'y' has no type annotation." in str(excinfo.value)


def test_validate_function_parameters_with_type_not_registered():
    """Test that the _validate_function_parameters method raises
    ValueError when there's no Parameter subclass registered."""

    # Define a mock type that we know does not have an associated
    # Parameter subclass in the `parameter_type_catalogue`.
    class UnregisteredType:
        pass

    # Register a function with an annotated parameter of the `UnregisteredType`.
    def func(x: UnregisteredType):
        _ = x
        pass

    with pytest.raises(ValueError) as excinfo:
        pm = ParameterManager(input_function=func)
        pm._validate_function_parameters()
    print(str(excinfo.value))

    # Check that the error message includes the correct unrecognized type name.
    assert "No Parameter class registered for type" in str(excinfo.value)
    # assert "'unregisteredtype'" in str(excinfo.value)
