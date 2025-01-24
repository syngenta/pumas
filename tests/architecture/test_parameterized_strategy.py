import pytest

from pumas.architecture.exceptions import (
    InvalidBoundaryError,
    InvalidParameterTypeError,
    ParameterDefinitionError,
    ParameterOverlapError,
    ParameterSettingError,
    ParameterSettingWarning,
    ParameterValueNotSet,
)
from pumas.architecture.parameters import FloatParameter
from pumas.architecture.parametrized_strategy import AbstractParametrizedStrategy


# Create fixtures for reusable desirability function instances
@pytest.fixture
def wall_function():
    """Returns an instance of ConcreteParameterizedStrategy
    using a linear utility function."""

    class ConcreteParameterizedStrategy(AbstractParametrizedStrategy):
        def __init__(self):
            super().__init__(
                coefficient_parameters_names=["w1", "w2"],
                input_parameters_names=["x"],
                utility_function=self.wall,
            )

        @staticmethod
        def wall(x: float, w1: float, w2: float) -> int:
            if w1 <= x <= w2:
                return 1
            else:
                return 0

        def compute(self, x: float):
            results = self._get_partial_utility_function(x=x)
            return results

    return ConcreteParameterizedStrategy()


@pytest.mark.parametrize(
    "coefficients,x,expected_output",
    [
        ({"w1": 0.0, "w2": 1.0}, 0.5, 1),
        ({"w1": 0.0, "w2": 1.0}, -0.5, 0),
        ({"w1": 0.0, "w2": 1.0}, 1.5, 0),
    ],
)
def test_wall_function_with_various_inputs(
    wall_function, coefficients, x, expected_output
):
    """Parametrized test for the wall function at various points."""
    wall_function.set_coefficient_parameters_values(coefficients)
    result = wall_function.compute(x)
    assert result == expected_output


def test_linear_function_parameters_set_correctly(wall_function):
    """Should successfully update coefficients with valid values for linear function."""
    coefficients = {"w1": 2.0, "w2": 5.0}
    wall_function.set_coefficient_parameters_values(values_dict=coefficients)
    assert wall_function.get_coefficient_parameters_values() == coefficients

    assert wall_function.input_parameters_names == ["x"]
    assert wall_function.coefficient_parameters_names == ["w1", "w2"]
    assert wall_function.parameters_map.keys() == {"x", "w1", "w2"}
    assert isinstance(wall_function.parameters_map["x"], FloatParameter)
    assert isinstance(wall_function.parameters_map["w1"], FloatParameter)
    assert isinstance(wall_function.parameters_map["w1"], FloatParameter)
    assert wall_function.coefficient_parameters_map.keys() == {"w1", "w2"}
    assert wall_function.input_parameters_map.keys() == {"x"}


def test_set_coefficient_parameters_values_partially_raises_warning(
    wall_function,
):
    """Attempting to compute without setting all coefficients should raise an error."""
    assert wall_function.get_coefficient_parameters_values() == {
        "w1": None,
        "w2": None,
    }
    coefficients = {"w1": 2.0}
    # it raises a warning because we are not setting all parameters at once
    # but does not stop the program
    with pytest.warns(ParameterSettingWarning):
        wall_function.set_coefficient_parameters_values(coefficients)
    assert wall_function.get_coefficient_parameters_values() == {
        "w1": 2.0,
        "w2": None,
    }

    # the error on computing is raised because w2 was not set and it is still None
    with pytest.raises(ParameterValueNotSet):
        wall_function.compute(x=1.0)

    coefficients = {"w2": 4.0}
    # it raises a warning because we are not setting all parameters at once
    # but does not stop the program
    with pytest.warns(ParameterSettingWarning):
        wall_function.set_coefficient_parameters_values(coefficients)
    # all coefficients are now set and computation can be done
    assert wall_function.get_coefficient_parameters_values() == {
        "w1": 2.0,
        "w2": 4.0,
    }

    result = wall_function.compute(x=1.0)
    assert result == 0


def test_set_coefficient_parameters_attributes_partially_raises_warning(
    wall_function,
):
    assert wall_function.get_coefficient_parameters_values() == {
        "w1": None,
        "w2": None,
    }
    attributes_map = {"w1": {"min": 0.0, "max": 10.0, "default": 2.0}}
    # it raises a warning because we are not setting all parameters at once
    # but does not stop the program
    with pytest.warns(ParameterSettingWarning):
        wall_function.set_coefficient_parameters_attributes(attributes_map)
    assert wall_function.get_coefficient_parameters_values() == {
        "w1": 2.0,
        "w2": None,
    }


def test_compute_without_all_coefficients_raises_error(wall_function):
    """Attempting to compute without having all coefficient
    set should raise an error."""
    assert wall_function.get_coefficient_parameters_values() == {
        "w1": None,
        "w2": None,
    }

    # the error is raised because no parameter was set and they are still None
    with pytest.raises(ParameterValueNotSet):
        wall_function.compute(x=1.0)


def test_set_unnecessary_coefficient_parameter_value_raises_warning(
    wall_function,
):
    """Setting coefficients with invalid names should raise warning."""
    coefficients = {"invalid_param": 4.0}
    with pytest.raises(ParameterSettingError):
        wall_function.set_coefficient_parameters_values(coefficients)


def test_set_unnecessary_coefficient_parameter_attribute_raises_warning(
    wall_function,
):
    """Setting coefficients with invalid names should raise warning."""
    coefficients = {"invalid_param": {"something": 4.0}}
    with pytest.raises(ParameterSettingError):
        wall_function.set_coefficient_parameters_attributes(coefficients)


def test_compute_with_invalid_input_type_raises_error(wall_function):
    """Passing an input value with incorrect type
    should raise an InvalidParameterTypeError."""
    coefficients = {"w1": 2.0, "w2": "5.0"}
    with pytest.raises(InvalidParameterTypeError):
        wall_function.set_coefficient_parameters_values(coefficients)


def test_compute_method(wall_function):
    """Test that the wall method returns correct values."""
    wall_function.set_coefficient_parameters_values({"w1": 2.0, "w2": 3.0})
    assert wall_function.compute(x=1.0) == 0
    assert wall_function.compute(x=2.5) == 1
    assert wall_function.compute(x=3.5) == 0


# Tests for parameter validation
def test_overlapping_parameter_names_raises_error():
    with pytest.raises(ParameterOverlapError):

        class Desirability(AbstractParametrizedStrategy):
            def __init__(self):
                super().__init__(
                    coefficient_parameters_names=["w1", "w2"],
                    input_parameters_names=["w1"],
                    utility_function=self.wall,
                )

            @staticmethod
            def wall(x: float, w1: float, w2: float):
                if w1 <= x <= w2:
                    return 1
                else:
                    return 0

            def compute(self, x: float):
                results = self._get_partial_utility_function(x=x)
                return results

        return Desirability()


def test_missing_parameter_in_utility_function_raises_error():
    with pytest.raises(ParameterDefinitionError):

        class Desirability(AbstractParametrizedStrategy):
            def __init__(self):
                super().__init__(
                    coefficient_parameters_names=["w1"],
                    input_parameters_names=["x"],
                    utility_function=self.wall,
                )

            @staticmethod
            def wall(x: float, w1: float, w2: float):
                if w1 <= x <= w2:
                    return 1
                else:
                    return 0

            def compute(self, x: float):
                results = self._get_partial_utility_function(x=x)
                return results

        return Desirability()


def test_additional_parameter_in_utility_function_raises_error():
    with pytest.raises(ParameterDefinitionError):

        class Desirability(AbstractParametrizedStrategy):
            def __init__(self):
                super().__init__(
                    coefficient_parameters_names=["w1", "w2", "error_param"],
                    input_parameters_names=["x"],
                    utility_function=self.wall,
                )

            @staticmethod
            def wall(x: float, w1: float, w2: float):
                if w1 <= x <= w2:
                    return 1
                else:
                    return 0

            def compute(self, x: float):
                results = self._get_partial_utility_function(x=x)
                return results

        return Desirability()


# Tests for setting parameter values
def test_set_invalid_coefficient_parameter_value_raises_error(
    wall_function,
):
    invalid_value = {"w1": "not_a_float_value"}
    with pytest.raises(InvalidParameterTypeError):
        wall_function.set_coefficient_parameters_values(invalid_value)


def test_set_unrecognized_coefficient_parameter_raises_error(
    wall_function,
):
    unrecognized_params = {"unknown_param": 1.0}
    with pytest.raises(ParameterSettingError):
        wall_function.set_coefficient_parameters_values(unrecognized_params)


def test_set_coefficient_parameter_value_boundary_check(wall_function):
    # Assuming there is a validation boundary for the coefficients w1 and w2,
    wall_function._parameter_manager.set_parameter_attributes(
        name="w1", attributes={"min": 0.0, "max": 10.0}
    )
    out_of_bounds_value = {"w1": -1.0}
    with pytest.raises(InvalidBoundaryError):
        wall_function.set_coefficient_parameters_values(out_of_bounds_value)


# Tests for computation
def test_compute_with_all_coefficients_set(wall_function):
    coefficients = {"w1": 2.0, "w2": 5.0}
    wall_function.set_coefficient_parameters_values(coefficients)
    assert wall_function.compute(x=3.0) == 1
    assert wall_function.compute(x=6.0) == 0
