from typing import Any, Dict, Optional

import pytest

from pumas.architecture.exceptions import (
    InvalidBoundaryError,
    InvalidParameterTypeError,
    ParameterSettingError,
    ParameterSettingWarning,
)
from pumas.architecture.parametrized_strategy import AbstractParametrizedStrategy
from pumas.uncertainty.uncertainties_wrapper import UFloat, ufloat

# Concrete classes for testing


class NoParameterOneInput(AbstractParametrizedStrategy):
    def __init__(self):
        super().__init__()
        self._set_parameter_definitions({})

    def compute_numeric(self, x: float) -> float:
        return x + 1

    def compute_ufloat(self, x: UFloat) -> UFloat:
        return x + 1


class OneParameterOneInput(AbstractParametrizedStrategy):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._set_parameter_definitions(
            {
                "a": {"type": "float", "min": 0.0, "max": 10.0, "default": 1.0},
            }
        )
        self._validate_and_set_parameters(params)

    def compute_numeric(self, x: float) -> float:
        self._validate_input(x, (int, float))  # Allow both int and float
        params = self.get_parameters_values()

        def func(q: float, a: float) -> float:
            return a * q

        return func(x, **params)

    def compute_ufloat(self, x: UFloat) -> UFloat:
        self._validate_input(x, UFloat)
        params = self.get_parameters_values()

        def func(q: UFloat, a: float) -> UFloat:
            return a * q  # type: ignore

        return func(x, **params)


class ManyParameterOneInput(AbstractParametrizedStrategy):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._set_parameter_definitions(
            {
                "a": {"type": "float", "min": 0.0, "max": 10.0, "default": 1.0},
                "b": {"type": "float", "min": -5.0, "max": 5.0, "default": 0.0},
                "c": {"type": "float", "min": 0.0, "max": 2.0, "default": 1.0},
            }
        )
        self._validate_and_set_parameters(params)

    def compute_numeric(self, x: float) -> float:
        self._validate_input(x, (int, float))  # Allow both int and float
        params = self.get_parameters_values()

        def func(q: float, a: float, b: float, c: float) -> float:
            return a * q + b + c

        return func(x, **params)

    def compute_ufloat(self, x: UFloat) -> UFloat:
        self._validate_input(x, UFloat)
        params = self.get_parameters_values()

        def func(q: UFloat, a: float, b: float, c: float) -> UFloat:
            return a * q + b + c  # type: ignore

        return func(x, **params)


class OneParameterTwoInputs(AbstractParametrizedStrategy):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._set_parameter_definitions(
            {
                "weight": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            }
        )
        self._validate_and_set_parameters(params)

    def compute_numeric(self, x: float, y: float) -> float:
        self._validate_input(x, (int, float))  # Allow both int and float
        params = self.get_parameters_values()

        def func(q: float, w: float, weight: float) -> float:
            return w * q + weight

        return func(q=x, w=y, **params)

    def compute_ufloat(self, x: UFloat, y: UFloat) -> UFloat:
        self._validate_input(x, UFloat)
        params = self.get_parameters_values()

        def func(q: UFloat, w: UFloat, weight: float) -> UFloat:
            return w * q + weight

        return func(q=x, w=y, **params)

    # Test fixtures


@pytest.fixture
def no_param_one_input():
    return NoParameterOneInput()


@pytest.fixture
def one_param_one_input():
    return OneParameterOneInput()


@pytest.fixture
def many_param_one_input():
    return ManyParameterOneInput()


@pytest.fixture
def one_param_two_inputs():
    return OneParameterTwoInputs()


# Test cases


# No parameters one input
def test_no_param_one_input_initialization(no_param_one_input):
    assert no_param_one_input.parameter_manager is not None
    assert len(no_param_one_input.parameter_manager.parameters_map) == 0
    assert no_param_one_input.get_parameters_values() == {}


def test_no_param_one_input_compute_numeric(no_param_one_input):
    assert no_param_one_input.compute_numeric(2.0) == 3.0


def test_no_param_one_input_compute_ufloat(no_param_one_input):
    result = no_param_one_input.compute_ufloat(ufloat(2.0, 0.1))
    assert result.nominal_value == 3.0
    assert result.std_dev == 0.1


def test_no_param_one_input_set_parameters_warning(no_param_one_input):
    with pytest.warns(UserWarning, match="This strategy does not accept parameters"):
        no_param_one_input.set_parameters_values({"a": 1.0})


def test_no_param_one_input_get_parameters(no_param_one_input):
    assert no_param_one_input.get_parameters_values() == {}


def test_no_param_one_input_with_invalid_input_type(no_param_one_input):
    with pytest.raises(TypeError):
        no_param_one_input.compute_numeric("not a number")


def test_no_parameter_strategy_set_values(no_param_one_input):
    """Test that setting values on a no-parameter strategy raises a warning."""
    with pytest.warns(
        ParameterSettingWarning, match="This strategy does not accept parameters"
    ):
        no_param_one_input.set_parameters_values({"test": 10})


def test_no_parameter_strategy_set_attributes(no_param_one_input):
    with pytest.warns(
        ParameterSettingWarning, match="This strategy does not accept parameters"
    ):
        no_param_one_input.set_coefficient_parameters_attributes({"test": {"min": 0}})


# one parameter one input
def test_one_param_one_input_initialization(one_param_one_input):
    assert one_param_one_input.parameter_manager is not None
    assert len(one_param_one_input.parameter_manager.parameters_map) == 1


def test_one_param_one_input_without_setting_parameters(
    one_param_one_input,
):
    """The test does not fail if the parameters are not when initializing
    the function as long as there are defaults"""

    # assert that each parameter required has a value (coming from the default)
    print(one_param_one_input.get_parameters_values())
    result = one_param_one_input.compute_numeric(0.5)
    assert result == 0.5


def test_one_param_one_input_with_invalid_input_type(one_param_one_input):
    one_param_one_input.set_parameters_values({"a": 1.0})
    with pytest.raises(InvalidParameterTypeError):
        one_param_one_input.compute_numeric("not_a_number")


def test_one_param_one_input_set_valid_parameters(one_param_one_input):
    valid_params = {"a": 5.0}
    one_param_one_input.set_parameters_values(valid_params)
    assert one_param_one_input.get_parameters_values() == valid_params


def test_one_param_one_input_set_invalid_parameter_type(one_param_one_input):
    invalid_params = {"a": "not_a_float"}
    with pytest.raises(InvalidParameterTypeError):
        one_param_one_input.set_parameters_values(invalid_params)


def test_one_param_one_input_set_out_of_bounds_parameter(one_param_one_input):
    invalid_params = {"a": 11.0}  # a should be between 0.0 and 10.0
    with pytest.raises(InvalidBoundaryError):
        one_param_one_input.set_parameters_values(invalid_params)


def test_one_param_one_input_set_unrecognized_parameter(one_param_one_input):
    invalid_params = {"unknown_param": 1.0}
    with pytest.raises(ParameterSettingError):
        one_param_one_input.set_parameters_values(invalid_params)


def test_one_param_one_input_compute_numeric(one_param_one_input):
    one_param_one_input.set_parameters_values({"a": 2.0})
    assert one_param_one_input.compute_numeric(3.0) == 6.0


def test_one_param_one_input_parameters_after_setting():
    one_param_one_input = OneParameterOneInput()
    one_param_one_input.set_parameters_values({"a": 2.0})
    assert one_param_one_input.get_parameters_values() == {"a": 2.0}


def test_one_param_one_input_parameters_after_initialization():
    one_param_one_input = OneParameterOneInput(params={"a": 2.0})
    assert one_param_one_input.get_parameters_values() == {"a": 2.0}


# one parameter two inputs
def test_one_param_two_inputs_initialization(one_param_two_inputs):
    assert one_param_two_inputs.parameter_manager is not None
    assert len(one_param_two_inputs.parameter_manager.parameters_map) == 1


def test_one_param_two_inputs_compute_numeric(one_param_two_inputs):
    one_param_two_inputs.set_parameters_values({"weight": 0.3})
    assert one_param_two_inputs.compute_numeric(1.0, 2.0) == 2.3


def test_one_param_two_inputs_compute_numeric_invalid_input(one_param_two_inputs):
    one_param_two_inputs.set_parameters_values({"weight": 0.3})
    with pytest.raises(TypeError):
        one_param_two_inputs.compute_numeric(1.0)  # Missing second input


# many parameters one input
def test_many_param_one_input_initialization(many_param_one_input):
    assert many_param_one_input.parameter_manager is not None
    assert len(many_param_one_input.parameter_manager.parameters_map) == 3


def test_many_param_one_input_compute_numeric(many_param_one_input):
    many_param_one_input.set_parameters_values({"a": 2.0, "b": 1.0, "c": 0.5})
    assert many_param_one_input.compute_numeric(3.0) == 7.5


def test_many_param_one_input_set_parameter_attributes(many_param_one_input):
    attributes = {"a": {"min": 1.0, "max": 5.0}}
    many_param_one_input.set_coefficient_parameters_attributes(attributes)
    assert many_param_one_input.parameter_manager.parameters_map["a"].min == 1.0
    assert many_param_one_input.parameter_manager.parameters_map["a"].max == 5.0


def test_many_param_one_input_set_invalid_parameter_attributes(many_param_one_input):
    invalid_attributes = {"unknown_param": {"min": 0.0}}
    with pytest.raises(ParameterSettingError):
        many_param_one_input.set_coefficient_parameters_attributes(invalid_attributes)
