from typing import Any, Dict, Optional

from pumas.desirability.base_models import Desirability
from pumas.uncertainty.uncertainties_wrapper import UFloat, ufloat


def compute_numeric_right_step(
    x: float, low: float, high: float, shift: float = 0.0
) -> float:
    _ = low
    result = 1.0 if x >= high else 0.0

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


def compute_ufloat_right_step(
    x: UFloat, low: float, high: float, shift: float = 0.0
) -> UFloat:
    _ = low
    x_nominal_value, x_std_dev = x.nominal_value, x.std_dev  # type: ignore
    result = (
        ufloat(nominal_value=1.0, std_dev=x_std_dev)
        if x_nominal_value >= high
        else ufloat(nominal_value=0.0, std_dev=x_std_dev)
    )

    # Apply the shift
    result = result * (1 - shift) + shift

    return result  # type: ignore


def compute_numeric_left_step(
    x: float, low: float, high: float, shift: float = 0.0
) -> float:
    _ = high
    result = 1.0 if x <= low else 0.0

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


def compute_ufloat_left_step(
    x: UFloat, low: float, high: float, shift: float = 0.0
) -> UFloat:
    _ = high
    x_nominal_value, x_std_dev = x.nominal_value, x.std_dev  # type: ignore
    result = (
        ufloat(nominal_value=1.0, std_dev=x_std_dev)
        if x_nominal_value <= low
        else ufloat(nominal_value=0.0, std_dev=x_std_dev)
    )

    # Apply the shift
    result = result * (1 - shift) + shift

    return result  # type: ignore


def compute_numeric_step(
    x: float, low: float, high: float, shift: float = 0.0
) -> float:
    result = 1.0 if low <= x <= high else 0.0

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


def compute_ufloat_step(
    x: UFloat, low: float, high: float, shift: float = 0.0
) -> UFloat:
    x_nominal_value, x_std_dev = x.nominal_value, x.std_dev  # type: ignore
    result = (
        ufloat(nominal_value=1.0, std_dev=x_std_dev)
        if low <= x_nominal_value <= high
        else ufloat(nominal_value=0.0, std_dev=x_std_dev)
    )

    # Apply the shift
    result = result * (1 - shift) + shift

    return result  # type: ignore


class RightStep(Desirability):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._set_parameter_definitions(
            {
                "low": {
                    "type": "float",
                    "min": float("-inf"),
                    "max": float("inf"),
                    "default": None,
                },
                "high": {
                    "type": "float",
                    "min": float("-inf"),
                    "max": float("inf"),
                    "default": None,
                },
                "shift": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            }
        )
        self._validate_and_set_parameters(params)

    def compute_numeric(self, x: float) -> float:
        """
        Compute the right step desirability for a numeric input.

        Args:
            x (float): The input value.

        Returns:
            float: The computed desirability value.

        Raises:
            InvalidParameterTypeError: If the input is not a float.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_input(x, float)
        self._check_coefficient_parameters_values()
        parameters = self.get_parameters_values()
        return compute_numeric_right_step(x=x, **parameters)

    def compute_ufloat(self, x: UFloat) -> UFloat:
        """
        Compute the right step desirability for an uncertain float input.

        Args:
            x (UFloat): The uncertain float input value.

        Returns:
            UFloat: The computed desirability value with uncertainty.

        Raises:
            InvalidParameterTypeError: If the input is not a UFloat.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_input(x, UFloat)
        self._check_coefficient_parameters_values()
        parameters = self.get_parameters_values()
        return compute_ufloat_right_step(x=x, **parameters)


class LeftStep(Desirability):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._set_parameter_definitions(
            {
                "low": {
                    "type": "float",
                    "min": float("-inf"),
                    "max": float("inf"),
                    "default": None,
                },
                "high": {
                    "type": "float",
                    "min": float("-inf"),
                    "max": float("inf"),
                    "default": None,
                },
                "shift": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            }
        )
        self._validate_and_set_parameters(params)

    def compute_numeric(self, x: float) -> float:
        """
        Compute the left step desirability for a numeric input.

        Args:
            x (float): The input value.

        Returns:
            float: The computed desirability value.

        Raises:
            InvalidParameterTypeError: If the input is not a float.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_input(x, float)
        self._check_coefficient_parameters_values()
        parameters = self.get_parameters_values()
        return compute_numeric_left_step(x=x, **parameters)

    def compute_ufloat(self, x: UFloat) -> UFloat:
        """
        Compute the left step desirability for an uncertain float input.

        Args:
            x (UFloat): The uncertain float input value.

        Returns:
            UFloat: The computed desirability value with uncertainty.

        Raises:
            InvalidParameterTypeError: If the input is not a UFloat.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_input(x, UFloat)
        self._check_coefficient_parameters_values()
        parameters = self.get_parameters_values()
        return compute_ufloat_left_step(x=x, **parameters)


class Step(Desirability):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._set_parameter_definitions(
            {
                "low": {
                    "type": "float",
                    "min": float("-inf"),
                    "max": float("inf"),
                    "default": None,
                },
                "high": {
                    "type": "float",
                    "min": float("-inf"),
                    "max": float("inf"),
                    "default": None,
                },
                "shift": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            }
        )
        self._validate_and_set_parameters(params)

    def compute_numeric(self, x: float) -> float:
        """
        Compute the step desirability for a numeric input.

        Args:
            x (float): The input value.

        Returns:
            float: The computed desirability value.

        Raises:
            InvalidParameterTypeError: If the input is not a float.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_input(x, float)
        self._check_coefficient_parameters_values()
        parameters = self.get_parameters_values()
        return compute_numeric_step(x=x, **parameters)

    def compute_ufloat(self, x: UFloat) -> UFloat:
        """
        Compute the step desirability for an uncertain float input.

        Args:
            x (UFloat): The uncertain float input value.

        Returns:
            UFloat: The computed desirability value with uncertainty.

        Raises:
            InvalidParameterTypeError: If the input is not a UFloat.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_input(x, UFloat)
        self._check_coefficient_parameters_values()
        parameters = self.get_parameters_values()
        return compute_ufloat_step(x=x, **parameters)
