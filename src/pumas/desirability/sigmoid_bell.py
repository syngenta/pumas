import math
from functools import partial
from types import ModuleType
from typing import Any, Dict, Optional, Union

from pumas.architecture.exceptions import InvalidBoundaryError
from pumas.desirability.base_models import Desirability
from pumas.desirability.sigmoid import sigmoid
from pumas.uncertainty.uncertainties_wrapper import UFloat, umath


def sigmoid_bell(
    x: Union[float, UFloat],
    x1: float,
    x2: float,
    x3: float,
    x4: float,
    k: float = 1.0,
    base: float = 10.0,
    invert: bool = False,
    shift: float = 0.0,
    math_module: ModuleType = math,
) -> Union[float, UFloat]:
    if x3 < x1 or x2 > x4 or x2 < x1 or x4 < x3:
        raise InvalidBoundaryError(
            "Invalid shape parameters. Ensure x1 < x2 < x3 < x4."
        )
    if base <= 1:
        raise InvalidBoundaryError("'base' must be greater than 1")

    # Combine two sigmoid functions
    sig1 = sigmoid(
        x, low=x1, high=x2, k=k, shift=0.0, base=base, math_module=math_module
    )
    sig2 = sigmoid(
        x, low=x3, high=x4, k=k, shift=0.0, base=base, math_module=math_module
    )

    # Calculate the result
    result = sig1 - sig2  # type: ignore

    # invert if needed
    if invert:
        result = 1 - result  # type: ignore

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


compute_numeric_sigmoid_bell = partial(sigmoid_bell, math_module=math)
compute_ufloat_sigmoid_bell = partial(sigmoid_bell, math_module=umath)


class SigmoidBell(Desirability):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._set_parameter_definitions(
            {
                "x1": {
                    "type": "float",
                    "min": float("-inf"),
                    "max": float("inf"),
                    "default": None,
                },
                "x2": {
                    "type": "float",
                    "min": float("-inf"),
                    "max": float("inf"),
                    "default": None,
                },
                "x3": {
                    "type": "float",
                    "min": float("-inf"),
                    "max": float("inf"),
                    "default": None,
                },
                "x4": {
                    "type": "float",
                    "min": float("-inf"),
                    "max": float("inf"),
                    "default": None,
                },
                "k": {"type": "float", "min": 1.0, "max": None, "default": 1.0},
                "base": {"type": "float", "min": 1.0, "max": 10.0, "default": 10.0},
                "invert": {"type": "bool", "default": False},
                "shift": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            }
        )
        self._validate_and_set_parameters(params)

    def compute_numeric(self, x: float) -> float:
        """
        Compute the double sigmoid desirability for a numeric input.

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
        return compute_numeric_sigmoid_bell(x, **parameters)  # type: ignore

    def compute_ufloat(self, x: UFloat) -> UFloat:
        """
        Compute the double sigmoid desirability for an uncertain float input.

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
        return compute_ufloat_sigmoid_bell(x, **parameters)  # type: ignore

    __call__ = compute_numeric
