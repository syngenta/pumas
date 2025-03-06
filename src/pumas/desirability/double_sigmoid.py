import math
from functools import partial
from types import ModuleType
from typing import Any, Dict, Optional, Union

from pumas.architecture.exceptions import InvalidBoundaryError
from pumas.desirability.base_models import Desirability
from pumas.desirability.sigmoid import hard_sigmoid, stable_sigmoid
from pumas.uncertainty.uncertainties_wrapper import UFloat, umath


def double_sigmoid(
    x: float,
    low: float,
    high: float,
    coef_div: float,
    coef_si: float,
    coef_se: float,
    base: float = 10.0,
    invert: bool = False,
    shift: float = 0.0,
    math_module: ModuleType = math,
) -> Union[float, UFloat]:
    """Utility function implementing a numerically stable double sigmoid desirability"""
    # neet to implement in the parameter definition the le and ge conditions
    if base <= 1:
        raise InvalidBoundaryError("Base must be greater than 1")

    # neet to implement in the parameter definition the constraints between parameters
    if high < low:
        raise InvalidBoundaryError("High must be greater than or equal to low")
    x_center = (high - low) / 2 + low

    if x < x_center:
        xl = x - low
        if coef_div == 0:
            result = hard_sigmoid(x=xl, k=coef_si)
        else:
            k_left_adjusted = coef_si / coef_div
            result = stable_sigmoid(
                x=xl, k=k_left_adjusted, base=base, math_module=math_module
            )
    else:
        xr = x - high
        if coef_div == 0:
            result = 1 - hard_sigmoid(x=xr, k=coef_se)  # type: ignore
        else:
            k_right_adjusted = coef_se / coef_div
            result = 1 - stable_sigmoid(  # type: ignore
                x=xr, k=k_right_adjusted, base=base, math_module=math_module
            )

    # invert if needed
    if invert:
        result = 1 - result  # type: ignore

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


compute_numeric_sigmoid = partial(double_sigmoid, math_module=math)

compute_ufloat_sigmoid = partial(double_sigmoid, math_module=umath)


class DoubleSigmoid(Desirability):
    """
    The class DoubleSigmoidDesirability implements a double sigmoid desirability function.

    This class wraps the `double_sigmoid` utility function and implements the interface to use it, identifying:
    * input parameters: 'x'
    * coefficient parameters: 'low', 'high', 'coef_div', 'coef_si', 'coef_se', 'base', and 'shift'.

    The double sigmoid function combines two sigmoid functions to create a plateau of high desirability
    between x_left and x_right, with smooth transitions on both sides.

    Usage Example:

    >>> from pumas.desirability import desirability_catalogue

    >>> desirability_class = desirability_catalogue.get("double_sigmoid")

    >>> params = {'low': 3.0, 'high': 7.0, 'coef_div': 1.0, 'coef_si': 2.0, 'coef_se': 2.0, 'base': 10.0, 'invert': False, 'shift': 0.0}
    >>> desirability = desirability_class(params=params)
    >>> print(desirability.get_parameters_values())
    {'low': 3.0, 'high': 7.0, 'coef_div': 1.0, 'coef_si': 2.0, 'coef_se': 2.0, 'base': 10.0, 'invert': False, 'shift': 0.0}
    """  # noqa E501

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
                "coef_div": {"type": "float", "min": 0.0, "max": None, "default": 1.0},
                "coef_si": {
                    "type": "float",
                    "min": 0.0,
                    "max": float("inf"),
                    "default": 1.0,
                },
                "coef_se": {
                    "type": "float",
                    "min": 0.0,
                    "max": float("inf"),
                    "default": 1.0,
                },
                "base": {"type": "float", "min": 1.0, "max": None, "default": 10.0},
                "invert": {"type": "bool", "default": False},
                "shift": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            }
        )
        self._validate_and_set_parameters(params)

    def compute_numeric(self, x: float) -> float:
        """
        Compute the sigmoid desirability for a numeric input.

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
        return compute_numeric_sigmoid(x=x, **parameters)  # type: ignore

    def compute_ufloat(self, x: UFloat) -> UFloat:
        """
        Compute the sigmoid desirability for an uncertain float input.

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
        return compute_ufloat_sigmoid(x=x, **parameters)  # type: ignore

    __call__ = compute_numeric
