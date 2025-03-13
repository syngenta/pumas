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
    """
    Calculate the sigmoid bell function value for a given input.

    This function combines two sigmoid functions to create a bell-shaped curve.

    Args:
        x (Union[float, UFloat]): The input value.
        x1, x2, x3, x4 (float): Shape parameters defining the bell curve.
        k (float): Slope coefficient. Default is 1.0.
        base (float): Base of the exponential function. Default is 10.0.
        invert (bool): Whether to invert the result. Default is False.
        shift (float): Vertical shift of the curve. Default is 0.0.
        math_module (ModuleType): Math module to use. Default is math.

    Returns:
        Union[float, UFloat]: The calculated sigmoid bell value.

    Raises:
        InvalidBoundaryError: If shape parameters are invalid or base <= 1.
    """
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
    """
    Sigmoid Bell desirability function implementation.

    This class implements a sigmoid bell desirability function with adjustable parameters.
    It provides methods to compute the desirability for both numeric and uncertain float inputs.

    Mathematical Definition:
    The sigmoid bell function is defined as the difference of two sigmoid functions:

    .. math::

        D(x) = S_1(x) - S_2(x)

    where:

    .. math::

        S_1(x) = \\frac{1}{1 + base^{-k \\cdot (x - x_1)/(x_2 - x_1)}}

        S_2(x) = \\frac{1}{1 + base^{-k \\cdot (x - x_3)/(x_4 - x_3)}}

    If invert is True, the function becomes:

    .. math::

        D_{inverted}(x) = 1 - D(x)

    Finally, the shift is applied:

    .. math::

        D_{final}(x) = D(x) \\cdot (1 - shift) + shift

    Parameters:
        params (Optional[Dict[str, Any]]): Initial parameters for the sigmoid bell function.

    Attributes:
        x1, x2, x3, x4 (float): Shape parameters defining the bell curve.
        k (float): Slope coefficient, range [1.0, inf), default 1.0.
        base (float): Base of the exponential function, range (1.0, 10.0], default 10.0.
        invert (bool): Whether to invert the result, default False.
        shift (float): Vertical shift of the curve, range [0.0, 1.0], default 0.0.

    Raises:
        InvalidBoundaryError: If shape parameters are invalid or base <= 1.
        InvalidParameterTypeError: If any parameter is of an invalid type.
        ParameterValueNotSet: If any required parameter is not set.

    Usage Example:

    >>> from pumas.desirability import desirability_catalogue

    >>> desirability_class = desirability_catalogue.get("sigmoid_bell")

    >>> params = {"x1": 20.0, "x4": 80.0, "x2": 45.0, "x3": 60.0, "k": 1.0, "base": 10.0, "invert": False, "shift": 0.0}
    >>> desirability = desirability_class(params=params)
    >>> print(desirability.get_parameters_values())
    {'x1': 20.0, 'x2': 45.0, 'x3': 60.0, 'x4': 80.0, 'k': 1.0, 'base': 10.0, 'invert': False, 'shift': 0.0}

    >>> result = desirability.compute_numeric(x=50.0)
    >>> print(f"{result:.2f}")
    1.00

    >>> result = desirability(x=50.0) # Same as compute_numeric
    >>> print(f"{result:.2f}")
    1.00

    >>> from pumas.uncertainty.uncertainties_wrapper import ufloat
    >>> result = desirability.compute_ufloat(x=ufloat(50.0, 20.0))
    >>> print(result)
    0.9999999+/-0.0000018
    """  # noqa: E501

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the SigmoidBell desirability function.

        Args:
            params (Optional[Dict[str, Any]]):
                Initial parameters for the sigmoid bell function.
        """
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
        Compute the sigmoid bell desirability for a numeric input.

        Args:
            x (float): The input value.

        Returns:
            float: The computed desirability value.

        Raises:
            InvalidParameterTypeError: If the input is not a float.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_compute_input(x, float)
        self._check_parameters_values_none()
        parameters = self.get_parameters_values()
        return compute_numeric_sigmoid_bell(x, **parameters)  # type: ignore

    def compute_ufloat(self, x: UFloat) -> UFloat:
        """
        Compute the sigmoid bell desirability for an uncertain float input.

        Args:
            x (UFloat): The uncertain float input value.

        Returns:
            UFloat: The computed desirability value with uncertainty.

        Raises:
            InvalidParameterTypeError: If the input is not a UFloat.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_compute_input(x, UFloat)
        self._check_parameters_values_none()
        parameters = self.get_parameters_values()
        return compute_ufloat_sigmoid_bell(x, **parameters)  # type: ignore

    __call__ = compute_numeric
