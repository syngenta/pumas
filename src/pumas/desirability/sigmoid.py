import math
from functools import partial
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Union, cast

from pumas.architecture.exceptions import InvalidBoundaryError
from pumas.desirability.base_models import Desirability
from pumas.uncertainty.uncertainties_wrapper import UFloat, umath


def hard_sigmoid(x: Union[float, UFloat], k: float) -> Union[float, UFloat]:
    """
    Compute the hard sigmoid function.

    Args:
        x (Union[float, UFloat]): The input value.
        k (float): The slope parameter.

    Returns:
        Union[float, UFloat]: The result of the hard sigmoid function.
    """
    result = 1.0 if k * x > 0 else 0.0  # type: ignore
    return result


def stable_sigmoid(
    x: Union[float, UFloat], k: float, base: float, math_module: ModuleType = math
) -> Union[float, UFloat]:
    """
    Compute the stable sigmoid function.

    Args:
        x (Union[float, UFloat]): The input value.
        k (float): The slope parameter.
        base (float): The base of the exponential function.
        math_module (ModuleType, optional): The math module to use. Defaults to math.

    Returns:
        Union[float, UFloat]: The result of the stable sigmoid function.
    """
    if base == 10:
        h = k * x * math_module.log(10)  # type: ignore
    else:
        h = k * x * math_module.log(base)  # type: ignore

    if h >= 0:
        result = 1.0 / (1.0 + math_module.exp(-h))
    else:
        result = math_module.exp(h) / (1.0 + math_module.exp(h))
    return result  # type: ignore


def sigmoid(
    x: Union[float, UFloat],
    low: float,
    high: float,
    k: float,
    shift: float = 0.0,
    base: float = 10.0,
    math_module: ModuleType = math,
) -> Union[float, UFloat]:
    """
    Compute the sigmoid function with adjustable parameters.

    Args:
        x (Union[float, UFloat]): The input value.
        low (float): The lower bound of the sigmoid range.
        high (float): The upper bound of the sigmoid range.
        k (float): The slope parameter.
        shift (float, optional): The vertical shift of the sigmoid. Defaults to 0.0.
        base (float, optional): The base of the exponential function. Defaults to 10.0.
        math_module (ModuleType, optional): The math module to use.
            It uses math for numerical computations and umath
            for uncertain computations. Defaults to math.

    Returns:
        Union[float, UFloat]: The result of the sigmoid function.

    """
    # neet to implement in the parameter definition the le and ge conditions
    if base <= 1:
        raise InvalidBoundaryError("Base must be greater than 1")

    # neet to implement in the parameter definition the constraints between parameters
    if high < low:
        raise InvalidBoundaryError("High must be greater than or equal to low")

    x_centered = x - (high + low) / 2

    if (high - low) == 0:
        # Hard sigmoid case
        result = hard_sigmoid(x=x_centered, k=k)
    else:
        # Stable sigmoid case
        k_adjusted = 10.0 * k / (high - low)
        result = stable_sigmoid(
            x=x_centered, k=k_adjusted, base=base, math_module=math_module
        )

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


compute_numeric_sigmoid: Callable[[float, float, float, float, float, float], float] = (
    cast(
        Callable[[float, float, float, float, float, float], float],
        partial(sigmoid, math_module=math),
    )
)

compute_ufloat_sigmoid: Callable[
    [UFloat, float, float, float, float, float], UFloat
] = cast(
    Callable[[UFloat, float, float, float, float, float], UFloat],
    partial(sigmoid, math_module=umath),
)


class Sigmoid(Desirability):
    """

    Sigmoid desirability function implementation.

    This class implements a sigmoid desirability function with adjustable parameters.
    It provides methods to compute the desirability for both numeric and uncertain float inputs.

    Mathematical Definition:
    The sigmoid function is defined as:

    .. math::

        F(x) = \\frac{1}{1 + {base}^{-arg}} * (1 - {shift}) + {shift}

    Where the  `arg` term is calculated as:

    .. math::

        {arg} = \\frac{10 * k * (x - \\frac{{low} + {high}}{2})}{{high} - {low}}

    Implementation Details:

    The sigmoid function uses two different implementations based on the input parameters:

    1. Hard Sigmoid: When `high` equals `low`, a hard sigmoid function is used, which
       returns 1.0 if `k * x > 0`, and 0.0 otherwise.

    2. Stable Sigmoid: For all other cases, a numerically stable sigmoid implementation
       is used. This implementation avoids overflow errors for large positive or negative
       inputs by using different formulas based on the sign of the intermediate calculation.

    The choice between these implementations is made automatically based on the input
    parameters, ensuring accurate and stable results across a wide range of inputs.

    Parameters:
        params (Optional[Dict[str, Any]]): Initial parameters for the sigmoid function. Defaults to None.

    Attributes:
        low (float): The lower bound of the sigmoid range.
        high (float): The upper bound of the sigmoid range.
        k (float): The slope parameter, range [-1.0, 1.0], default 0.5.
        base (float): The base of the exponential function, range (1.0, 10.0], default 10.0.
        shift (float): The vertical shift of the sigmoid, range [0.0, 1.0], default 0.0.


    Raises:
        InvalidBoundaryError: If base is less than or equal to 1, or if high is less than low.
        InvalidParameterTypeError: If any parameter is of an invalid type.
        ParameterValueNotSet: If any required parameter is not set.


    Usage Example:

    >>> from pumas.desirability import desirability_catalogue

    >>> desirability_class = desirability_catalogue.get("sigmoid")

    >>> params = {'low': 0.0, 'high': 1.0, 'k': 0.1, 'base': 10.0, 'shift': 0.0}
    >>> desirability = desirability_class(params=params)
    >>> print(desirability.get_parameters_values())
    {'low': 0.0, 'high': 1.0, 'k': 0.1, 'base': 10.0, 'shift': 0.0}

    >>> result = desirability.compute_numeric(x=0.5)
    >>> print(f"{result:.2f}")
    0.50

    >>> result = desirability(x=0.5) # Same as compute_numeric
    >>> print(f"{result:.2f}")
    0.50

    >>> from pumas.uncertainty.uncertainties_wrapper import ufloat
    >>> result = desirability.compute_ufloat(x=ufloat(0.5, 0.1))
    >>> print(result)
    0.50+/-0.06
    """  # noqa: E501

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Sigmoid desirability function.

        Args:
            params (Optional[Dict[str, Any]]): Initial parameters for the sigmoid function.
        """  # noqa: E501
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
                "k": {"type": "float", "min": -1.0, "max": 1.0, "default": 0.5},
                "base": {"type": "float", "min": 1.0, "max": 10.0, "default": 10.0},
                "shift": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            }
        )
        self._validate_and_set_parameters(params)

    def compute_numeric(self, x: float) -> float:
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
        self._validate_compute_input(x, float)
        self._check_parameters_values_none()
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
        self._validate_compute_input(x, UFloat)
        self._check_parameters_values_none()
        parameters = self.get_parameters_values()
        return compute_ufloat_sigmoid(x=x, **parameters)  # type: ignore

    __call__ = compute_numeric
