import math
import sys
from functools import partial
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Union, cast

from pumas.desirability.base_models import Desirability
from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import (
    UFloat,
    umath,
)


def exponential_decay(
    x: Union[float, UFloat],
    k: float,
    shift: float = 0.0,
    math_module: ModuleType = math,
) -> Union[float, UFloat]:
    """
    Exponential decay function: exp(-k*x) for x >= 0, clamped to 1.0 for x < 0.

    Args:
        x (Union[float, UFloat]): The input value.
        k (float): The decay rate parameter.
        shift (float, optional): The vertical shift. Defaults to 0.0.
        math_module (ModuleType, optional): The math module to use.
            It uses math for numerical computations and umath
            for uncertain computations. Defaults to math.

    Returns:
        Union[float, UFloat]: The result of the exponential decay function.
    """
    if x < 0:  # type: ignore
        result = 1.0
    else:
        result = math_module.exp(-k * x)  # type: ignore

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


compute_numeric_exponential_decay: Callable[[float, float, float], float] = cast(
    Callable[[float, float, float], float],
    partial(exponential_decay, math_module=math),
)

compute_ufloat_exponential_decay: Callable[[UFloat, float, float], UFloat] = cast(
    Callable[[UFloat, float, float], UFloat],
    partial(exponential_decay, math_module=umath),
)


class ExponentialDecay(Desirability):
    name = "exponential_decay"
    """
    Exponential decay desirability function implementation.

    Mathematical Definition:

    The exponential decay function is defined as:

    .. math::

        f(x) = \\begin{cases}
        1.0 & \\text{if } x < 0 \\\\
        e^{-k \\cdot x} & \\text{if } x \\geq 0
        \\end{cases}

    With optional shift transformation:

    .. math::

        f_{final}(x) = f(x) \\cdot (1 - shift) + shift

    Where:
        * `x` is the input value.
        * `k` is the decay rate parameter (k > 0). Higher values result
          in faster decay.
        * `shift` is the vertical shift applied to the entire curve,
          ranging from 0 (no shift) to 1 (maximum shift).

    For x < 0, the function is "rectified" or "clamped" to 1.0, meaning
    full desirability.

    Parameters:
        params (Optional[Dict[str, Any]]): Initial parameters for the
            exponential decay function. Defaults to None.

    Attributes:
        k (float): The decay rate parameter (k > 0). Higher values result
            in faster decay.
        shift (float): The vertical shift applied to the entire curve,
            ranging from 0 (no shift) to 1 (maximum shift).

    Usage Example:

    >>> from pumas.desirability import desirability_catalogue

    >>> desirability_class = desirability_catalogue.get("exponential_decay")

    >>> params = {'k': 1.0, 'shift': 0.0}
    >>> desirability = desirability_class(params=params)
    >>> print(desirability.get_parameters_values())
    {'k': 1.0, 'shift': 0.0}

    >>> result = desirability.compute_numeric(x=0.0)
    >>> print(f"{result:.2f}")
    1.00

    >>> result = desirability.compute_numeric(x=1.0)
    >>> print(f"{result:.2f}")
    0.37

    >>> result = desirability(x=-1.0)  # Clamped to 1.0
    >>> print(f"{result:.2f}")
    1.00

    >>> from uncertainties import ufloat
    >>> result = desirability.compute_ufloat(x=ufloat(1.0, 0.1))
    >>> print(result)
    0.37+/-0.04
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the ExponentialDecay desirability function.

        Args:
            params (Optional[Dict[str, Any]]): Initial parameters for the
                exponential decay function.
        """
        super().__init__()
        self._set_parameter_definitions(
            {
                "k": {
                    "type": "float",
                    "min": sys.float_info.epsilon,
                    "max": float("inf"),
                    "default": 1.0,
                },
                "shift": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.0,
                },
            }
        )
        self._validate_and_set_parameters(params)

    def compute_numeric(self, x: Union[int, float]) -> float:
        """
        Compute the exponential decay desirability for a numeric input.

        Args:
            x (Union[int, float]): The numeric input value.

        Returns:
            float: The computed desirability value.

        Raises:
            InvalidParameterTypeError: If the input is not a float.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_compute_input(item=x, expected_type=(int, float))
        self._check_parameters_values_none()
        parameters = self.get_parameters_values()
        return compute_numeric_exponential_decay(x=x, **parameters)  # type: ignore

    def compute_ufloat(self, x: UFloat) -> UFloat:
        """
        Compute the exponential decay desirability for an uncertain float
        input.

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
        return compute_ufloat_exponential_decay(x=x, **parameters)  # type: ignore

    __call__ = compute_numeric
