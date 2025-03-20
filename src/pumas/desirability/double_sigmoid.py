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
    """
    Compute the double sigmoid function with adjustable parameters.
    Args:
        x (float): The input value.
        low (float): The lower bound of the sigmoid range.
        high (float): The upper bound of the sigmoid range.
        coef_div (float): The divisor coefficient for slope adjustment.
        coef_si (float): The slope coefficient for the increasing part.
        coef_se (float): The slope coefficient for the decreasing part.
        base (float, opional): The base of the exponential function. Defaults to 10.0.
        invert (bool, optional): Whether to invert the result. Defaults to False.
        shift (float, optional): The vertical shift of the sigmoid. Defaults to 0.0.
        math_module (ModuleType, optional): The math module to use. Defaults to math.

    Returns:
        Union[float, UFloat]: The result of the double sigmoid function.
    """

    #  need to implement in the parameter definition the le and ge conditions
    if base <= 1:
        raise InvalidBoundaryError("Base must be greater than 1")

    #  need to implement in the parameter definition the constraints between parameters   # noqa: E501
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
    Double Sigmoid desirability function implementation.

    This class implements a double sigmoid desirability function with adjustable parameters.
    It provides methods to compute the desirability for both numeric and uncertain float inputs.

    The double sigmoid function combines two sigmoid functions to create a plateau of high
    desirability between low and high, with smooth transitions on both sides.

    Mathematical Definition:

    The double sigmoid function is defined as:

    .. math::

        F(x) = \\begin{cases}
            S_1(x) & \\text{if } x < x_{center} \\\\
            1 - S_2(x) & \\text{if } x \\geq x_{center}
        \\end{cases}

    where:

    .. math::

        S_1(x) = \\frac{1}{1 + base^{-\\frac{10 \\cdot coef_{si}}{coef_{div}} \\cdot (x - low)}}

        S_2(x) = \\frac{1}{1 + base^{-\\frac{10 \\cdot coef_{se}}{coef_{div}} \\cdot (x - high)}}

        x_{center} = \\frac{high + low}{2}

    If invert is True, the function becomes:

    .. math::

        D_{inverted}(x) = 1 - D(x)

    Finally, the shift is applied:

    .. math::

        D_{final}(x) = D(x) \\cdot (1 - shift) + shift

    Parameters:
        params (Optional[Dict[str, Any]]): Initial parameters for the double sigmoid function. Defaults to None.

    Attributes:
        low (float): The lower bound of the sigmoid range.
        high (float): The upper bound of the sigmoid range.
        coef_div (float): The divisor coefficient for slope adjustment, range [0.0, inf), default 1.0.
        coef_si (float): The slope coefficient for the increasing part, range [0.0, inf), default 1.0.
        coef_se (float): The slope coefficient for the decreasing part, range [0.0, inf), default 1.0.
        base (float): The base of the exponential function, range (1.0, inf), default 10.0.
        invert (bool): Whether to invert the result, default False.
        shift (float): The vertical shift of the sigmoid, range [0.0, 1.0], default 0.0.

    Raises:
        InvalidBoundaryError: If base is less than or equal to 1, or if high is less than low.
        InvalidParameterTypeError: If any parameter is of an invalid type.
        ParameterValueNotSet: If any required parameter is not set.

    Implementation Details:
    The double sigmoid function uses two different implementations based on the input parameters:

    1. Hard Sigmoid: When `coef_div` is 0, a hard sigmoid function is used.
    2. Stable Sigmoid: For all other cases, a numerically stable sigmoid implementation is used.

    The choice between these implementations is made automatically based on the input
    parameters, ensuring accurate and stable results across a wide range of inputs.

    Usage Example:

    >>> from pumas.desirability import desirability_catalogue

    >>> desirability_class = desirability_catalogue.get("double_sigmoid")

    >>> params = {'low': 3.0, 'high': 7.0, 'coef_div': 1.0, 'coef_si': 2.0, 'coef_se': 2.0, 'base': 10.0, 'invert': False, 'shift': 0.0}
    >>> desirability = desirability_class(params=params)
    >>> print(desirability.get_parameters_values())
    {'low': 3.0, 'high': 7.0, 'coef_div': 1.0, 'coef_si': 2.0, 'coef_se': 2.0, 'base': 10.0, 'invert': False, 'shift': 0.0}

    >>> result = desirability.compute_numeric(x=5.0)
    >>> print(f"{result:.2f}")
    1.00

    >>> result = desirability(x=5.0) # Same as compute_numeric
    >>> print(f"{result:.2f}")
    1.00

    >>> from uncertainties import ufloat
    >>> result = desirability.compute_ufloat(x=ufloat(5.0, 1.0))
    >>> print(result)
    0.9999+/-0.0005

    """  # noqa E501

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the DoubleSigmoid desirability function.

        Args:
            params (Optional[Dict[str, Any]]): Initial parameters for the double sigmoid function.
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

    def compute_numeric(self, x: Union[int, float]) -> float:
        """
        Compute the double sigmoid desirability for a numeric input.

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
        return compute_numeric_sigmoid(x=x, **parameters)  # type: ignore

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
        self._validate_compute_input(x, UFloat)
        self._check_parameters_values_none()
        parameters = self.get_parameters_values()
        return compute_ufloat_sigmoid(x=x, **parameters)  # type: ignore

    __call__ = compute_numeric
