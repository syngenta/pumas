import math
import sys
from functools import partial
from types import ModuleType
from typing import Any, Dict, Optional, Union

from pumas.desirability.base_models import Desirability
from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import (
    UFloat,
    umath,
)


def bell(
    x: Union[float, UFloat],
    width: float,
    slope: float,
    center: float,
    invert: bool = False,
    shift: float = 0.0,
    math_module: ModuleType = math,
) -> Union[float, UFloat]:

    exponent = 2 * abs(slope)
    base = abs((x - center) / width)  # type: ignore

    if base > 1 and exponent > math_module.log(sys.float_info.max) / math_module.log(
        base
    ):
        return shift

    result = 1 / (1 + base**exponent)

    # invert if needed
    if invert:
        result = 1 - result

    # Apply the shift
    result = result * (1 - shift) + shift

    return result  # type: ignore


compute_numeric_bell = partial(bell, math_module=math)

compute_ufloat_bell = partial(bell, math_module=umath)


def get_bell_inflection_points(
    center: float, width: float, slope: float
) -> tuple[float, float]:
    """
    Calculate the inflection points of a generalized bell membership function.

    Args:
    center (float): The center of the bell curve.
    width (float): The width parameter of the bell curve.
    slope (float): The slope parameter of the bell curve.

    Returns:
    tuple: A tuple containing the two inflection points (left, right).
    """
    if width <= 0 or slope <= 0:
        raise ValueError("Width and slope must be positive.")

    # Calculate the distance from the center to each inflection point
    distance = width * (2 ** (1 / (2 * slope)) - 1) ** (1 / (2 * slope))

    # Calculate the two inflection points
    left_inflection = center - distance
    right_inflection = center + distance

    return left_inflection, right_inflection


def get_bell_slope_pivot_points(
    center: float, width: float, slope: float
) -> tuple[float, float]:
    """
    Calculate the pivot points of a generalized bell membership function.
    The values of these points do not change when varying the slope parameter.

    Args:
    center (float): The center of the bell curve.
    width (float): The width parameter of the bell curve.
    slope (float): The slope parameter of the bell curve.

    Returns:
    tuple: A tuple containing the two pivot points (left, right).
    """
    if width <= 0 or slope <= 0:
        raise ValueError("Width and slope must be positive.")

    # Calculate the two pivot points
    left_pivot = center - width
    right_pivot = center + width

    return left_pivot, right_pivot


class Bell(Desirability):
    """
    Bell desirability function implementation.

    Mathematical Definition:

    The bell function is defined as:

    .. math::

        f(x) = \\frac{1}{1 + |\\frac{x - center}{width}|^{2 * slope}} * (1 - shift) + shift

    Where:
        * `x` is the input value.
        * `width` is the width parameter of the bell curve, controls the horizontal spread (width > 0).
        * `slope` is the slope parameter of the bell curve, controls the steepness of the curve's sides.
        * `center` is the center of the bell curve, indicates the `x` value at the highest point of the curve (peak).
        * `invert` is a boolean parameter that, if True, inverts the curve.
        * `shift` is the vertical shift applied to the entire curve, ranging from 0 (no shift) to 1 (maximum shift).


    Parameters:
        params (Optional[Dict[str, Any]]): Initial parameters for the sigmoid function. Defaults to None.


    Attributes:
        width (float): The width parameter of the bell curve, controls the horizontal spread (width > 0).
        slope (float): The slope parameter of the bell curve, controls the steepness of the curve's sides.
        center (float): The center of the bell curve, indicates the `x` value at the highest point of the curve (peak).
        invert (bool): If True, the curve is inverted, i.e., the desirability decreases as `x` approaches the center.
        shift (float): The vertical shift applied to the entire curve, ranging from 0 (no shift) to 1 (maximum shift).

    Usage Example:

    >>> from pumas.desirability import desirability_catalogue

    >>> desirability_class = desirability_catalogue.get("bell")

    >>> params = {'center': 0.5, 'width': 1.0, 'slope': 2.0, 'invert': False, 'shift': 0.0}
    >>> desirability = desirability_class(params=params)
    >>> print(desirability.get_parameters_values())
    {'center': 0.5, 'width': 1.0, 'slope': 2.0, 'invert': False, 'shift': 0.0}

    >>> result = desirability.compute_numeric(x=0.5)
    >>> print(f"{result:.2f}")
    1.00

    >>> result = desirability(x=0.5) # Same as compute_numeric
    >>> print(f"{result:.2f}")
    1.00

    >>> from uncertainties import ufloat
    >>> result = desirability.compute_ufloat(x=ufloat(0.5, 0.1))
    >>> print(result)
    1.0+/-0
    """  # noqa: E501

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Bell desirability function.

        Args:
            params (Optional[Dict[str, Any]]): Initial parameters for the sigmoid function.
        """  # noqa: E501
        super().__init__()
        self._set_parameter_definitions(
            {
                "center": {
                    "type": "float",
                    "min": float("-inf"),
                    "max": float("inf"),
                    "default": None,
                },
                "width": {
                    "type": "float",
                    "min": sys.float_info.epsilon,
                    "max": float("inf"),
                    "default": None,
                },
                "slope": {
                    "type": "float",
                    "min": 0.0,
                    "max": float("inf"),
                    "default": 1.0,
                },
                "invert": {"type": "bool", "default": False},
                "shift": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            }
        )
        self._validate_and_set_parameters(params)

    def compute_numeric(self, x: Union[int, float]) -> float:
        """
        Compute the bell desirability for a numeric input.

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
        return compute_numeric_bell(x, **parameters)  # type: ignore

    def compute_ufloat(self, x: UFloat) -> UFloat:
        """
        Compute the bell desirability for an uncertain float input.

        Args:
            x (UFloat): The uncertain float input value.

        Returns:
            UFloat: The computed desirability value with uncertainty.

        Raises:
            InvalidParameterTypeError: If the input is not a UFloat.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_compute_input(x, UFloat)
        # self._check_parameters_values_none()
        parameters = self.get_parameters_values()
        return compute_ufloat_bell(x, **parameters)  # type: ignore

    __call__ = compute_numeric
