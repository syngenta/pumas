import math
import sys

from pumas.desirability.base_models import Desirability


def bell(
    x: float,
    width: float,
    slope: float,
    center: float,
    invert: bool = False,
    shift: float = 0.0,
) -> float:
    """
    Utility function to calculate the membership degree of `x` in a general bell-shaped fuzzy membership function.

    The bell-shaped membership function is defined by three parameters: `width`, `slope`, and `center`,
    where `width` controls the width (parameter 'a'), `slope` is a slope parameter (parameter 'b'),
    and `center` is the center of the curve (parameter 'c'). An additional parameter `shift` vertically
    displaces the curve.

    The mathematical representation of the bell-shaped function including an upward shift is as follows:

    .. math::

        f(x) = \\frac{1}{1 + |\\frac{x - center}{width}|^{2 * slope}} * (1 - shift) + shift

    Args:
        x (float): The input value for which the membership degree is to be calculated.
        width (float): The width parameter of the bell curve, controls the horizontal spread (width > 0).
        slope (float): The slope parameter of the bell curve, controls the steepness of the curve's sides.
        center (float): The center of the bell curve, indicates the `x` value at the highest point of the curve (peak).
        invert (bool): If True, the curve is inverted, i.e., the membership degree decreases as `x` approaches the center.
        shift (float): The vertical shift applied to the entire curve, ranging from 0 (no shift) to 1 (maximum shift).

    Returns:
        float: The membership degree of `x` within the bell-shaped fuzzy set, adjusted by specified `shift`.

    Raises:
        ValueError: If `width` is not greater than 0, as a non-positive width does not define a bell-shaped curve.
    """  # noqa: E501
    if not 0 <= shift <= 1:
        raise ValueError("The 'shift' parameter must be between 0 and 1, inclusive.")

    if width <= 0:
        raise ValueError("The 'width' parameter must be greater than 0.")

    if abs(width) < sys.float_info.epsilon:
        raise ValueError(
            "Width is too close to zero, which may cause numerical instability."
        )

    exponent = 2 * abs(slope)
    base = abs((x - center) / width)

    if base > 1 and exponent > math.log(sys.float_info.max) / math.log(base):
        return shift

    result = 1 / (1 + base**exponent)

    # invert if needed
    if invert:
        result = 1 - result

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


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
    The class :class:`SigmoidDesirability<mpstk.desirability.desirability.GeneralizedBellMembershipDesirability>` implements a bell-shaped desirability function.

    It implements the definition of input and coefficient parameters.

    This class wraps the general_bell_membership utility function and implements the interface to use it, identifying
        * input parameters: 'x'
        * coefficient parameters: 'width', 'slope', 'center', 'invert', and 'shift'.

    The bell-shaped function has a smooth and adjustable curve that is controlled by its parameters
    and can be used to model scenarios where desirability achieves a peak at a certain value and decreases smoothly.

    Upon initialization, the attributes of the coefficient parameters are set to
    reasonable defaults, which can be changed by the user.

    Usage Example:

    >>> desirability = Bell()
    >>> # print the default values of the coefficient parameters
    >>> print(desirability.get_coefficient_parameters_values())
    {'width': None, 'slope': 1.0, 'center': None, invert: False, 'shift': 0.0}
    >>> desirability.set_coefficient_parameters_values({'width': 1.0, 'slope': 2.0,
    ...                                                 'center': 0.5, 'invert': False,
    ...                                                 'shift': 0.0})
    >>> print(desirability.get_coefficient_parameters_values())
    {'width': 1.0, 'slope': 2.0, 'center': 0.5, invert: False, 'shift': 0.0}
    >>> result = desirability.compute_score(x=0.5)
    >>> print(result)
    1.0
    """  # noqa: E501

    def __init__(self):
        """Initializes the BellDesirability with predefined parameter names
        and the general_bell_membership utility function.
        """
        super().__init__(
            utility_function=bell,
            coefficient_parameters_names=[
                "width",
                "slope",
                "center",
                "invert",
                "shift",
            ],
            input_parameters_names=["x"],
        )
        # Set defaults and boundaries for the parameters:
        attributes_change_map = {
            "width": {"min": sys.float_info.epsilon, "max": None, "default": None},
            "slope": {"min": 1.0, "max": None, "default": 1.0},
            "center": {"min": None, "max": None, "default": None},
            "invert": {"default": False},
            "shift": {"min": 0.0, "max": 1.0, "default": 0.0},
        }
        self.set_coefficient_parameters_attributes(attributes_map=attributes_change_map)
