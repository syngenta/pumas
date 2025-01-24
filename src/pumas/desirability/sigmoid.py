import math

from pumas.desirability.base_models import Desirability


def hard_sigmoid(x: float, k: float) -> float:
    return 1.0 if k * x > 0 else 0.0


def stable_sigmoid(x: float, k: float, base: float) -> float:
    if base == 10:
        h = k * x * math.log(10)
    else:
        h = k * x * math.log(base)

    if h >= 0:
        result = 1.0 / (1.0 + math.exp(-h))
    else:
        result = math.exp(h) / (1.0 + math.exp(h))
    return result


def sigmoid(
    x: float,
    low: float,
    high: float,
    k: float,
    shift: float = 0.0,
    base: float = 10.0,
) -> float:
    """Utility function implementing a numerically stable sigmoid desirability

    Args:
        x (float): The input value for which to compute the desirability.
        low (float): The lower boundary of the desirability function.
        high (float): The upper boundary of the desirability function.
        k (float): Determines the steepness of the sigmoid function.
        base (float): The base of the exponential function used in the sigmoid,
            defaults to 10.
        shift (float): Vertical upward shift of the sigmoid function.

    Returns:
        float: The desirability score computed using the sigmoid function.

    Raises:
        ValueError: If input parameters are invalid.
    """
    if base <= 1:
        raise ValueError("Base must be greater than 1")

    if not 0 <= shift <= 1:
        raise ValueError("Shift must be between 0 and 1")

    if high < low:
        raise ValueError("High must be greater than or equal to low")

    x_centered = x - (high + low) / 2

    if (high - low) == 0:
        # Hard sigmoid case
        result = hard_sigmoid(x=x_centered, k=k)
    else:
        # Stable sigmoid case
        k_adjusted = 10.0 * k / (high - low)
        result = stable_sigmoid(x=x_centered, k=k_adjusted, base=base)

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


class Sigmoid(Desirability):
    """

    The class :class:`Sigmoid<mpstk.desirability.sigmoid.SigmoidDesirability>` implements a sigmoid desirability function.

    The sigmoid function has a flexible curve that is controlled by its parameters
    and can be used to model scenarios where desirability transitions smoothly
    between low and high values.

    Upon initialization, the attributes of some coefficient parameters are set to
    reasonable defaults, which can be changed by the user.

    This class wraps the :function: `sigmoid<mpstk.desirability.sigmoid.sigmpoid>` utility function and
        implements the interface to use it, identifying
        * input parameters: 'x'
        * coefficient parameters: 'low', 'high', 'k', 'base' and 'shift'.

    The `sigmoid(x)` is defined as:

    .. math::

        {sigmoid}(x) = \\frac{1}{1 + {base}^{-arg}} * (1 - {shift}) + {shift}

    Where the  `arg` term is calculated as:

    .. math::

        {arg} = \\frac{10 * k * (x - \\frac{{low} + {high}}{2})}{{high} - {low}}

    Where:
        * x is the input value.
        * low and high define the range within which the sigmoid transitions from near 0 to near 1.
        * k determines the steepness of the sigmoid curve. A positive slope gives an upward sigmoid, and a negative slope gives a downward sigmoid.
        * base determines the base of the exponential function, with base=10 for a base-10 exponential.
        * shift vertically shifts the sigmoid curve. A shift of 0 means the curve ranges from 0 to 1. Positive shifts move the curve upwards maintaining the range within 1. negative shifts are not allowed.

    Usage Example:

    >>> from mpstk.desirability.sigmoid import Sigmoid
    >>> desirability = Sigmoid()
    >>> # print the default values of the coefficient parameters
    >>> print(desirability.get_coefficient_parameters_values())
    {'low': None, 'high': None, 'k': 0.5, 'base': 10.0, 'shift': 0.0}
    >>> desirability.set_coefficient_parameters_values({'low': 0.0, 'high': 1.0,
    ...                                                 'k': 0.1,
    ...                                                 'shift': 0.0,
    ...                                                 'base': 10.
    ...                                                })
    >>> print(desirability.get_coefficient_parameters_values())
    {'low': 0.0, 'high': 1.0, 'k': 0.1, 'base': 10.0, 'shift': 0.0}
    >>> result = desirability.compute_score(x=0.5)
    >>> print(f"{result:.6f}")  # Format to 6 decimal places
    0.500000

    """  # noqa: E501

    def __init__(self):
        """Initializes the SigmoidDesirability with predefined parameter names
        and the sigmoid utility function.
        """
        super().__init__(
            utility_function=sigmoid,
            coefficient_parameters_names=["low", "high", "k", "base", "shift"],
            input_parameters_names=["x"],
        )
        # fix some defaults and boundaries by editing the attributes of some parameters:
        # we se to None the attributes of the parameters for which we
        # have no reasonable attributes
        # to avoid a warning at initialization time

        attributes_change_map = {
            "low": {"min": None, "max": None, "default": None},
            "high": {"min": None, "max": None, "default": None},
            "k": {"min": -1.0, "max": 1.0, "default": 0.5},
            "base": {"min": 1.0, "max": 10.0, "default": 10.0},
            "shift": {"min": 0.0, "max": 1.0, "default": 0.0},
        }
        self.set_coefficient_parameters_attributes(attributes_map=attributes_change_map)
