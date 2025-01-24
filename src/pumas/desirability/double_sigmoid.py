from pumas.desirability.base_models import Desirability
from pumas.desirability.sigmoid import hard_sigmoid, stable_sigmoid


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
) -> float:
    """Utility function implementing a numerically stable double sigmoid desirability"""
    # Input validation

    if coef_si < 0:
        raise ValueError("'coef_si' must be positive")
    if coef_se < 0:
        raise ValueError("'coef_se' must be positive")
    if base <= 1:
        raise ValueError("'base' must be greater than 1")
    if shift < 0 or shift > 1:
        raise ValueError("'shift' must be between 0 and 1")

    x_left, x_right, k, k_left, k_right = low, high, coef_div, coef_si, coef_se

    x_center = (x_right - x_left) / 2 + x_left

    if x < x_center:
        xl = x - x_left
        if k == 0:
            result = hard_sigmoid(xl, k_left)
        else:
            k_left_adjusted = k_left / k
            result = stable_sigmoid(xl, k_left_adjusted, base=base)
    else:
        xr = x - x_right
        if k == 0:
            result = 1 - hard_sigmoid(xr, k_right)
        else:
            k_right_adjusted = k_right / k
            result = 1 - stable_sigmoid(xr, k_right_adjusted, base=base)

    # invert if needed
    if invert:
        result = 1 - result

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


class DoubleSigmoid(Desirability):
    """
    The class DoubleSigmoidDesirability implements a double sigmoid desirability function.

    This class wraps the `double_sigmoid` utility function and implements the interface to use it, identifying:
    * input parameters: 'x'
    * coefficient parameters: 'x_left', 'x_right', 'k', 'k_left', 'k_right', 'base', and 'shift'.

    The double sigmoid function combines two sigmoid functions to create a plateau of high desirability
    between x_left and x_right, with smooth transitions on both sides.

    Usage Example:

    >>> from mpstk.desirability.double_sigmoid import DoubleSigmoid
    >>> desirability = DoubleSigmoid()
    >>> # print the default values of the coefficient parameters
    >>> print(desirability.get_coefficient_parameters_values())
    {'x_left': None, 'x_right': None, 'k': 1.0, 'k_left': 1.0, 'k_right': 1.0, 'base': 10.0, 'shift': 0.0}
    >>> desirability.set_coefficient_parameters_values({
    ...     'x_left': 3.0, 'x_right': 7.0, 'k': 1.0, 'k_left': 2.0, 'k_right': 2.0, 'shift': 0.0, 'base': 10.0
    ... })
    >>> print(desirability.get_coefficient_parameters_values())
    {'x_left': 3.0, 'x_right': 7.0, 'k': 1.0, 'k_left': 2.0, 'k_right': 2.0, 'base': 10.0, 'shift': 0.0}
    >>> result = desirability.compute_score(x=5.0)
    >>> print(f"{result:.6f}")  # Format to 6 decimal places
    0.999089
    """  # noqa E501

    def __init__(self):
        """Initializes the DoubleSigmoidDesirability with predefined parameter names
        and the double_sigmoid utility function.
        """
        super().__init__(
            utility_function=double_sigmoid,
            coefficient_parameters_names=[
                "low",
                "high",
                "coef_div",
                "coef_si",
                "coef_se",
                "base",
                "invert",
                "shift",
            ],
            input_parameters_names=["x"],
        )

        attributes_change_map = {
            "low": {"min": None, "max": None, "default": None},
            "high": {"min": None, "max": None, "default": None},
            "coef_div": {"min": 0.0, "max": None, "default": 1.0},
            "coef_si": {"min": 0.0, "max": None, "default": 1.0},
            "coef_se": {"min": 0.0, "max": None, "default": 1.0},
            "base": {"min": 0.0, "max": None, "default": 10.0},
            "invert": {"default": False},
            "shift": {"min": 0.0, "max": 1.0, "default": 0.0},
        }
        self.set_coefficient_parameters_attributes(attributes_map=attributes_change_map)
