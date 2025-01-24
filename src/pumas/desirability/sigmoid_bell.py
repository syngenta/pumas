from pumas.desirability.base_models import Desirability
from pumas.desirability.sigmoid import sigmoid


def sigmoid_bell(
    x: float,
    x1: float,
    x2: float,
    x3: float,
    x4: float,
    k: float = 1.0,
    base: float = 10.0,
    invert: bool = False,
    shift: float = 0.0,
) -> float:
    if x3 < x1 or x2 > x4 or x2 < x1 or x4 < x3:
        raise ValueError("Invalid shape parameters. Ensure x1 < x2 < x3 < x4.")
    if base <= 1:
        raise ValueError("'base' must be greater than 1")

    # Combine two sigmoid functions
    sig1 = sigmoid(x, low=x1, high=x2, k=k, shift=0.0, base=base)
    sig2 = sigmoid(x, low=x3, high=x4, k=k, shift=0.0, base=base)

    # Calculate the result
    result = sig1 - sig2

    # invert if needed
    if invert:
        result = 1 - result

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


class SigmoidBell(Desirability):
    def __init__(self):
        """Initializes the DoubleSigmoidDesirability with predefined parameter names
        and the double_sigmoid utility function.
        """
        super().__init__(
            utility_function=sigmoid_bell,
            coefficient_parameters_names=[
                "x1",
                "x2",
                "x3",
                "x4",
                "k",
                "base",
                "invert",
                "shift",
            ],
            input_parameters_names=["x"],
        )

        attributes_change_map = {
            "x1": {"min": None, "max": None, "default": None},
            "x2": {"min": None, "max": None, "default": None},
            "x3": {"min": None, "max": None, "default": None},
            "x4": {"min": None, "max": None, "default": None},
            "k": {"min": 1.0, "max": None, "default": 1.0},
            "base": {"min": 1.0, "max": None, "default": 10.0},
            "invert": {"default": False},
            "shift": {"min": 0.0, "max": 1.0, "default": 0.0},
        }
        self.set_coefficient_parameters_attributes(attributes_map=attributes_change_map)
