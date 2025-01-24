from pumas.desirability.base_models import Desirability


def right_step(x: float, low: float, high: float, shift: float = 0.0):
    if not 0 <= shift <= 1:
        raise ValueError("Shift must be between 0 and 1")

    _ = low
    result = 1.0 if x >= high else 0.0

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


def left_step(x: float, low: float, high: float, shift: float = 0.0):
    if not 0 <= shift <= 1:
        raise ValueError("Shift must be between 0 and 1")
    _ = high
    result = 1.0 if x <= low else 0.0

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


def step(x: float, low: float, high: float, shift: float = 0.0):
    if not 0 <= shift <= 1:
        raise ValueError("Shift must be between 0 and 1")
    result = 1.0 if low <= x <= high else 0.0

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


class RightStep(Desirability):
    def __init__(self):
        super().__init__(
            utility_function=right_step,
            coefficient_parameters_names=["low", "high", "shift"],
            input_parameters_names=["x"],
        )
        # Set defaults and boundaries for the parameters:
        attributes_change_map = {
            "low": {"min": None, "max": None, "default": None},
            "high": {"min": None, "max": None, "default": None},
            "shift": {"min": 0.0, "max": 1.0, "default": 0.0},
        }
        self.set_coefficient_parameters_attributes(attributes_map=attributes_change_map)


class LeftStep(Desirability):
    def __init__(self):
        super().__init__(
            utility_function=left_step,
            coefficient_parameters_names=["low", "high", "shift"],
            input_parameters_names=["x"],
        )
        # Set defaults and boundaries for the parameters:
        attributes_change_map = {
            "low": {"min": None, "max": None, "default": None},
            "high": {"min": None, "max": None, "default": None},
            "shift": {"min": 0.0, "max": 1.0, "default": 0.0},
        }
        self.set_coefficient_parameters_attributes(attributes_map=attributes_change_map)


class Step(Desirability):
    def __init__(self):
        super().__init__(
            utility_function=step,
            coefficient_parameters_names=["low", "high", "shift"],
            input_parameters_names=["x"],
        )
        # Set defaults and boundaries for the parameters:
        attributes_change_map = {
            "low": {"min": None, "max": None, "default": None},
            "high": {"min": None, "max": None, "default": None},
            "shift": {"min": 0.0, "max": 1.0, "default": 0.0},
        }
        self.set_coefficient_parameters_attributes(attributes_map=attributes_change_map)
