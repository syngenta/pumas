from typing import Dict

from pumas.desirability.base_models import Desirability


def value_mapping(x: str, mapping: Dict[str, float], shift: float = 0.0) -> float:
    # none of the mappings values should be lower than 0 or higher than 1
    if not all(0 <= value <= 1 for value in mapping.values()):
        raise ValueError("Mapping values should be between 0 and 1")

    if shift < 0 or shift > 1:
        raise ValueError("Shift must be between 0 and 1")

    result = mapping.get(x, None)

    if result is None:
        return float("nan")

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


class ValueMapping(Desirability):
    def __init__(self):
        super().__init__(
            utility_function=value_mapping,
            coefficient_parameters_names=["mapping", "shift"],
            input_parameters_names=["x"],
        )
        # Set defaults and boundaries for the parameters:
        attributes_change_map = {
            "mapping": {"default": None},
            "shift": {"min": 0.0, "max": 1.0, "default": 0.0},
        }
        self.set_coefficient_parameters_attributes(attributes_map=attributes_change_map)
