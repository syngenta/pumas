from typing import Any, Dict, Optional

from pumas.desirability.base_models import Desirability
from pumas.uncertainty.uncertainties_wrapper import UFloat


def value_mapping(x: str, mapping: Dict[str, float], shift: float = 0.0) -> float:
    # none of the mappings values should be lower than 0 or higher than 1
    if not all(0 <= value <= 1 for value in mapping.values()):
        raise ValueError("Mapping values should be between 0 and 1")

    result = mapping.get(x, float("nan"))

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


class ValueMapping(Desirability):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._set_parameter_definitions(
            {
                "mapping": {"type": "iterable", "default": None},
                "shift": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            }
        )
        self._validate_and_set_parameters(params)

    def compute_string(self, x: str) -> float:
        """
        Compute the value mapping desirability for a string input.

        Args:
            x (float): The input value.

        Returns:
            float: The computed desirability value.

        Raises:
            InvalidParameterTypeError: If the input is not a float.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_input(x, str)
        self._check_coefficient_parameters_values()
        parameters = self.get_parameters_values()
        return value_mapping(x=x, **parameters)

    def compute_numeric(self, x: float) -> float:
        raise NotImplementedError

    def compute_ufloat(self, x: UFloat) -> UFloat:
        raise NotImplementedError

    __call__ = compute_string
