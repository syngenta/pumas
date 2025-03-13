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
    """
        Value Mapping desirability function implementation.

        This class implements a desirability function that maps string inputs to
        desirability values based on a predefined mapping. It provides a method to
        compute the desirability for string inputs.

        The value mapping function D(x) is defined as follows:

        .. math::

            D(x) = \\begin{cases}
                mapping[x] * (1 - shift) + shift & \\text{if } x \\in mapping \\\\
                NaN & \\text{if } x \\notin mapping
            \\end{cases}

        Where:
        - mapping[x] is the desirability value associated with the input string x
        - shift is an optional vertical shift applied to the result

        Parameters:
            params (Optional[Dict[str, Any]]): Initial parameters for the value mapping function.

        Attributes:
            mapping (Dict[str, float]): A dictionary mapping string inputs to desirability values.
            shift (float): Vertical shift of the function, range [0.0, 1.0], default 0.0.

        Raises:
            ValueError: If any of the mapping values are not between 0 and 1.
            InvalidParameterTypeError: If the input is not a string.
            ParameterValueNotSet: If any required parameter is not set.

        Usage Example:

        >>> from pumas.desirability import desirability_catalogue

        >>> desirability_class = desirability_catalogue.get("value_mapping")

        >>> mapping = {"low": 0.2, "medium": 0.5, "high": 0.8}
        >>> params={"mapping": mapping, "shift": 0.0}
        >>> desirability = desirability_class(params=params)
        >>> desirability.compute_string(x="medium")
        0.5

        >>> result = desirability(x="medium") # Same as compute_string
        >>> print(f"{result:.2f}")
        0.5
        """  # noqa: E501

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
        self._validate_compute_input(x, str)
        self._check_parameters_values_none()
        parameters = self.get_parameters_values()
        return value_mapping(x=x, **parameters)

    def compute_numeric(self, x: float) -> float:
        raise NotImplementedError

    def compute_ufloat(self, x: UFloat) -> UFloat:
        raise NotImplementedError

    __call__ = compute_string
