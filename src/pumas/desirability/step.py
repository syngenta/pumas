from typing import Any, Dict, Optional

from pumas.desirability.base_models import Desirability
from pumas.uncertainty.uncertainties_wrapper import UFloat, ufloat


def compute_numeric_right_step(
    x: float, low: float, high: float, shift: float = 0.0
) -> float:
    """
    Calculate the right step function value for a given numeric input.

    Args:
        x (float): The input value.
        low (float): Lower bound (unused in this function).
        high (float): Upper bound (step threshold).
        shift (float): Vertical shift of the step. Default is 0.0.

    Returns:
        float: The calculated right step value.
    """
    _ = low
    result = 1.0 if x >= high else 0.0

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


def compute_ufloat_right_step(
    x: UFloat, low: float, high: float, shift: float = 0.0
) -> UFloat:
    """
    Calculate the right step function value for a given uncertain float input.

    Args:
        x (UFloat): The uncertain float input value.
        low (float): Lower bound (unused in this function).
        high (float): Upper bound (step threshold).
        shift (float): Vertical shift of the step. Default is 0.0.

    Returns:
        UFloat: The calculated right step value with uncertainty.
    """
    _ = low
    x_nominal_value, x_std_dev = x.nominal_value, x.std_dev  # type: ignore
    result = (
        ufloat(nominal_value=1.0, std_dev=x_std_dev)
        if x_nominal_value >= high
        else ufloat(nominal_value=0.0, std_dev=x_std_dev)
    )

    # Apply the shift
    result = result * (1 - shift) + shift

    return result  # type: ignore


def compute_numeric_left_step(
    x: float, low: float, high: float, shift: float = 0.0
) -> float:
    """
    Calculate the left step function value for a given numeric input.

    Args:
        x (float): The input value.
        low (float): Lower bound (step threshold).
        high (float): Upper bound (unused in this function).
        shift (float): Vertical shift of the step. Default is 0.0.

    Returns:
        float: The calculated left step value.
    """
    _ = high
    result = 1.0 if x <= low else 0.0

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


def compute_ufloat_left_step(
    x: UFloat, low: float, high: float, shift: float = 0.0
) -> UFloat:
    """
    Calculate the left step function value for a given uncertain float input.

    Args:
        x (UFloat): The uncertain float input value.
        low (float): Lower bound (step threshold).
        high (float): Upper bound (unused in this function).
        shift (float): Vertical shift of the step. Default is 0.0.

    Returns:
        UFloat: The calculated left step value with uncertainty.
    """
    _ = high
    x_nominal_value, x_std_dev = x.nominal_value, x.std_dev  # type: ignore
    result = (
        ufloat(nominal_value=1.0, std_dev=x_std_dev)
        if x_nominal_value <= low
        else ufloat(nominal_value=0.0, std_dev=x_std_dev)
    )

    # Apply the shift
    result = result * (1 - shift) + shift

    return result  # type: ignore


def compute_numeric_step(
    x: float, low: float, high: float, invert: bool, shift: float = 0.0
) -> float:
    """
    Calculate the centered step function value for a given numeric input.

    Args:
        x (float): The input value.
        low (float): Lower bound of the step.
        high (float): Upper bound of the step.
        invert (bool): Whether to invert the result. Default is False.
        shift (float): Vertical shift of the step. Default is 0.0.

    Returns:
        float: The calculated step value.
    """
    result = 1.0 if low <= x <= high else 0.0

    # invert if needed
    if invert:
        result = 1 - result

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


def compute_ufloat_step(
    x: UFloat, low: float, high: float, invert: bool, shift: float = 0.0
) -> UFloat:
    """
    Calculate the centered step function value for a given uncertain float input.

    Args:
        x (UFloat): The uncertain float input value.
        low (float): Lower bound of the step.
        high (float): Upper bound of the step.
        invert (bool): Whether to invert the result. Default is False.
        shift (float): Vertical shift of the step. Default is 0.0.

    Returns:
        UFloat: The calculated step value with uncertainty.
    """
    x_nominal_value, x_std_dev = x.nominal_value, x.std_dev  # type: ignore
    result = (
        ufloat(nominal_value=1.0, std_dev=x_std_dev)
        if low <= x_nominal_value <= high
        else ufloat(nominal_value=0.0, std_dev=x_std_dev)
    )

    # invert if needed
    if invert:
        result = 1 - result

    # Apply the shift
    result = result * (1 - shift) + shift

    return result  # type: ignore


class RightStep(Desirability):
    """
    Right Step desirability function implementation.

    This class implements a right step desirability function with adjustable parameters.
    It provides methods to compute the desirability for both numeric and uncertain float inputs.

    The right step function is defined as:

    .. math::

        D(x) = \\begin{cases}
            1 & \\text{if } x \\geq high \\\\
            0 & \\text{otherwise}
        \\end{cases}

    Finally, the shift is applied:

    .. math::

        D_{final}(x) = D(x) \\cdot (1 - shift) + shift

    Parameters:
        params (Optional[Dict[str, Any]]): Initial parameters for the right step function.

    Attributes:
        low (float): Lower bound (unused in this function).
        high (float): Upper bound (step threshold).
        shift (float): Vertical shift of the step, range [0.0, 1.0], default 0.0.

       Usage Example:

    >>> from pumas.desirability import desirability_catalogue

    >>> desirability_class = desirability_catalogue.get("rightstep")

    >>> params = {'low': 0.0, 'high': 1.0, 'shift': 0.0}
    >>> desirability = desirability_class(params=params)
    >>> print(desirability.get_parameters_values())
    {'low': 0.0, 'high': 1.0, 'shift': 0.0}

    >>> result = desirability.compute_numeric(x=0.5)
    >>> print(f"{result:.2f}")
    0.00

    >>> result = desirability(x=1.5) # Same as compute_numeric
    >>> print(f"{result:.2f}")
    1.00

    >>> from pumas.uncertainty.uncertainties_wrapper import ufloat
    >>> result = desirability.compute_ufloat(x=ufloat(1.0, 0.1))
    >>> print(result)
    1.00+/-0.10
    """  # noqa: E501

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the RightStep desirability function.

        Args:
            params (Optional[Dict[str, Any]]):
                Initial parameters for the right step function.
        """
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
                "shift": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            }
        )
        self._validate_and_set_parameters(params)

    def compute_numeric(self, x: float) -> float:
        """
        Compute the right step desirability for a numeric input.

        Args:
            x (float): The input value.

        Returns:
            float: The computed desirability value.

        Raises:
            InvalidParameterTypeError: If the input is not a float.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_compute_input(x, float)
        self._check_parameters_values_none()
        parameters = self.get_parameters_values()
        return compute_numeric_right_step(x=x, **parameters)

    def compute_ufloat(self, x: UFloat) -> UFloat:
        """
        Compute the right step desirability for an uncertain float input.

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
        return compute_ufloat_right_step(x=x, **parameters)

    __call__ = compute_numeric


class LeftStep(Desirability):
    """
    Left Step desirability function implementation.

    This class implements a left step desirability function with adjustable parameters.
    It provides methods to compute the desirability for both numeric and uncertain float inputs.

    The left step function is defined as:

    .. math::

        D(x) = \\begin{cases}
            1 & \\text{if } x \\leq low \\\\
            0 & \\text{otherwise}
        \\end{cases}

    Finally, the shift is applied:

    .. math::

        D_{final}(x) = D(x) \\cdot (1 - shift) + shift

    Parameters:
        params (Optional[Dict[str, Any]]): Initial parameters for the left step function.

    Attributes:
        low (float): Lower bound (step threshold).
        high (float): Upper bound (unused in this function).
        shift (float): Vertical shift of the step, range [0.0, 1.0], default 0.0.

        Usage Example:

    >>> from pumas.desirability import desirability_catalogue

    >>> desirability_class = desirability_catalogue.get("leftstep")

    >>> params = {'low': 1.0, 'high': 2.0, 'shift': 0.0}
    >>> desirability = desirability_class(params=params)
    >>> print(desirability.get_parameters_values())
    {'low': 1.0, 'high': 2.0, 'shift': 0.0}

    >>> result = desirability.compute_numeric(x=0.5)
    >>> print(f"{result:.2f}")
    1.00

    >>> result = desirability(x=1.5) # Same as compute_numeric
    >>> print(f"{result:.2f}")
    0.00

    >>> from pumas.uncertainty.uncertainties_wrapper import ufloat
    >>> result = desirability.compute_ufloat(x=ufloat(1.0, 0.1))
    >>> print(result)
    1.00+/-0.10
    """  # noqa: E501

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the LeftStep desirability function.

        Args:
            params (Optional[Dict[str, Any]]): Initial parameters for the left step function.
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
                "shift": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            }
        )
        self._validate_and_set_parameters(params)

    def compute_numeric(self, x: float) -> float:
        """
        Compute the left step desirability for a numeric input.

        Args:
            x (float): The input value.

        Returns:
            float: The computed desirability value.

        Raises:
            InvalidParameterTypeError: If the input is not a float.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_compute_input(x, float)
        self._check_parameters_values_none()
        parameters = self.get_parameters_values()
        return compute_numeric_left_step(x=x, **parameters)

    def compute_ufloat(self, x: UFloat) -> UFloat:
        """
        Compute the left step desirability for an uncertain float input.

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
        return compute_ufloat_left_step(x=x, **parameters)

    __call__ = compute_numeric


class Step(Desirability):
    """
    Centered Step desirability function implementation.

    This class implements a centered step desirability function.


    The centered step function is defined as:

    .. math::

        D(x) = \\begin{cases}
            1 & \\text{if } low \\leq x \\leq high \\\\
            0 & \\text{otherwise}
        \\end{cases}

    If invert is True, the function becomes:

    .. math::

        D_{inverted}(x) = 1 - D(x)

    Finally, the shift is applied:

    .. math::

        D_{final}(x) = D(x) \\cdot (1 - shift) + shift

    Parameters:
        params (Optional[Dict[str, Any]]): Initial parameters for the centered step function.

    Attributes:
        low (float): Lower bound of the step.
        high (float): Upper bound of the step.
        invert (bool): Whether to invert the result, default False.
        shift (float): Vertical shift of the step, range [0.0, 1.0], default 0.0.

        Usage Example:

    >>> from pumas.desirability import desirability_catalogue

    >>> desirability_class = desirability_catalogue.get("step")

    >>> params = {'low': 0.0, 'high': 1.0, "invert": False, 'shift': 0.0}
    >>> desirability = desirability_class(params=params)
    >>> print(desirability.get_parameters_values())
    {'low': 0.0, 'high': 1.0, 'invert': False, 'shift': 0.0}

    >>> result = desirability.compute_numeric(x=-1.0)
    >>> print(f"{result:.2f}")
    0.00

    >>> result = desirability.compute_numeric(x=0.5)
    >>> print(f"{result:.2f}")
    1.00

    >>> result = desirability(x=1.5) # Same as compute_numeric
    >>> print(f"{result:.2f}")
    0.00

    >>> from pumas.uncertainty.uncertainties_wrapper import ufloat
    >>> result = desirability.compute_ufloat(x=ufloat(0.5, 0.1))
    >>> print(result)
    1.00+/-0.10

    Note: The uncertainty in the last example is 1.00 because the step function
    is discontinuous, and the error is takne from the input.
    """  # noqa: E501

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the Step desirability function.

        Args:
            params (Optional[Dict[str, Any]]): Initial parameters for the centered step function.
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
                "invert": {"type": "bool", "default": False},
                "shift": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            }
        )
        self._validate_and_set_parameters(params)

    def compute_numeric(self, x: float) -> float:
        """
        Compute the centered step desirability for a numeric input.

        Args:
            x (float): The input value.

        Returns:
            float: The computed desirability value.

        Raises:
            InvalidParameterTypeError: If the input is not a float.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_compute_input(x, float)
        self._check_parameters_values_none()
        parameters = self.get_parameters_values()
        return compute_numeric_step(x=x, **parameters)

    def compute_ufloat(self, x: UFloat) -> UFloat:
        """
        Compute the centered step desirability for an uncertain float input.

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
        return compute_ufloat_step(x=x, **parameters)

    __call__ = compute_numeric
