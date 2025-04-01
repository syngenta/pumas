"""
Parameter Classes
==================

The package provides a collection of concrete parameter classes designed to wrap
Python's nominal types (int, float, str, and bool) and an extension to support
UFloat types from the uncertainties package. These parameter classes are intended
to serve as robust building blocks for systems implementing a parameterized version
of the Strategy pattern. The purpose is to facilitate the validation and use of
configurable parameters across different algorithms and strategies.

The parameter classes included offer features such as:

- Initialization to `None` by default, unless a value is explicitly provided.
- Methods to safely change the parameter values while maintaining internal consistency.
- Extensive validation to reduce the chance of errors when using the parameters.
- The guaranteed type of value, boundary checking (for numeric types), and accepted
  values checking (for string types).

Classes:
    IntParameter: Wraps an integer value with an optional min and max boundary.
    FloatParameter: Wraps a float value with an optional min and max boundary.
    StrParameter: Wraps a string value, with optional accepted values.
    BoolParameter: Wraps a boolean value.
    UFloatParameter: Wraps a UFloat (uncertain float) value, encapsulating numbers with uncertainties, whilst allowing optional min and max boundaries based on the nominal value.

Every parameter class have the 'name' attribute, which is used to identify the parameter.
The 'value' attribute is used to store the value of the parameter.
The optional `default` attribute is used to set the default value of the parameter.
Each parameter class possess different properties, depending on the type of the parameter.
They all possess a param_type attribute containing the type the parameter wraps.

The `IntParameter`, `FloatParameter`, and `UFloatParameter` classes have the optional 'min' and 'max' attributes,
which are used to set the boundaries of the parameter.
The `StrParameter` class has the optional `accepted_values` attribute, which is used to set the accepted values of the parameter.

Both the value. the defualt, and the other parameter attributes can be set set upos instantiation of the parameter class.
However, only the value of the parameter should be changed after instantiation, using the `set_value()` method.

Class Diagram
-------------

.. mermaid::

    classDiagram

          class Parameter {
              <<abstract>>
              +String name
              +get_value()
              +set_value(value)
              +param_type (Property)
          }
          class IntParameter {
              -Optional[int] default
              -Optional[int] min
              -Optional[int] max
              +set_value(value: Optional[int])
          }
          class FloatParameter {
              -Optional[float] default
              -Optional[float] min
              -Optional[float] max
              +set_value(value: Optional[float])
          }
          class StrParameter {
              -Optional[str] default
              -Optional[List[str]] accepted_values
              +set_value(value: Optional[str])
          }
          class BoolParameter {
              -Optional[bool] default
              +set_value(value: Optional[bool])
          }
          class UFloatParameter {
              -Optional[UFloat] default
              -Optional[UFloat] min
              -Optional[UFloat] max
              +set_value(value: Optional[UFloat])
          }
         class IterableParameter {
              -Optional[Iterable] default
              -Optional[Iterable] accepted_values
              +set_value(value: Optional[Iterable])
         }
          Parameter <|-- IntParameter
          Parameter <|-- FloatParameter
          Parameter <|-- StrParameter
          Parameter <|-- BoolParameter
          Parameter <|-- UFloatParameter
          Parameter <|-- IterableParameter



Example Usage:

>>> from pumas.architecture.parameters import IntParameter

>>> int_param = IntParameter(name="IntParam", default=10, min=0, max=20)
>>> int_param.value
10
>>> int_param.set_value(15)
>>> int_param.value
15

>>> from uncertainties import ufloat
>>> from pumas.architecture.parameters import UFloatParameter

>>> ufloat_param = UFloatParameter(name="UFloatParam", default=ufloat(1, 0.1), min=ufloat(0, 0.1), max=ufloat(2, 0.1))
>>> ufloat_param.value.nominal_value
1.0
>>> ufloat_param.set_value(ufloat(1.5, 0.1))
>>> ufloat_param.value.nominal_value
1.5

The use of these parameter classes helps to enforce a consistent approach to
handling parameters across different parts of an application.

Note:
    - Instantiation of any parameter with a type that doesn't match the expected
      will result in `InvalidParameterTypeError`.
    - Attempting to set values outside the range for `IntParameter`, `FloatParameter`, and `UFloatParameter`
      will result in `InvalidBoundaryError`.
    - Passing non-strings for a parameter `name` will result in `InvalidParameterNameError`.
    - Setting an unrecognized string value for `StrParameter` when `accepted_values` is
      defined will result in `InvalidAcceptedValuesError`.

Parameter Manager
==================
The ParameterManager class is responsible for generatign Parameter
objects from a function using its type hints.
In addition to that the ParameterManager class is also responsible for updating the parameters attributes.
This is necessary because, in our implementation the attributes of the parameters,
are not supposed to be changed, with the nociable exception of the value attribute,
that can be changed by the Parametert set_value() method.
The ParameterManager class is thus responsible for updating the parameters attributes,
by creating a new instance of the parameter class, initialized with the new attributes.

Example Usage:

>>> from pumas.architecture.parameters import ParameterManager

Initialize the ParameterManager with parameter definitions.

>>> param_defs = {
...     "q": {"type": "int", "default": 5}
... }
>>> pm = ParameterManager(param_defs)

Check the initial state of the parameters managed by the ParameterManager.
The expected output should show a dictionary containing representation of an `IntParameter` instance for 'x'.
In this case, the default value of 'q' is 5, and no boundaries are defined.

>>> pm.parameters_map
{'q': IntParameter(name='q', default=5, min=None, max=None)}

Check the current value of the parameter: it defaults to 5.

>>> pm.parameters_map['q'].value
5

>>> id_1 = id(pm.parameters_map['q'])

>>> pm.set_parameter_value('q', 10)  # Set a new value for the parameter.
>>> pm.parameters_map['q'].value  # Check the new value of the parameter.
10
>>> id_2 = id(pm.parameters_map['q'])

Changing the value of the parameter should not change the parameter instance.
>>> id_1 == id_2
True

"""  # noqa: E501

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Type, Union

from pumas.architecture.catalogue import Catalogue
from pumas.architecture.exceptions import (
    InvalidAcceptedValuesError,
    InvalidBoundaryDefinitionError,
    InvalidBoundaryError,
    InvalidParameterAttributeError,
    InvalidParameterNameError,
    InvalidParameterTypeError,
    ParameterDefinitionError,
    ParameterNotFoundError,
    ParameterSettingError,
    ParameterSettingWarning,
    ParameterUpdateAttributeWarning,
    ParameterValueNotSet,
)
from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import UFloat

# monkey patching the name attribute that is missing from the Iterable type
Iterable.__name__ = "Iterable"


# Helper functions to validate types
def validate_name(name: str) -> None:
    if isinstance(name, str) and len(name.strip()) == 0:
        raise InvalidParameterNameError(
            f"The parameter name is empty: {type(name).__name__}"
        )

    if not isinstance(name, str):
        raise InvalidParameterNameError(
            f"Expected type str, got {type(name).__name__} instead."
        )


def validate_type(item: Any, expected_type: Type[Any]) -> None:
    if (isinstance(item, bool) and expected_type is not bool and item is not None) or (
        not isinstance(item, expected_type) and item is not None
    ):
        raise InvalidParameterTypeError(
            f"Expected type {expected_type.__name__}, "
            f"got {type(item).__name__} instead."
        )


def validate_range_values(
    min_value: Optional[Union[int, float]], max_value: Optional[Union[int, float]]
) -> None:
    if min_value is not None and max_value is not None and min_value > max_value:
        raise InvalidBoundaryDefinitionError(
            f"Minimum value {min_value} is greater than maximum value {max_value}."
        )


def validate_range_types(
    min_value: Optional[Union[int, float]],
    max_value: Optional[Union[int, float]],
    expected_type: Type[Any],
) -> None:
    if min_value is not None and not isinstance(min_value, expected_type):
        raise InvalidBoundaryDefinitionError(
            f"Expected type {expected_type}, got {type(min_value).__name__} instead."
        )
    if max_value is not None and not isinstance(max_value, expected_type):
        raise InvalidBoundaryDefinitionError(
            f"Expected type {expected_type}, got {type(max_value).__name__} instead."
        )


def validate_range_application(
    item: Any,
    min_value: Optional[Union[int, float]],
    max_value: Optional[Union[int, float]],
) -> None:
    if (min_value is not None and item < min_value) or (
        max_value is not None and item > max_value
    ):
        raise InvalidBoundaryError(
            f"Parameter Value {item} is "
            f"outside the allowed range [{min_value}, {max_value}]."
        )


def validate_accepted_values_types(
    accepted_values: Optional[Iterable[Any]], expected_type: Type[Any]
) -> None:
    if accepted_values is not None:
        if not all(isinstance(av, expected_type) for av in accepted_values):
            raise InvalidBoundaryDefinitionError(
                f"Erroneous type for accepted values (expected type {expected_type}) "
            )


def validate_accepted_values_application(
    item: Any, accepted_values: Optional[Iterable[Any]]
) -> None:
    if item is None:
        return
    if accepted_values and item not in accepted_values:
        raise InvalidAcceptedValuesError(
            f"Value {item} is not in the list of accepted values: {accepted_values}."
        )


@dataclass
class Parameter(ABC):
    """
    Abstract base class for different types of parameters.

    Each subclass must implement a specific 'param_type' property and a
    method to set the value of the parameter.
    This class provides a getter and setter for the parameter value
    with basic type validation.

    Attributes:
        name (str): The name of the parameter identifying it uniquely among others.
        _value (Any): The internal storage for the parameter's value,
            initialized to None.
    """

    name: str = field(init=True, repr=True)
    _value: Any = field(init=False, repr=False, default=None)

    @abstractmethod
    def param_type(self) -> Type[Any]:
        """Expected data_frame type of the parameter value."""

    @property
    def value(self) -> Any:
        """Current value of the parameter, possibly None if not yet set."""
        return self._value

    @abstractmethod
    def set_value(self, value: Any) -> None:
        """Sets the value of the parameter after validation."""


@dataclass
class IntParameter(Parameter):
    """
    Represents an integer parameter with optional minimum and maximum boundaries.

    Inherits from the abstract `Parameter` class and sets `param_type` to `int`.
    It allows setting a value within specified boundaries. If the set value is outside
    these boundaries, an `InvalidBoundaryError` is raised.

    Attributes:
        default (Optional[int]): The default value for the parameter.
        min (Optional[int]): The minimum boundary for the parameter value.
        max (Optional[int]): The maximum boundary for the parameter value.
    """

    default: Optional[int] = field(init=True, repr=True, default=None)
    min: Optional[int] = field(init=True, repr=True, default=None)
    max: Optional[int] = field(init=True, repr=True, default=None)

    @property
    def param_type(self) -> Type[Any]:
        """Returns the parameter type (`int`)."""
        return int

    def __post_init__(self):
        validate_name(name=self.name)
        validate_range_types(
            min_value=self.min, max_value=self.max, expected_type=self.param_type
        )
        validate_range_values(min_value=self.min, max_value=self.max)
        if self.default is not None:
            self.set_value(self.default)

    def set_value(self, value: Optional[int]) -> None:
        """
        Sets the value of the integer parameter after validating its type and range.

        Args:
            value (Optional[int]): The value to be set for the parameter.
                None is allowed to unset.

        Raises:
            InvalidParameterTypeError: If the value's type does not match `int`.
            InvalidBoundaryError: If the value is outside the allowed range (
                if 'min' or 'max' are defined).
        """
        if value is not None:
            validate_type(item=value, expected_type=self.param_type)
            validate_range_application(
                item=value, min_value=self.min, max_value=self.max
            )
        self._value = value


@dataclass
class FloatParameter(Parameter):
    """
    Represents a floating-point parameter with optional minimum and maximum boundaries.

    Inherits from the abstract `Parameter` class and sets `param_type` to `float`.
    It allows setting a value within specified boundaries. If the set value is outside
    these boundaries, an `InvalidBoundaryError` is raised.

    Attributes:
        default (Optional[float]): The default value for the parameter.
        min (Optional[float]): The minimum boundary for the parameter value.
        max (Optional[float]): The maximum boundary for the parameter value.
    """

    default: Optional[float] = field(init=True, repr=True, default=None)
    min: Optional[float] = field(init=True, repr=True, default=None)
    max: Optional[float] = field(init=True, repr=True, default=None)

    @property
    def param_type(self) -> Type[Any]:
        """Returns the parameter type (`float`)."""
        return float

    def __post_init__(self):
        validate_name(name=self.name)
        validate_range_types(
            min_value=self.min, max_value=self.max, expected_type=self.param_type
        )
        validate_range_values(min_value=self.min, max_value=self.max)
        if self.default is not None:
            self.set_value(self.default)

    def set_value(self, value: Optional[float]) -> None:
        """
        Sets the value of the floating-point parameter
        after validating its type and range.

        Args:
            value (Optional[float]): The value to be set for the parameter.
                None is allowed to unset.

        Raises:
            InvalidParameterTypeError: If the value's type does not match `float`.
            InvalidBoundaryError: If the value is outside the
                allowed range (if 'min' or 'max' are defined).
        """
        if value is not None:
            validate_type(item=value, expected_type=self.param_type)
            validate_range_application(
                item=value, min_value=self.min, max_value=self.max
            )
        self._value = value


@dataclass
class StrParameter(Parameter):
    """
    Represents a string parameter that may have a set of acceptable values.

    Inherits from the abstract `Parameter` class and sets `param_type` to `str`.

    Attributes:
        default (Optional[str]): The default value for the parameter.
        accepted_values (Optional[Iterable[str]]):
            An iterable of acceptable string values.
    """

    default: Optional[str] = field(init=True, repr=True, default=None)
    accepted_values: Optional[Iterable[str]] = field(init=True, repr=True, default=None)

    @property
    def param_type(self) -> Type[Any]:
        """Returns the parameter type (`str`)."""
        return str

    def __post_init__(self):
        validate_name(name=self.name)
        validate_accepted_values_types(
            accepted_values=self.accepted_values, expected_type=self.param_type
        )
        if self.default is not None:
            self.set_value(self.default)

    def set_value(self, value: Optional[str]) -> None:
        """
        Sets the value of the string parameter after validating
        its type and against accepted values.

        Args:
            value (Optional[str]): The value to be set for the parameter.
                None is allowed to unset.

        Raises:
            InvalidParameterTypeError: If the value's type does not match `str`.
            InvalidAcceptedValuesError: If the value is not in the set of
                accepted values.
        """
        if value is not None:
            validate_type(item=value, expected_type=self.param_type)
            if self.accepted_values is not None:
                validate_accepted_values_application(
                    item=value, accepted_values=self.accepted_values
                )
        self._value = value


@dataclass
class BoolParameter(Parameter):
    """
    Represents a boolean parameter.

    Inherits from the abstract `Parameter` class and sets `param_type` to `bool`.

    Attributes:
        default (Optional[bool]): The default value for the parameter.
    """

    default: Optional[bool] = field(init=True, repr=True, default=None)

    @property
    def param_type(self) -> Type[Any]:
        """Returns the parameter type (`bool`)."""
        return bool

    def __post_init__(self):
        validate_name(name=self.name)
        if self.default is not None:
            self.set_value(self.default)

    def set_value(self, value: Optional[bool]) -> None:
        """
        Sets the value of the boolean parameter after validating its type.

        Args:
            value (Optional[bool]): The value to be set for the parameter.
                None is allowed.

        Raises:
            InvalidParameterTypeError: If the value's type does not match `bool`.
        """
        if value is not None:
            validate_type(item=value, expected_type=self.param_type)
        self._value = value


# TODO: define better the min-max boundaries for
#  UFloatParameter and remove type ignoring


@dataclass
class UFloatParameter(Parameter):
    """
    Represents  floating-point parameter with uncertainty and optional minimum
    and maximum boundaries.

    Inherits from the abstract `Parameter` class and sets `param_type` to `UFloat`.
    It allows setting a value within specified boundaries. If the set value is
    outside these boundaries, an `InvalidBoundaryError` is raised.

    Attributes:
        default (Optional[UFloat]): The default value for the parameter.
        min (Optional[Ufloat]): The minimum nominal value for the parameter.
        max (Optional[Ufloat]): The maximum nominal value for the parameter.
    """

    default: Optional[UFloat] = field(init=True, repr=True, default=None)
    min: Optional[UFloat] = field(init=True, repr=True, default=None)
    max: Optional[UFloat] = field(init=True, repr=True, default=None)

    @property
    def param_type(self) -> Type[Any]:
        """Returns the parameter type (`UFloat`)."""
        return UFloat

    def __post_init__(self):
        validate_name(name=self.name)
        validate_range_types(
            min_value=self.min, max_value=self.max, expected_type=self.param_type  # type: ignore # noqa: E501
        )
        validate_range_values(min_value=self.min, max_value=self.max)  # type: ignore
        if self.default is not None:
            self.set_value(self.default)

    def set_value(self, value: Optional[UFloat]) -> None:
        """
        Sets the value of the UFloat parameter after validating its type and range.

        Args:
            value (Optional[UFloat]): The value to be set for the parameter.
                                      None is allowed to unset.

        Raises:
            InvalidParameterTypeError: If the value's type does not match `UFloat`.
            InvalidBoundaryError: If the nominal value is outside the allowed range
                                  (if 'min' or 'max' are defined).
        """
        if value is not None:
            validate_type(item=value, expected_type=self.param_type)
            validate_range_application(
                item=value, min_value=self.min, max_value=self.max  # type: ignore
            )
        self._value = value


@dataclass
class IterableParameter(Parameter):
    """
    Represents an iterable parameter.

    Inherits from the abstract `Parameter` class and sets `param_type` to `Iterable`.

    Attributes:
        default (Optional[Iterable]): The default value for the parameter.
    """

    default: Optional[Iterable[Any]] = field(init=True, repr=True, default_factory=list)

    @property
    def param_type(self) -> Type[Any]:
        return Iterable

    def __post_init__(self):
        validate_name(name=self.name)

        if self.default is not None:
            self.set_value(self.default)

    def set_value(self, value: Optional[Iterable[Any]]) -> None:
        """
        Sets the value of the Iterable parameter after validating its type and range.

        Args:
            value (Optional[Iterable]): The value to be set for the parameter.
                                      None is allowed to unset.

        Raises:
            InvalidParameterTypeError: If the value's type does not match `Iterable`.
        """
        if value is not None:
            validate_type(item=value, expected_type=self.param_type)
        self._value = value


@dataclass
class MappingParameter(Parameter):
    """
    Represents a mapping parameter.

    Inherits from the abstract `Parameter` class and sets `param_type` to `Dict`.

    Attributes:
        default (Optional[Dict]): The default value for the parameter.
    """

    default: Optional[Dict[Any, Any]] = field(
        init=True, repr=True, default_factory=dict
    )

    @property
    def param_type(self) -> Type[Any]:
        return Dict

    def __post_init__(self):
        validate_name(name=self.name)

        if self.default is not None:
            self.set_value(self.default)

    def set_value(self, value: Optional[Iterable[Any]]) -> None:
        """
        Sets the value of the Mapping parameter after validating its type and range.

        Args:
            value (Optional[Dict]): The value to be set for the parameter.
                                      None is allowed to unset.

        Raises:
            InvalidParameterTypeError: If the value's type does not match `Dict`.
        """
        if value is not None:
            validate_type(item=value, expected_type=self.param_type)
        self._value = value


parameter_type_catalogue = Catalogue(item_type=Parameter)
parameter_type_catalogue.register(name="int", item=IntParameter)
parameter_type_catalogue.register(name="float", item=FloatParameter)
parameter_type_catalogue.register(name="str", item=StrParameter)
parameter_type_catalogue.register(name="bool", item=BoolParameter)
parameter_type_catalogue.register(name="ufloat", item=UFloatParameter)
parameter_type_catalogue.register(name="iterable", item=IterableParameter)
parameter_type_catalogue.register(name="mapping", item=IterableParameter)


@dataclass
class ParameterManager:
    """A manager that creates and maintains a mapping of parameters based on provided definitions."""  # noqa: E501

    parameter_definitions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    parameters_map: Dict[str, Parameter] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Post-initialization processing to prepare the ParameterManager."""
        self._validate_parameter_definitions()
        self.parameters_map = self._create_parameters_map()

    def _validate_parameter_definitions(self):
        """Validates the provided parameter definitions."""
        for name, definition in self.parameter_definitions.items():
            if "type" not in definition:
                raise ParameterDefinitionError(
                    f"Parameter '{name}' is missing a type definition."
                )
            if definition["type"] not in parameter_type_catalogue.list_items():
                raise ParameterDefinitionError(
                    f"No Parameter class registered for type '{definition['type']}'."
                )

    def _create_parameters_map(self) -> Dict[str, Parameter]:
        """Creates a map from parameter names to Parameter instances."""
        parameters_map = {}
        for name, definition in self.parameter_definitions.items():
            parameter_cls = parameter_type_catalogue.get(definition["type"])
            kwargs = {"name": name}
            kwargs.update({k: v for k, v in definition.items() if k != "type"})
            parameters_map[name] = parameter_cls(**kwargs)
        return parameters_map

    def set_parameter_attributes(self, name: str, attributes: Dict[str, Any]) -> None:
        """Updates properties of a parameter if it exists, excluding its value."""
        if name not in self.parameters_map:
            raise ParameterNotFoundError(
                f"Parameter '{name}' does not exist in among the "
                f"defined parameters: {list(self.parameters_map.keys())}."
            )

        current_parameter = self.parameters_map[name]
        current_properties = vars(current_parameter)

        if "value" in attributes:
            warnings.warn(
                f"Attempt to update 'value' attribute for parameter '{name}' ignored."
                f" Use set_parameter_value() instead.",
                ParameterUpdateAttributeWarning,
            )
            del attributes["value"]
        if "_value" in current_properties:
            del current_properties["_value"]

        try:
            updated_properties = {**current_properties, **attributes}
            parameter_type = type(current_parameter)
            updated_parameter = parameter_type(**updated_properties)
            self.parameters_map[name] = updated_parameter
        except TypeError as e:
            raise InvalidParameterAttributeError(
                f"Invalid attribute for parameter '{name}': {str(e)}"
            )

    def set_parameter_value(self, name: str, value: Any) -> None:
        """Sets the value of a parameter if it exists."""
        if name not in self.parameters_map:
            raise ParameterNotFoundError(
                f"Parameter '{name}' does not exist in among the "
                f"defined parameters: {list(self.parameters_map.keys())}."
            )

        try:
            self.parameters_map[name].set_value(value=value)
        except Exception as e:
            # Add context to the error message while preserving
            # the original error type and message
            error_type = type(e)
            original_message = str(e)
            context_message = f"Error in parameter '{name}'"
            new_message = f"{context_message}: {original_message}"
            raise error_type(new_message) from None

        """

        try:
            self.parameters_map[name].set_value(value=value)
        except (
                InvalidParameterTypeError,
                InvalidBoundaryError,
                InvalidAcceptedValuesError,
        ) as e:
            raise InvalidParameterAttributeError(
                f"Invalid value for parameter '{name}': {str(e)}"
            )
        """

    def get_parameters_values(self) -> Dict[str, Any]:
        """
        Returns a dictionary with parameter names as keys
        and their current values as values.

        Returns:
            Dict[str, Any]: A dictionary of parameter
            names and their current values.
        """
        return {name: param.value for name, param in self.parameters_map.items()}

    def set_parameters_values(self, values_dict: Dict[str, Any]) -> None:
        """
        Sets the values of multiple parameters based on the provided dictionary.

        Args:
            values_dict (Dict[str, Any]): A dictionary where keys are parameter names
                                          and values are the new values to be set.

        Raises:
            ParameterSettingError: If any parameter in the dictionary is not recognized.
            ParameterSettingWarning: If not all parameters are being set.
            InvalidParameterTypeError: If a value's type does not match the parameter's
                                       expected type.
            InvalidBoundaryError: If a value is outside the parameter's allowed range.
            InvalidAcceptedValuesError: If a value is not in the list of accepted values.

        """  # noqa: E501

        for name, value in values_dict.items():
            self.set_parameter_value(name, value)

    def check_provided_parameters_values(self, values_dict: Dict[str, Any]) -> None:
        expected_parameter_names = set(self.parameters_map.keys())
        provided_parameter_names = set(values_dict.keys())
        missing_parameters = expected_parameter_names - provided_parameter_names
        extra_parameters = provided_parameter_names - expected_parameter_names
        if len(extra_parameters) > 0:
            raise ParameterSettingError(
                f"Attempting to set unrecognized parameter(s): {list(extra_parameters)}"
            )
        if len(missing_parameters) > 0:
            warnings.warn("Not all parameters are being set", ParameterSettingWarning)

    def check_parameters_values_none(self):
        """Ensures that all parameters have a non-None value.

        Raises:
            ParameterValueNotSet: If any  parameter is not set.
        """
        parameter_values = self.get_parameters_values()
        unset_parameter_values = {
            name: value for name, value in parameter_values.items() if value is None
        }
        if len(unset_parameter_values) > 0:
            raise ParameterValueNotSet(
                f"All parameters must be set (non-None) "
                f"before computation. Please set the value of "
                f"{list(unset_parameter_values.keys())}"
            )
