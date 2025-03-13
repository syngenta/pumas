import warnings
from abc import ABC
from typing import Any, Dict, Optional, Tuple, Type, Union

from pumas.architecture.exceptions import (
    InvalidParameterTypeError,
    ParameterSettingError,
    ParameterSettingWarning,
    ParameterValueNotSet,
)
from pumas.architecture.parameters import ParameterManager


class AbstractParametrizedStrategy(ABC):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.parameter_manager: ParameterManager = ParameterManager()
        self._params = params

        self._check_parameters_values_none()

    @property
    def parameters_map(self) -> Dict[str, Any]:
        return self.parameter_manager.parameters_map

    def _set_parameter_definitions(
        self, parameter_definitions: Dict[str, Dict[str, Any]]
    ) -> None:
        self.parameter_manager = ParameterManager(parameter_definitions)
        if self._params:
            self._validate_and_set_parameters(self._params)

    def _validate_and_set_parameters(self, params: Optional[Dict[str, Any]]) -> None:
        if params:
            self.set_parameters_values(params)

    def get_parameters_values(self) -> Dict[str, Any]:
        return self.parameter_manager.get_parameters_values()

    def set_parameters_values(self, values_dict: Dict[str, Any]) -> None:
        if values_dict and not self.parameter_manager.parameters_map:
            warnings.warn(
                "This strategy does not accept parameters. "
                "The provided parameters will be ignored.",
                ParameterSettingWarning,
            )
            return

        self.parameter_manager.set_parameters_values(values_dict=values_dict)

    def set_parameters_attributes(
        self, attributes_map: Dict[str, Dict[str, Any]]
    ) -> None:
        if not self.parameter_manager.parameters_map:
            warnings.warn(
                "This strategy does not accept parameters. "
                "The provided attributes will be ignored.",
                ParameterSettingWarning,
            )
            return

        if not all(
            param in self.parameter_manager.parameters_map for param in attributes_map
        ):
            raise ParameterSettingError(
                "Attempting to set attributes for unrecognized parameter(s)"
            )

        if len(attributes_map) != len(self.parameter_manager.parameters_map):
            warnings.warn(
                "Not setting attributes for all parameters", ParameterSettingWarning
            )

        for param_name, attributes in attributes_map.items():
            try:
                self.parameter_manager.set_parameter_attributes(param_name, attributes)
            except Exception as e:
                raise ParameterSettingError(
                    f"Error setting attributes for parameter '{param_name}': {str(e)}"
                )

    def _get_parameter_value(self, name: str) -> Any:
        values = self.get_parameters_values()
        if name not in values:
            raise ParameterValueNotSet(f"Parameter '{name}' has not been set.")
        return values[name]

    @staticmethod
    def _validate_compute_input(
        item: Any, expected_type: Union[Type[Any], Tuple[Type[Any], ...]]
    ) -> None:
        if not isinstance(item, expected_type):
            raise InvalidParameterTypeError(
                f"Expected {expected_type.__name__ if isinstance(expected_type, type) else expected_type}, "  # noqa: E501
                f"got {type(item).__name__} instead."
            )

    def _check_parameters_values_none(self):
        """Ensures that all parameters are set before computation.

        Raises:
            ParameterValueNotSet: If any  parameter is not set.
        """
        self.parameter_manager.check_parameters_values_none()
