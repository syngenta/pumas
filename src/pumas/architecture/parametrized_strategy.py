import warnings
from abc import ABC
from functools import partial
from typing import Any, Callable, Dict, List

from pumas.architecture.exceptions import (
    ParameterDefinitionError,
    ParameterOverlapError,
    ParameterSettingError,
    ParameterSettingWarning,
    ParameterValueNotSet,
)
from pumas.architecture.parameters import Parameter, ParameterManager


class AbstractParametrizedStrategy(ABC):
    """Represents an abstract base class for desirability functions.

    This class is used to define the structure of desirability functions
    which include utility functions, coefficient parameters, and input parameters.

    Attributes:
        utility_function (Callable): A callable representing the utility function
            used to compute desirability.
        input_parameters_names (List[str]): A list of names of parameters
            that are used as inputs in the utility function.
        coefficient_parameters_names (List[str]): A list of names of parameters
            that are considered as coefficients in the utility function.

    Raises:
        ParameterOverlapError: If a parameter name is both in coefficient and
            input parameter names.
        ParameterDefinitionError: If there's a mismatch between defined parameters
            and the ones required by the utility function.
    """

    def __init__(
        self,
        utility_function: Callable,
        coefficient_parameters_names: List[str],
        input_parameters_names: List[str],
    ):
        self.utility_function: Callable = utility_function
        self._coefficient_parameters_names: List[str] = coefficient_parameters_names
        self._input_parameters_names: List[str] = input_parameters_names

        self._parameter_manager = ParameterManager(input_function=self.utility_function)

        self._parameters_map: Dict[str, Parameter] = (
            self._parameter_manager.parameters_map
        )

        self._validate_parameters()

    @property
    def input_parameters_names(self) -> List[str]:
        """Returns the list of names of input parameters."""
        return self._input_parameters_names

    @property
    def coefficient_parameters_names(self) -> List[str]:
        """Returns the list of names of coefficient parameters."""
        return self._coefficient_parameters_names

    @property
    def parameters_map(self) -> Dict[str, Parameter]:
        """Returns the dictionary of parameter names and Parameter objects."""
        return self._parameter_manager.parameters_map

    @property
    def input_parameters_map(self) -> Dict[str, Parameter]:
        return {
            name: self._parameters_map[name] for name in self.input_parameters_names
        }

    @property
    def coefficient_parameters_map(self) -> Dict[str, Parameter]:
        return {
            name: self._parameters_map[name]
            for name in self.coefficient_parameters_names
        }

    def _validate_parameters(self):
        """Validates parameter definitions ensuring no overlap and correct definition.

        This method calls helpers to check for overlapping parameter names
        and the presence of all required parameter names.

        Raises:
            ParameterOverlapError: If parameter names are defined in both coefficient
                and input parameter lists.
            ParameterDefinitionError: If there's a mismatch between required parameters
                by the utility function and those provided in the initialization.
        """
        self._check_parameter_overlap()
        self._check_parameter_definition()

    def _check_parameter_overlap(self):
        overlap = set(self._coefficient_parameters_names) & set(
            self._input_parameters_names
        )
        if overlap:
            raise ParameterOverlapError(
                f"Parameters cannot be both coefficients "
                f"and inputs: {', '.join(overlap)}"
            )

    def _check_parameter_definition(self):
        function_parameters = set(self._parameters_map.keys())
        provided_parameters = set(
            self._coefficient_parameters_names + self._input_parameters_names
        )
        mismatched_parameters_1 = function_parameters - provided_parameters
        mismatched_parameters_2 = provided_parameters - function_parameters
        if mismatched_parameters_1:
            raise ParameterDefinitionError(
                f"Some parameters required by the utility function are not defined "
                f"in the concrete Desirability class initialization: "
                f"{', '.join(list(mismatched_parameters_1))} \n"
                f" review the definition of coefficient_parameters_names "
                f"and input_parameters_names"
            )
        if mismatched_parameters_2:
            raise ParameterDefinitionError(
                f"Some parameters defined in the concrete Desirability class "
                f"initialization are not required by the utility"
                f" function: {', '.join(list(mismatched_parameters_2))} \n"
                f" review the definition of coefficient_parameters_names "
                f"and input_parameters_names"
            )

    def _check_coefficient_parameter_editing(self, parameter_names: List[str]):
        unknown_params = set(parameter_names) - set(
            self.coefficient_parameters_map.keys()
        )
        if unknown_params:
            raise ParameterSettingError(
                f"Unknown coefficient parameters: {', '.join(unknown_params)}"
            )

        unmodified_params = set(self.coefficient_parameters_map.keys()) - set(
            parameter_names
        )

        if unmodified_params:
            warnings.warn(
                ParameterSettingWarning(
                    f"The following coefficient parameters "
                    f"are not modified: {', '.join(unmodified_params)}"
                )
            )

    def _check_coefficient_parameters_values(self):
        """Ensures that all coefficient parameters are set before computation.

        Raises:
            ParameterValueNotSet: If any coefficient parameter is not set.
        """
        parameter_coefficient_values = self.get_coefficient_parameters_values()
        if any(value is None for value in parameter_coefficient_values.values()):
            raise ParameterValueNotSet(
                "All coefficient parameters must be set (non-None) before computation."
            )

    def set_coefficient_parameters_values(self, values_dict: Dict[str, Any]):
        """Sets values for the coefficient parameters of the utility function.

        This method takes a dictionary of coefficient parameter names and values
        and sets those values in the respective Parameter objects. If a parameter name
        provided does not exist, a ParameterSettingError is raised.
        Additionally, it warns if provided values do not modify
        all coefficient parameters.

        Args:
            values_dict (Dict[str, Any]): A dictionary mapping parameter names to
                their desired values. Each key-value pair corresponds
                to a parameter name and the value to set for it.

        Raises:
            ParameterSettingError: If a parameter in
                `values_dict` does not correspond to
                any known coefficient parameter.
            ParameterDefinitionError: Potentially raised by internal methods if
                parameter values are not consistent with parameter definitions.

        Warnings:
            ParameterSettingWarning: This is issued if there are coefficient parameters
                that are not being modified.

        """

        self._check_coefficient_parameter_editing(
            parameter_names=list(values_dict.keys())
        )
        # Set the values of the parameters through the parameter manager
        for name, new_value in values_dict.items():
            self._parameter_manager.set_parameter_value(name=name, value=new_value)

    def get_coefficient_parameters_values(self) -> Dict[str, Any]:
        """Retrieves the current values of all coefficient parameters.

        This method iterates over the coefficient parameters map and
        compiles a dictionary
        with parameter names as keys and their corresponding current values.
        If a parameter value is not yet set, it will be `None`.

        Note: The returned dictionary will include all coefficient parameter names
        defined during initialization regardless of whether their values have been set.

        Returns:
            Dict[str, Any]: A dictionary mapping each coefficient parameter name to its
            current value (or `None` if not set).

        Usage:
            coefficients = desirability_object.get_coefficient_parameters_values()
        """
        return {
            name: param.value for name, param in self.coefficient_parameters_map.items()
        }

    def set_coefficient_parameters_attributes(
        self, attributes_map: Dict[str, Dict[str, Any]]
    ) -> None:
        """Sets attributes for the coefficient parameters of the utility function.
        # Set the values of the parameters through the parameter manager"""
        self._check_coefficient_parameter_editing(
            parameter_names=list(attributes_map.keys())
        )

        for name, attributes in attributes_map.items():
            self._parameter_manager.set_parameter_attributes(
                name=name, attributes=attributes
            )

    @property
    def _get_partial_utility_function(self) -> Callable:
        """Creates a partial utility function with set coefficient parameter values.

        It verifies that coefficient parameters have been set and then creates a partial
        utility function by filling in these parameters. This partial function can be
        used to compute desirability by only needing to pass in the input parameters.

        As a private method, the approach used ensures that the coefficients
        are set before
        any computation takes place and reduces the risk of errors at runtime due to
        missing coefficients.

        Returns:
            Callable: The utility function with the coefficient parameters already set,
            ready to receive input parameter(s) and compute the desirability score.

        Raises:
            ParameterValueNotSet: If any of the coefficient
            parameters are not set (None).

        Usage:
            partial_function = self._get_partial_utility_function()
            desirability_score = partial_function(input_value)
        """
        self._check_coefficient_parameters_values()
        parameter_coefficient_values = self.get_coefficient_parameters_values()
        return partial(self.utility_function, **parameter_coefficient_values)
