from abc import ABC, abstractmethod
from typing import Optional

from pumas.aggregation import aggregation_catalogue as default_aggregation_catalogue
from pumas.architecture.catalogue import Catalogue
from pumas.dataframes.dataframe import DataFrame
from pumas.desirability import desirability_catalogue as default_desirability_catalogue
from pumas.framework.exceptions import (
    DesirabilityInitializationError,
    ScorerMissingColumnError,
)
from pumas.scoring_profile.models import Profile


class BaseFramework(ABC):
    def __init__(
        self,
        properties: DataFrame,
        scoring_profile: Profile,
        desirability_catalogue: Optional[Catalogue] = None,
        aggregation_catalogue: Optional[Catalogue] = None,
    ):
        self._properties_dataframe = properties
        self._scoring_profile = scoring_profile

        self._desirability_catalogue = (
            desirability_catalogue or default_desirability_catalogue
        )
        self._aggregation_catalogue = (
            aggregation_catalogue or default_aggregation_catalogue
        )

        self._desirability_functions_map = {}

        self._check_identifiers()
        self._validate_desirability_functions()
        self._validate_aggregation_method()
        self._build_desirability_functions_map()

        self.desirability_dataframe = None
        self.aggregated_dataframe = None

    @property
    def properties_dataframe(self):
        return self._properties_dataframe

    def _check_identifiers(self):
        expected_identifiers = [
            objective.name for objective in self._scoring_profile.objectives
        ]
        available_identifiers = self._properties_dataframe.columns
        missing_identifiers = list(
            set(expected_identifiers) - set(available_identifiers)
        )

        if missing_identifiers:
            raise ScorerMissingColumnError(
                "Missing necessary columns to compute desirability scores."
            )

    def _validate_desirability_functions(self):
        missing_functions = []
        for property_def in self._scoring_profile.objectives:
            if (
                property_def.desirability_function.name
                not in self._desirability_catalogue.list_items()
            ):
                missing_functions.append(property_def.desirability_function.name)
        if missing_functions:
            raise DesirabilityInitializationError(
                f"Desirability function(s) not found: {missing_functions}"
            )

    def _validate_aggregation_method(self):
        if (
            self._scoring_profile.aggregation_function.name
            not in self._aggregation_catalogue.list_items()
        ):
            raise DesirabilityInitializationError(
                f"Aggregation function "
                f"'{self._scoring_profile.aggregation_function.name}' not found."
            )

    def _build_desirability_functions_map(self):
        """
        Retrieve the necessary desirability functions
        from the catalogue and map them against the objectives
        """
        for objective in self._scoring_profile.objectives:
            # get the desirability function from the catalogue
            desirability_function = self._desirability_catalogue.get(
                objective.desirability_function.name
            )()
            # set the parameters of the desirability function
            desirability_function.set_parameters_values(
                values_dict=objective.desirability_function.parameters
            )
            self._desirability_functions_map[objective.name] = desirability_function

    @abstractmethod
    def compute(self): ...
