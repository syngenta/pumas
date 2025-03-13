# type: ignore
from typing import Any, Dict

from pumas.aggregation import aggregation_catalogue
from pumas.aggregation.base_models import Aggregation
from pumas.desirability import desirability_catalogue
from pumas.desirability.base_models import Desirability
from pumas.scoring_profile.scoring_profile import Profile


class ScoringFunction:
    def __init__(self, profile: Profile):
        self.profile = profile
        self.desirability_functions = self._initialize_desirability_functions()
        self.aggregation_function = self._initialize_aggregation_function()

    def _initialize_desirability_functions(self) -> Dict[str, Desirability]:
        functions = {}
        for objective in self.profile.objectives:
            desirability_class = desirability_catalogue.get(
                objective.desirability_function.name
            )
            desirability_instance = desirability_class(
                params=objective.desirability_function.parameters
            )
            functions[objective.name] = desirability_instance
        return functions

    def _initialize_aggregation_function(self) -> Aggregation:
        aggregation_class = aggregation_catalogue.get(
            self.profile.aggregation_function.name
        )
        aggregation_instance = aggregation_class(
            params=self.profile.aggregation_function.parameters or {}
        )
        return aggregation_instance

    def compute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # self._validate_input_data(data)
        desirability_scores = {}
        for objective in self.profile.objectives:
            desirability = self.desirability_functions.get(objective.name)
            x = data.get(objective.name)

            try:
                desirability_scores[objective.name] = desirability.compute_numeric(x=x)
            except Exception as e:
                raise ValueError(
                    f"Error computing desirability for {objective.name}: {str(e)}"
                )

        values = [desirability_scores[obj.name] for obj in self.profile.objectives]
        weights = [obj.weight for obj in self.profile.objectives]

        try:
            aggregated_score = self.aggregation_function.compute_numeric(
                values=values, weights=weights
            )
        except Exception as e:
            raise ValueError(f"Error computing aggregated score: {str(e)}")

        return {
            "aggregated_score": aggregated_score,
            "desirability_scores": desirability_scores,
        }

    def _validate_input_data(self, data: Dict[str, Any]) -> None:
        expected_keys = set(obj.name for obj in self.profile.objectives)
        actual_keys = set(data.keys())
        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            error_msg = []
            if missing:
                error_msg.append(f"Missing keys: {missing}")
            if extra:
                error_msg.append(f"Unexpected keys: {extra}")
            # raise ValueError(", ".join(error_msg))

        for key, value in data.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"Value for {key} must be a number, got {type(value)}")
