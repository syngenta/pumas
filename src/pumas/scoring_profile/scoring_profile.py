from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from pumas.aggregation import aggregation_catalogue
from pumas.desirability import desirability_catalogue
from pumas.desirability.base_models import Desirability


class DesirabilityFunction(BaseModel):
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    def validate_desirability_function_name(cls, v: str) -> str:
        valid_functions = desirability_catalogue.list_items()
        if v not in valid_functions:
            raise ValueError(
                f"Unknown desirability function: "
                f"'{v}'. Valid options are: {', '.join(valid_functions)}"
            )
        return v

    @field_validator("parameters")
    def validate_desirability_function_parameters(
        cls, parameters: Dict[str, Any], info: ValidationInfo
    ) -> Dict[str, Any]:
        name = info.data.get("name")
        if not name:
            raise ValueError("Name must be provided before parameters can be validated")

        desirability_class = desirability_catalogue.get(name)

        try:
            desirability: Desirability = desirability_class(params=parameters)
            desirability._check_parameters_values_none()
        except Exception as e:
            raise ValueError(
                f"Invalid parameters for desirability function '{name}': {str(e)}"
            )
        return parameters


class Objective(BaseModel):
    name: str
    desirability_function: DesirabilityFunction
    weight: Optional[float] = None
    # value_type: Optional[Literal["float", "str", "bool"]] = None
    # kind: Optional[Literal["numerical", "categorical"]] = None


class AggregationFunction(BaseModel):
    name: str
    parameters: Optional[Dict[str, Any]] = None

    @field_validator("name")
    def validate_aggregation_function(cls, v: str) -> str:
        valid_functions = aggregation_catalogue.list_items()
        if v not in valid_functions:
            raise ValueError(
                f"Unknown aggregation function: '{v}'. "
                f"Valid options are: {', '.join(valid_functions)}"
            )
        return v

    model_config = {"extra": "forbid"}


class ScoringProfile(BaseModel):
    objectives: List[Objective]
    aggregation_function: AggregationFunction

    @model_validator(mode="after")
    def validate_weights(self) -> "ScoringProfile":
        weights = [obj.weight for obj in self.objectives if obj.weight is not None]

        if len(weights) != 0 and len(weights) != len(self.objectives):
            raise ValueError("Either all objectives have weights, or none have weights")

        return self

    @field_validator("objectives")
    def validate_unique_objective_names(cls, v: List[Objective]) -> List[Objective]:
        names = [obj.name for obj in v]
        if len(names) != len(set(names)):
            raise ValueError("Objective names must be unique")
        return v

    model_config = {"extra": "forbid"}

    def write_to_file(self, file_path: Union[Path, str]) -> None:
        """
        Writes a scoring profile to a JSON file.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        with file_path.open("w") as file:
            file.write(self.model_dump_json(indent=2))

    @classmethod
    def read_from_file(cls, file_path: Union[Path, str]) -> "ScoringProfile":
        """
        Reads a scoring profile from a JSON file.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        with file_path.open("r") as file:
            profile = ScoringProfile.model_validate_json(file.read())
        return profile
