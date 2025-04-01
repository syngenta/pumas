from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class DesirabilityFunction(BaseModel):
    name: str
    parameters: Dict[str, Any]


class Objective(BaseModel):
    name: str
    desirability_function: DesirabilityFunction
    weight: float
    value_type: Literal["float", "str", "bool"]
    kind: Literal["numerical", "categorical"]


class AggregationFunction(BaseModel):
    name: str
    parameters: Optional[Dict[str, Any]] = None

    class Config:
        extra = "forbid"


class Profile(BaseModel):
    objectives: List[Objective]
    aggregation_function: AggregationFunction

    class Config:
        extra = "forbid"
