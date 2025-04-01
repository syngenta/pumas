from typing import Callable, Dict, Generic, List, Optional, Sequence, Type, Union, cast

from pumas.aggregation import aggregation_catalogue
from pumas.aggregation.base_models import Aggregation
from pumas.desirability import desirability_catalogue
from pumas.desirability.base_models import Desirability
from pumas.scoring_framework.models import ObjectPropertiesMap, ScoringResult
from pumas.scoring_framework.type_definitions import R, T
from pumas.scoring_profile.scoring_profile import (
    AggregationFunction,
    Objective,
    ScoringProfile,
)


class TypedDesirability(Generic[T, R]):
    def __init__(
        self,
        desirability: Desirability,
        compute_method: Callable[[Desirability, T], R],
    ):
        self.desirability: Desirability = desirability
        self.compute_method: Callable[[Desirability, T], R] = compute_method

    def compute(self, value: T) -> R:
        return self.compute_method(self.desirability, value)


class TypedAggregation(Generic[R]):
    def __init__(
        self,
        aggregation: Aggregation,
        compute_method: Callable[
            [Aggregation, List[Optional[R]], List[Optional[float]]], Optional[R]
        ],
    ):
        self.aggregation: Aggregation = aggregation
        self.compute_method: Callable[
            [Aggregation, List[Optional[R]], List[Optional[float]]], Optional[R]
        ] = compute_method

    def compute(
        self, values: List[Optional[R]], weights: List[Optional[float]]
    ) -> Optional[R]:
        return self.compute_method(self.aggregation, values, weights)


class DesirabilityFunctionFactory:
    @staticmethod
    def create(objective: Objective) -> Desirability:
        desirability_class = cast(
            Type[Desirability],
            desirability_catalogue.get(objective.desirability_function.name),
        )
        return desirability_class(params=objective.desirability_function.parameters)

    @classmethod
    def create_all(cls, objectives: Sequence[Objective]) -> Dict[str, Desirability]:
        return {obj.name: cls.create(obj) for obj in objectives}


class AggregationFunctionFactory:
    @staticmethod
    def create(aggregation_function: AggregationFunction) -> Aggregation:
        aggregation_class = cast(
            Type[Aggregation], aggregation_catalogue.get(aggregation_function.name)
        )
        return aggregation_class(params=aggregation_function.parameters or {})


class ScoreComputer(Generic[T, R]):
    def __init__(
        self,
        desirability_functions: Dict[str, TypedDesirability[T, R]],
        aggregation_function: TypedAggregation[R],
        objectives: Sequence[Objective],
    ):
        self.desirability_functions: Dict[str, TypedDesirability[T, R]] = (
            desirability_functions
        )
        self.aggregation_function: TypedAggregation[R] = aggregation_function
        self.objectives: Sequence[Objective] = objectives

    def compute(self, data: ObjectPropertiesMap[T]) -> ScoringResult[R]:
        desirability_scores = self._compute_desirability_scores(data)
        aggregated_score = self._compute_aggregated_score(desirability_scores)
        return ScoringResult[R](
            aggregated_score=aggregated_score, desirability_scores=desirability_scores
        )

    def _compute_desirability_scores(
        self, data: ObjectPropertiesMap[T]
    ) -> Dict[str, Optional[R]]:
        return {
            name: self._compute_single_desirability(name, data.get(name))
            for name in self.desirability_functions
        }

    def _compute_single_desirability(
        self, name: str, value: Optional[T]
    ) -> Optional[R]:
        if value is None:
            return None
        return self.desirability_functions[name].compute(value)

    def _compute_aggregated_score(
        self, desirability_scores: Dict[str, Optional[R]]
    ) -> Optional[R]:
        values = [desirability_scores.get(obj.name) for obj in self.objectives]
        weights = [obj.weight for obj in self.objectives]
        return self.aggregation_function.compute(values, weights)


class ScoringFunction(Generic[T, R]):
    def __init__(
        self,
        profile: ScoringProfile,
        desirability_wrapper: Callable[[Desirability], TypedDesirability[T, R]],
        aggregation_wrapper: Callable[[Aggregation], TypedAggregation[R]],
    ):
        self.profile: ScoringProfile = profile
        self.desirability_functions: Dict[str, TypedDesirability[T, R]] = {
            obj.name: desirability_wrapper(DesirabilityFunctionFactory.create(obj))
            for obj in profile.objectives
        }
        self.aggregation_function: TypedAggregation[R] = aggregation_wrapper(
            AggregationFunctionFactory.create(profile.aggregation_function)
        )
        self.score_computer: ScoreComputer[T, R] = ScoreComputer[T, R](
            self.desirability_functions, self.aggregation_function, profile.objectives
        )

    def compute(
        self, object_properties_map: Union[ObjectPropertiesMap[T]]
    ) -> ScoringResult[R]:

        return self.score_computer.compute(object_properties_map)
