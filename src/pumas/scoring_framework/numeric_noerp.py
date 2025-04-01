from typing import Any, Dict, List, Optional

from pumas.aggregation.base_models import Aggregation
from pumas.desirability.base_models import Desirability
from pumas.scoring_framework.base_models import BaseScoringStrategy
from pumas.scoring_framework.models import (
    InputData,
    ObjectPropertiesMap,
    ScoringResult,
    ScoringResults,
)
from pumas.scoring_framework.scoring_function import (
    ScoringFunction,
    TypedAggregation,
    TypedDesirability,
)
from pumas.scoring_profile.scoring_profile import ScoringProfile


class NumericScoringStrategy(BaseScoringStrategy[float, float]):
    @staticmethod
    def _desirability_wrapper(d: Desirability) -> TypedDesirability[float, float]:
        return TypedDesirability[float, float](
            d, NumericScoringStrategy._compute_desirability
        )

    @staticmethod
    def _aggregation_wrapper(a: Aggregation) -> TypedAggregation[float]:
        return TypedAggregation[float](a, NumericScoringStrategy._compute_aggregation)

    @staticmethod
    def _compute_desirability(des: Desirability, val: float) -> float:
        return des.compute_numeric(val)

    @staticmethod
    def _compute_aggregation(
        agg: Aggregation,
        vals: List[Optional[float]],
        weights: List[Optional[float]],
    ) -> float:
        return agg.compute_numeric(vals, weights)

    def _create_scoring_function(
        self, profile: ScoringProfile
    ) -> ScoringFunction[float, float]:
        return ScoringFunction[float, float](
            profile,
            self._desirability_wrapper,
            self._aggregation_wrapper,
        )

    def serial_process_with_uid(
        self, data: Dict[str, ObjectPropertiesMap[float]]
    ) -> Dict[str, ScoringResult[float]]:
        all_results = {
            uid: self.scoring_function.compute(object_properties_map=object_data)
            for uid, object_data in data.items()
        }
        return all_results

    def compute(
        self,
        input_data: InputData[float],
        chunk_size: int = 100,
        n_jobs: int = -1,
        **compute_options: Dict[str, Any],
    ) -> ScoringResults[float]:

        self._validate_input(input_data)

        results = self.serial_process_with_uid(data=input_data.data)

        return ScoringResults[float](results=results)
