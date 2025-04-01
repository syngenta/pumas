from typing import Any, Dict, List, Union

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
from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import UFloat


class NumericFOERPScoringStrategy(BaseScoringStrategy[UFloat, UFloat]):
    @staticmethod
    def _desirability_wrapper(
        d: Desirability,
    ) -> TypedDesirability[UFloat, UFloat]:
        return TypedDesirability[UFloat, UFloat](
            d, NumericFOERPScoringStrategy._compute_desirability
        )

    @staticmethod
    def _aggregation_wrapper(a: Aggregation) -> TypedAggregation[UFloat]:
        return TypedAggregation[UFloat](
            a, NumericFOERPScoringStrategy._compute_aggregation
        )

    @staticmethod
    def _compute_desirability(des: Desirability, val: UFloat) -> UFloat:
        return des.compute_ufloat(x=val)

    @staticmethod
    def _compute_aggregation(
        agg: Aggregation,
        vals: List[Union[UFloat, None]],
        weights: List[Union[float, None]],
    ) -> UFloat:
        return agg.compute_ufloat(
            [v for v in vals if v is not None],
            weights,
        )

    def _create_scoring_function(
        self, profile: ScoringProfile
    ) -> ScoringFunction[UFloat, UFloat]:
        return ScoringFunction[UFloat, UFloat](
            profile,
            self._desirability_wrapper,
            self._aggregation_wrapper,
        )

    def serial_process_with_uid(
        self, data: Dict[str, ObjectPropertiesMap[UFloat]]
    ) -> Dict[str, ScoringResult[UFloat]]:
        results = {
            uid: self.scoring_function.compute(object_properties_map=object_data)
            for uid, object_data in data.items()
        }
        return results

    def compute(
        self,
        input_data: InputData[UFloat],
        chunk_size: int = 100,
        n_jobs: int = -1,
        **compute_options: Dict[str, Any],
    ) -> ScoringResults[UFloat]:
        self._validate_input(input_data)

        results = self.serial_process_with_uid(data=input_data.data)

        return ScoringResults[UFloat](results=results)
