from abc import ABC, abstractmethod
from typing import Any, Dict, Generic

from pumas.scoring_framework.models import InputData, ScoringResults
from pumas.scoring_framework.scoring_function import ScoringFunction
from pumas.scoring_framework.type_definitions import R, T
from pumas.scoring_profile.scoring_profile import ScoringProfile


class BaseScoringStrategy(Generic[T, R], ABC):
    def __init__(self, profile: ScoringProfile):
        self.profile = profile
        self.scoring_function: ScoringFunction[T, R] = self._create_scoring_function(
            profile
        )

    @abstractmethod
    def _create_scoring_function(
        self, profile: ScoringProfile
    ) -> ScoringFunction[T, R]:
        pass

    @abstractmethod
    def compute(
        self,
        input_data: InputData[T],
        chunk_size: int = 100,
        n_jobs: int = -1,
        **compute_options: Dict[str, Any],
    ) -> ScoringResults[R]:
        pass

    def _validate_input(self, data: InputData[T]) -> None:
        required_objectives = [obj.name for obj in self.profile.objectives]
        if not data.validate_objectives(required_objectives):
            raise ValueError(
                f"Input data missing required objectives: {required_objectives}"
            )
