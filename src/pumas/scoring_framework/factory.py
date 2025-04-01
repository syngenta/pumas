# strategies.py
from enum import Enum
from typing import Any, Dict, Union, cast

from pumas.scoring_framework.models import InputData
from pumas.scoring_framework.numeric_foerp import NumericFOERPScoringStrategy
from pumas.scoring_framework.numeric_noerp import NumericScoringStrategy
from pumas.scoring_profile.scoring_profile import ScoringProfile
from pumas.uncertainty_management.distributions.models import DistributionValue
from pumas.uncertainty_management.uncertainties.models import UncertainValue
from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import (
    UFloat,
    ufloat,
)


class StrategyType(Enum):
    NUMERIC = "numeric"
    FOERP = "foerp"


class ScoringStrategyFactory:
    INPUT_TYPES = {StrategyType.NUMERIC: float, StrategyType.FOERP: UFloat}

    STRATEGY_CLASSES = {
        StrategyType.NUMERIC: NumericScoringStrategy,
        StrategyType.FOERP: NumericFOERPScoringStrategy,
    }

    @classmethod
    def create_input_data(
        cls,
        strategy_type: StrategyType,
        data: Dict[str, Dict[str, Any]],
    ) -> InputData[Any]:
        input_type = cls.INPUT_TYPES[strategy_type]

        converted_data: Union[
            Dict[str, Dict[str, DistributionValue]],
            Dict[str, Dict[str, UFloat]],
            Dict[str, Dict[str, float]],
            Dict[str, Dict[str, Any]],
        ]

        if input_type == DistributionValue:
            converted_data = cls._convert_to_distribution_values(data)
        elif input_type == UFloat:
            converted_data = cls._convert_to_ufloat_values(data)

        elif input_type == float:
            converted_data = data
        else:
            converted_data = data

        return InputData[Any](data=converted_data)

    @classmethod
    def create_strategy(
        cls, strategy_type: StrategyType, profile: ScoringProfile
    ) -> Union[NumericScoringStrategy, NumericFOERPScoringStrategy]:
        strategy_class = cls.STRATEGY_CLASSES.get(strategy_type)
        if not strategy_class:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")
        return cast(
            Union[NumericScoringStrategy, NumericFOERPScoringStrategy],
            strategy_class(profile),
        )

    @staticmethod
    def _convert_to_uncertain_values(
        data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, UncertainValue]]:
        return {
            uid: {
                obj_name: (
                    UncertainValue.from_dict(value)
                    if isinstance(value, dict)
                    else value
                )
                for obj_name, value in objectives.items()
            }
            for uid, objectives in data.items()
        }

    @staticmethod
    def _convert_to_ufloat_values(
        data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, UncertainValue]]:
        return {
            uid: {obj_name: ufloat(**value) for obj_name, value in objectives.items()}
            for uid, objectives in data.items()
        }

    @staticmethod
    def _convert_to_distribution_values(
        data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, DistributionValue]]:
        return {
            uid: {
                obj_name: (
                    DistributionValue(**obj_value)
                    if isinstance(obj_value, dict)
                    else obj_value
                )
                for obj_name, obj_value in objectives.items()
            }
            for uid, objectives in data.items()
        }
