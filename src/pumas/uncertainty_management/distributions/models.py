from typing import Dict, List, Union

from pydantic import BaseModel

from pumas.uncertainty_management.distributions.scipy_wrapper import stats


class DistributionValue(BaseModel):
    distribution_name: str
    shape_parameters: Dict[str, Union[float, List[float]]]

    @property
    def distribution(self) -> stats.rv_continuous:
        continuous_distribution: stats.rv_continuous = getattr(
            stats, self.distribution_name
        )(**self.shape_parameters)
        return continuous_distribution
