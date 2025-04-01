from typing import TypeVar

from pumas.uncertainty_management.distributions.models import DistributionValue
from pumas.uncertainty_management.uncertainties.models import UncertainValue
from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import UFloat

T = TypeVar("T", float, UFloat, UncertainValue, DistributionValue)
R = TypeVar("R", float, UFloat, UncertainValue, DistributionValue)
