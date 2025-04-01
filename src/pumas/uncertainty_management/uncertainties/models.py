from typing import Dict

from pydantic import BaseModel

from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import (
    UFloat,
    ufloat,
)


class UncertainValue(BaseModel):
    """Internal model for values with uncertainty"""

    nominal_value: float
    std_dev: float

    def to_ufloat(self) -> UFloat:
        """Convert to uncertainties.ufloat"""
        uf: UFloat = ufloat(nominal_value=self.nominal_value, std_dev=self.std_dev)
        return uf

    @classmethod
    def from_ufloat(cls, uf: UFloat) -> "UncertainValue":
        """Convert from uncertainties.ufloat"""
        return cls(nominal_value=uf.nominal_value, std_dev=uf.std_dev)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "UncertainValue":
        return cls(nominal_value=data["nominal_value"], std_dev=data["std"])
