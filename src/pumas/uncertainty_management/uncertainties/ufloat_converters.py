from abc import ABC, abstractmethod

from pumas.architecture.catalogue import Catalogue
from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import (
    UFloat,
    ufloat,
)


class Converter(ABC):
    @abstractmethod
    def convert(self, value: float) -> UFloat:
        """Abstract method to convert a float value to a ufloat value"""


float_to_ufloat_conversion_catalogue = Catalogue(Converter)


@float_to_ufloat_conversion_catalogue.register_decorator("zero_uncertainty")
class ZeroUncertaintyConverter(Converter):
    def convert(self, value):
        return ufloat(nominal_value=value, std_dev=0.0)


@float_to_ufloat_conversion_catalogue.register_decorator("fixed_value")
class FixedValueUncertaintyConverter(Converter):
    def __init__(self, fixed_uncertainty):
        self.fixed_uncertainty = fixed_uncertainty

    def convert(self, value):
        return ufloat(value, self.fixed_uncertainty)


@float_to_ufloat_conversion_catalogue.register_decorator("percentage_of_value")
class PercentageUncertaintyConverter(Converter):
    def __init__(self, percentage):
        if not 0.0 <= percentage <= 100.0:
            raise ValueError("Percentage must be between 0 and 100.")
        self.percentage = percentage

    def convert(self, value):
        std_dev = (self.percentage / 100.0) * value
        return ufloat(value, std_dev)


@float_to_ufloat_conversion_catalogue.register_decorator("multiplier")
class MultiplierUncertaintyConverter(Converter):
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def convert(self, value):
        std_dev = self.multiplier * value
        return ufloat(value, std_dev)


# Add new conversion classes here by using the Converter interface


def ufloat_from_float(value: float, method: str, **kwargs) -> UFloat:
    converter_class = float_to_ufloat_conversion_catalogue.get(method)
    converter = converter_class(**kwargs)
    result = converter.convert(value)
    return result
