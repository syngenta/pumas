"""
This module is a wrapper around the  `uncertainties package`_ is an open source Python library for doing
calculations on numbers that have uncertainties (like 3.14±0.01).

The :mod:`uncertainties` library calculates uncertainties using linear `error
propagation theory`_ by automatically calculating derivatives and analytically propagating these to the results.

At the core of the uncertainty package is the UFloat class, which represents a number with an uncertainty.
The `ufloat` function yield UFloat instances by providing a nominal value and a standard deviation.

The calculations done with this package will propagate the uncertainties to the result of mathematical calculations.


>>> from pumas.uncertainty.uncertainties_wrapper import ufloat
>>> x = ufloat(nominal_value=2, std_dev=0.1)   # x = 2+/-0.1
>>> y = ufloat(nominal_value=3, std_dev=0.2)   # y = 3+/-0.2
>>> print(2*x)
4.00+/-0.20
>>> print(x+y)
5.00+/-0.22
>>> print(x*y)
6.0+/-0.5

Correlations between variables are automatically handled.

>>> x = ufloat(1, 0.1)  # x = 1+/-0.1
>>> y = ufloat(1, 0.1)  # x = 1+/-0.1

In the case of independent variables, represented by two distinct varialbe instances:

>>> print(x-y)
0.00+/-0.14

In the case of non-independent variables, represented by the same variable instance:

>>> print(x-x)
0.0+/-0

Many other error propagation codes return the incorrect value 0±0.1414… because
they wrongly assume that the two subtracted quantities are independent random variables.
The uncertainties package correctly takes into account the fact that the two quantities are correlated, and returns the correct result: 0±0.


The uncertainties package natively provides convenience methods to construct UFloat instances from strings:

>>> from pumas.uncertainty.uncertainties_wrapper import ufloat_from_str # this is a wrapper around the package's ufloat_fromstr function
>>> x = ufloat_from_str("1.0+/-0.1")
>>> print(x)
1.00+/-0.10

In addition to construct UFloat instances from strings, the module provides a mechanism to convert float values to UFloat, assuming and imposing a certain distribution of uncertainty.

Assing a zero uncertainty to a float value:

>>> from pumas.uncertainty.uncertainties_wrapper import ufloat_from_float
>>> x = ufloat_from_float(1.0, method="zero_uncertainty")
>>> print(x)
1.0+/-0

Assing a fixed uncertainty to a float value:

>>> x = ufloat_from_float(1.0, method="fixed_value", fixed_uncertainty=0.1)
>>> print(x)
1.00+/-0.10

Assing a percentage of the value as uncertainty to a float value:

>>> x = ufloat_from_float(1.0, method="percentage_of_value", percentage=10)
>>> print(x)
1.00+/-0.10

Assing a multiplier of the value as uncertainty to a float value:

>>> x = ufloat_from_float(1.0, method="multiplier", multiplier=0.1)
>>> print(x)
1.00+/-0.10

The module provides a mechanism to extend the conversion methods by implementing a new class that inherits from the Converter interface and registering it in the float_to_ufloat_conversion_catalogue.



.. _uncertainties package: https://pypi.python.org/pypi/uncertainties/
.. _error propagation theory: https://en.wikipedia.org/wiki/Propagation_of_uncertainty

"""  # noqa: E50

from abc import ABC, abstractmethod

from uncertainties import UFloat, ufloat
from uncertainties import ufloat_fromstr as ufloat_from_str
from uncertainties import umath

from pumas.architecture.catalogue import Catalogue

__all__ = ["UFloat", "ufloat", "ufloat_from_str", "ufloat_from_float", "umath"]


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
