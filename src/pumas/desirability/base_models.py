import pumas.desirability.desirability_utility_functions
from pumas.architecture.parametrized_strategy import AbstractParametrizedStrategy
from pumas.error_propagation.uncertainties import UFloat
from pumas.utils.module_import import switch_library


class Desirability(AbstractParametrizedStrategy):
    def compute_score(self, x: float) -> float:
        results = self._get_partial_utility_function(x=x)
        return results

    def compute_uscore(self, x: UFloat) -> UFloat:
        with switch_library(
            pumas.desirability.desirability_utility_functions,
            "math",
            pumas.desirability.desirability_utility_functions.math_switcher,
            "umath",
        ):
            results = self._get_partial_utility_function(x=x)
        return results
