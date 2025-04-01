from pumas.reporting.exceptions import OptionalDependencyNotInstalled

try:
    from uncertainties import UFloat, ufloat
    from uncertainties import ufloat_fromstr as ufloat_from_str
    from uncertainties import umath

    UNCERTAINTIES_AVAILABLE = True
except ImportError:
    from pumas.uncertainty_management.uncertainties.uncertainties_stubs import (
        UFloatStub as UFloat,
    )
    from pumas.uncertainty_management.uncertainties.uncertainties_stubs import (
        ufloat_from_str_stub as ufloat_from_str,
    )
    from pumas.uncertainty_management.uncertainties.uncertainties_stubs import (
        ufloat_stub as ufloat,
    )
    from pumas.uncertainty_management.uncertainties.uncertainties_stubs import (
        umath_stub as umath,
    )

    UNCERTAINTIES_AVAILABLE = False


def check_uncertainties_available() -> None:
    if not UNCERTAINTIES_AVAILABLE:
        raise OptionalDependencyNotInstalled(
            package_name="uncertainties", extra_name="uncertainty"
        )


__all__ = [
    "UFloat",
    "ufloat",
    "ufloat_from_str",
    "umath",
    "check_uncertainties_available",
    "UNCERTAINTIES_AVAILABLE",
]
