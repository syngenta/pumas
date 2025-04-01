from pumas.reporting.exceptions import OptionalDependencyNotInstalled

try:
    from scipy import stats

    UNCERTAINTIES_AVAILABLE = True
except ImportError:
    from pumas.uncertainty_management.distributions.scipy_stubs import (
        stats_stub as stats,
    )

    UNCERTAINTIES_AVAILABLE = False


def check_uncertainties_available() -> None:
    if not UNCERTAINTIES_AVAILABLE:
        raise OptionalDependencyNotInstalled(
            package_name="scipy", extra_name="uncertainties"
        )


__all__ = ["stats"]
