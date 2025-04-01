from pumas.reporting.exceptions import OptionalDependencyNotInstalled


class UFloatStub:
    def __init__(self, *args, **kwargs):
        raise OptionalDependencyNotInstalled(
            package_name="uncertainties", extra_name="uncertainty"
        )


def ufloat_stub(*args, **kwargs):
    raise OptionalDependencyNotInstalled(
        package_name="uncertainties", extra_name="uncertainty"
    )


def ufloat_from_float_stub(*args, **kwargs):
    raise OptionalDependencyNotInstalled(
        package_name="uncertainties", extra_name="uncertainty"
    )


def ufloat_from_str_stub(*args, **kwargs):
    raise OptionalDependencyNotInstalled(
        package_name="uncertainties", extra_name="uncertainty"
    )


class umath_stub:
    def __getattr__(self, name):
        raise OptionalDependencyNotInstalled(
            package_name="uncertainties", extra_name="uncertainty"
        )
