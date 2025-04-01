from pumas.reporting.exceptions import OptionalDependencyNotInstalled


class stats_stub:
    def __getattr__(self, name):
        raise OptionalDependencyNotInstalled(
            package_name="scipy", extra_name="uncertainty"
        )
