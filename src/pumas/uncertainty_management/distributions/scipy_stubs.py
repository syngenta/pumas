from pumas.reporting.exceptions import OptionalDependencyNotInstalled


class stats_stub:
    rv_continuous = None  # Placeholder for the continuous distribution class
    rv_discrete = None  # Placeholder for the discrete distribution class
    rv_histogram = None  # Placeholder for the histogram distribution class

    def __getattr__(self, name):
        raise OptionalDependencyNotInstalled(
            package_name="scipy", extra_name="uncertainty"
        )
