class OptionalDependencyNotInstalled(ImportError):
    """Exception raised when an optional dependency is not installed."""

    def __init__(self, package_name, extra_name=None):
        self.package_name = package_name
        self.extra_name = extra_name
        message = f"The optional dependency '{package_name}' is not installed."
        if extra_name:
            message += f" Install it with 'pip install {package_name}'"
            message += (
                " \n Alternatively, you can include "
                "the optional package while installing PUMAS"
            )
            message += f" Install it with 'pip install pumas[{extra_name}]'"
        super().__init__(message)
