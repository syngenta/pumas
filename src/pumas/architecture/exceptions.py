# Custom exceptions
class InvalidParameterTypeError(Exception):
    pass


class InvalidValueError(Exception):
    pass


class InvalidBoundaryError(Exception):
    pass


class InvalidBoundaryDefinitionError(Exception):
    pass


class InvalidDefaultValueError(Exception):
    pass


class InvalidAcceptedValuesError(Exception):
    pass


class InvalidParameterNameError(Exception):
    pass


class ParameterOverlapError(Exception):
    pass


class ParameterSettingError(Exception):
    pass


class ParameterValueNotSet(Exception):
    pass


class ParameterSettingWarning(UserWarning):
    pass


class ParameterUpdateAttributeWarning(UserWarning):
    pass


class ParameterDefinitionError(Exception):
    pass


class InvalidParameterValueError(Exception):
    pass


class ParameterNotFoundError(Exception):
    pass


class InvalidParameterAttributeError(Exception):
    pass


class InvalidInputTypeError(Exception):
    pass
