class AggregationException(Exception):
    pass


class AggregationValuesToWeightLengthMismatchException(AggregationException):
    pass


class AggregationNegativeValuesException(AggregationException):
    pass


class AggregationNegativeWeightsException(AggregationException):
    pass


class AggregationZeroWeightsException(AggregationException):
    pass


class AggregationNullWeightsException(AggregationException):
    pass


class AggregationNullValuesException(AggregationException):
    pass


class AggregationNullValuesWarning(UserWarning):
    pass


class AggregationNullWeightsWarning(UserWarning):
    pass
