class NonUniqueIndexError(Exception):
    """Raised when the DataFrame index is not unique."""

    pass


class NonUniqueColumnsError(Exception):
    """Raised when a dataframe has multiple columns with the same name."""

    ...


class NonUniqueColumnValuesError(Exception):
    """Raised when the values hosted in a column are not unique."""

    ...


class ColumnNotFoundError(Exception):
    """Raised when a specified column is not found in the DataFrame."""

    ...


class NonNumericColumnError(Exception):
    """Raised when a non-numeric column is provided for conversion to ufloat."""

    ...


class DuplicateValuesError(Exception):
    """Raised when there are duplicate values in a column or."""

    ...


class UnsupportedIndexCreationMethodError(Exception):
    """Raised when an unsupported index creation method is requested."""

    ...
