from typing import Any, TypeVar, Union

Number = Union[int, float]
T = TypeVar("T", bound=Number)

class UFloat:
    def __init__(self, nominal_value: float, std_dev: float = ...) -> None: ...
    def __mul__(self, other: Union[Number, "UFloat"]) -> "UFloat": ...
    def __add__(self, other: Union[Number, "UFloat"]) -> "UFloat": ...
    def __sub__(self, other: Union[Number, "UFloat"]) -> "UFloat": ...
    def __truediv__(self, other: Union[Number, "UFloat"]) -> "UFloat": ...
