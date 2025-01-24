"""

This module provides the DataFrame class,
designed host data_frame in the context of multi-objective scoring activities.

The DataFrame class resembles the pandas DataFrame class, but doe non inherit from it.
The DataFrame class is designed to be more lightweight and
tailored to the specific needs of the package.

An instance of DataFrame hosts information about a collection of items.
Each row corresponds to an item, and each column represents a property of these objects.

Simple Usage Examples:

The module can be imported as follows:

>>> from mpstk.dataframes.dataframe import DataFrame

The data_frame can be provided as a list of dictionaries.
Each dictionary in the list represents an item and contributes a row to the data_frame frame;
each key contributes a column name.

>>> input_data = [{"A": 1, "B": 4, "C": "x"}, {"A": 2, "B": 4, "C": "y"}]

1. Initialization
---------------------

>>> df = DataFrame(row_data=input_data)
>>> df.num_rows
2
>>> df.num_columns
3
>>> df.shape
(2, 3)
>>> df.size
6
>>> df.columns
['A', 'B', 'C']


2. Index Management
---------------------

Each DataFrame instance has an index attribute that can be used to access the index values.

>>> df.index.values
[0, 1]

It is possible to reset the index of the DataFrame
>>> df.rebuild_index(strategy="range")
>>> df.index.values
[0, 1]

>>> df.rebuild_index(strategy="uuids")
>>> df.index.values # doctest: +ELLIPSIS
[UUID('...'), UUID('...')]

It is possible to set the index from an existing column.

>>> df.set_index_from_column(column_name="A")
>>> df.index.values
[1, 2]

The column, however, should have unique values, otherwise the following error will be raised:

>>> try:
...     df.set_index_from_column(column_name="B")
... except DuplicateValuesError as e:
...     print(e)
Index values must be unique.

This exception, if raised, can be caught and handled as needed.
Alternatively it is possible to avoid it by checking the uniqueness of th the values of a column before setting it as an index.

>>> df.column_has_unique_values(column_name="B")
False


3. Metadata Management
----------------------------

Metadata are additional data_frame attributes that do not find a place in the main data_frame structure.
Metadata are useful to store additional information about the data_frame, such as units, descriptions, or other properties.

The dataframe initialization creates blank metadata for the columns and rows.

>>> df.column_metadata_map
{'A': ColumnMetadata(uid='A', properties=None), 'B': ColumnMetadata(uid='B', properties=None), 'C': ColumnMetadata(uid='C', properties=None)}

If metadata is provided during initialization, it will be used to populate the metadata map.

>>> column_metadata_map = {
...                       "A": {"uid": "1", "properties": {"unit": "count"}},
...                       "B": {"uid": "2", "properties": {"unit": "count"}},
...                       "C": {"uid": "3", "properties": {"unit": "count"}}
...                       }
>>> df = DataFrame(row_data=input_data,column_metadata_map=column_metadata_map)
>>> df.column_metadata_map
{'A': ColumnMetadata(uid='1', properties={'unit': 'count'}), 'B': ColumnMetadata(uid='2', properties={'unit': 'count'}), 'C': ColumnMetadata(uid='3', properties={'unit': 'count'})}

4. Data Type Management
----------------------------

Before using the data_frame, it might be necessary to observe or change the data types of the data contained in the data_frame.
It is necessary that all the data in a column have the same data type, wich is used as a column flag.

If the column has mixed data types, the column flag will be set to UnspecifiedDataType.

>>> input_data = [{"A": 1, "B": 4, "C": "5"}, {"A": 2, "B": "4", "C": "y"}]
>>> df = DataFrame(row_data=input_data)
>>> df.dtypes_map
{'A': <class 'int'>, 'B': <class 'mpstk.dataframes.dataframe.UnspecifiedDataType'>, 'C': <class 'str'>}

It is possible to set the data types of the columns using a dictionary.
This attempts a casting of the data in the columns to the specified data type.

>>> dtype_map = {"A": int, "B": float, "C": str}
>>> df = DataFrame(row_data=input_data,dtypes_map=dtype_map)
>>> df.dtypes_map
{'A': <class 'int'>, 'B': <class 'float'>, 'C': <class 'str'>}

If the casting fails on any element of the column, the values of the columns and data type of the column will not be changed.
In this case, a warning will be raised.

>>> dtype_map = {"A": int, "B": str, "C": int}
>>> df = DataFrame(row_data=input_data,dtypes_map=dtype_map)
>>> df.dtypes_map
{'A': <class 'int'>, 'B': <class 'str'>, 'C': <class 'str'>}

It is possible to perform a partial casting of the data types of the columns.
In this case, only the columns specified in the dtype_map will be casted.

>>> dtype_map = {"A": float}
>>> df = DataFrame(row_data=input_data,dtypes_map=dtype_map)
>>> df.dtypes_map
{'A': <class 'float'>, 'B': <class 'mpstk.dataframes.dataframe.UnspecifiedDataType'>, 'C': <class 'str'>}

5. Applying Functions to Data
--------------------------------------------
The DataFrame class provides methods to apply functions to the data contained in the DataFrame.

5.1 Elementwise Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Elementwise operations allow you to apply a function to each element of a column.  The function can be applied in parallel.
This might be useful when the function encoding the operation does not work vectroized data, or require special handling.
The index of the original DataFrame is maintained in the new DataFrame.

>>> input_data = [{"A": 1, "B": 4}, {"A": 2, "B": 5}, {"A": 3, "B": 6}]
>>> df = DataFrame(row_data=input_data)
>>> df.set_index_from_column(column_name="A")
>>> df.index.values
[1, 2, 3]

We will apply the `square` function to column 'A'. This function does not require any additional parameters.

>>> def square(x):
...     return x ** 2
>>> df_squared = df.apply_elementwise_column(column_name='A', new_column_name='A_squared', func=square)
>>> df_squared.row_data
[{'A_squared': 1}, {'A_squared': 4}, {'A_squared': 9}]
>>> df_squared.index.values
[1, 2, 3]

We will apply the `add` function to column 'A', adding 5 to each value. This function requires an additional parameter.

>>> def add(x, amount):
...     return x + amount
>>> df_added = df.apply_elementwise_column(column_name='A', new_column_name='A_added', func=add, func_kwargs={'amount': 5})
>>> df_added.row_data
[{'A_added': 6}, {'A_added': 7}, {'A_added': 8}]
>>> df_added.index.values
[1, 2, 3]

This method ensures the index of the original DataFrame is maintained in the new DataFrame, regardless of the parallelization strategy used.


6 Concatenate DataFrames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""  # noqa: E50

import uuid
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Hashable, List, Literal, Optional

from pumas.dataframes.exceptions import (
    ColumnNotFoundError,
    DuplicateValuesError,
    UnsupportedIndexCreationMethodError,
)
from pumas.utils.parallel_utils import parallelize, parallelize_with_indices


class UnspecifiedDataType:
    pass


@dataclass
class ColumnMetadata:
    uid: str
    properties: Dict[str, Any] = None


@dataclass
class RowMetadata:
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Column:
    index: int
    name: str


class Index:
    def __init__(self, values: List[Hashable]):
        self._values = values

        if not self.is_unique:
            raise DuplicateValuesError("Index values must be unique.")

        if any([val is None for val in self._values]):
            raise ValueError("Index values cannot be None.")

    def __len__(self):
        return len(self._values)

    def __getitem__(self, item):
        return self._values[item]

    def __iter__(self):
        return iter(self._values)

    @property
    def values(self) -> List[Hashable]:
        return self._values

    @property
    def is_unique(self) -> bool:
        return len(set(self._values)) == len(self._values)

    def to_list(self) -> List[Hashable]:
        return list(self._values)

    def copy(self):
        return Index(self._values.copy())

    @classmethod
    def rebuild(cls, size: int, strategy: str):
        if strategy == "range":
            return cls.from_range(size)
        elif strategy == "uuids":
            return cls.from_uids(size)
        else:
            raise UnsupportedIndexCreationMethodError(
                f"Unsupported index creation method: {strategy}"
            )

    @classmethod
    def from_range(cls, size: int):
        return cls(values=list(range(size)))

    @classmethod
    def from_uids(cls, size: int):
        return cls(values=[uuid.uuid4() for _ in range(size)])


class DataFrame:
    def __init__(
        self,
        row_data: List[Dict[str, Any]] = None,
        column_metadata_map: Dict[Hashable, Dict[Hashable, Any]] = None,
        dtypes_map: Dict[str, type] = None,
        index: List[Hashable] = None,
    ):
        row_oriented_data_uniformed = self._uniform_data(row_data or [])

        self.column_map = self._initialize_column_map(
            row_oriented_data=row_oriented_data_uniformed
        )
        self.data = self._convert_to_column_oriented(
            row_oriented_data=row_oriented_data_uniformed
        )

        self.dtypes_map = self._initialize_dtypes_map()
        self.column_metadata_map = self._initialize_column_metadata_map(
            column_metadata_map=column_metadata_map
        )

        if index:
            if len(index) != self.num_rows:
                raise ValueError("Length of index does not match the number of rows.")
            self._index = Index(values=index)
        else:
            self._index = Index.from_range(size=self.num_rows)

        if dtypes_map:
            self._apply_dtype_map(dtypes_map)

    @property
    def column_data(self) -> Dict[str, List[Any]]:
        return self.data

    @property
    def row_data(self) -> List[Dict[str, Any]]:
        return self._convert_to_row_oriented()

    @property
    def index(self) -> Index:
        return self._index

    @property
    def num_rows(self) -> int:
        return len(next(iter(self.column_data.values()), []))

    @property
    def num_columns(self) -> int:
        return len(self.column_data)

    @property
    def shape(self) -> tuple:
        return self.num_rows, self.num_columns

    @property
    def size(self) -> int:
        return self.num_rows * self.num_columns

    @property
    def columns(self) -> List[str]:
        return list(self.column_map.keys())

    def rebuild_index(self, strategy: Literal["range", "uuids"]) -> None:
        self._index = Index.rebuild(size=self.num_rows, strategy=strategy)

    def column_has_unique_values(self, column_name: str) -> bool:
        column_values = self.get_column_values(column_name=column_name)
        return len(set(column_values)) == len(column_values)

    def set_index_from_column(self, column_name: str) -> None:
        self._check_column_exists(column_name=column_name)
        column_values = self.get_column_values(column_name=column_name)
        self._index = Index(values=column_values)

    def get_column_values(self, column_name: str) -> List[Any]:
        self._check_column_exists(column_name=column_name)
        return self.column_data.get(column_name)

    def _check_column_exists(self, column_name: str) -> None:
        if column_name not in self.column_map:
            raise ColumnNotFoundError(f"Column '{column_name}' not found.")

    def _initialize_dtypes_map(self) -> Dict[str, type]:
        dtypes_map = {
            column_name: self._infer_column_dtype(
                items=self.column_data.get(column_name)
            )
            for column_name in self.column_map.keys()
        }
        return dtypes_map

    @staticmethod
    def _infer_column_dtype(items: list) -> type:
        column_values = [val for val in items if val is not None]

        if len(column_values) == 0:
            return UnspecifiedDataType

        unique_dtypes = set(map(type, column_values))
        if len(unique_dtypes) == 1:
            return unique_dtypes.pop()
        else:
            return UnspecifiedDataType

    def _initialize_column_metadata_map(
        self,
        column_metadata_map: Dict[Hashable, Dict[Hashable, Any]],
    ) -> Dict[str, ColumnMetadata]:
        if column_metadata_map is None:
            column_metadata_map = {}
        full_column_metadata_map = {}
        for column_name in self.column_map.keys():
            if column_name not in column_metadata_map:
                full_column_metadata_map[column_name] = ColumnMetadata(uid=column_name)
            else:
                full_column_metadata_map[column_name] = ColumnMetadata(
                    **column_metadata_map[column_name]
                )
        return full_column_metadata_map

    @staticmethod
    def _uniform_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Gather all unique column names in the order they appear

        column_names = OrderedDict()
        for row in data:
            for column_name in row.keys():
                if column_name not in column_names:
                    column_names[column_name] = None
        # Normalize data by filling missing columns with None
        uniformed_data = []
        for row in data:
            normalized_row = {
                column_name: row.get(column_name, None) for column_name in column_names
            }
            uniformed_data.append(normalized_row)
        return uniformed_data

    @staticmethod
    def _initialize_column_map(
        row_oriented_data: List[Dict[str, Any]]
    ) -> Dict[str, Column]:
        column_map = {}

        column_names = list(row_oriented_data[0].keys()) if row_oriented_data else []

        for index, column_name in enumerate(column_names):
            column_map[column_name] = Column(index, column_name)

        return column_map

    def _convert_to_column_oriented(
        self, row_oriented_data: List[Dict[str, Any]]
    ) -> Dict[str, List[Any]]:
        column_names = self.column_map.keys()
        column_oriented_data = {column_name: [] for column_name in column_names}
        for row in row_oriented_data:
            for column_name in column_names:
                column_oriented_data[column_name].append(row.get(column_name))
        return column_oriented_data

    def _convert_to_row_oriented(self) -> List[Dict[str, Any]]:
        row_oriented_data = []
        for i in range(self.num_rows):
            row = {}
            for column_name in self.columns:
                row[column_name] = self.column_data[column_name][i]
            row_oriented_data.append(row)
        return row_oriented_data

    def _apply_dtype_map(self, dtypes_map: Dict[str, type]) -> None:
        for column_name, target_dtype in dtypes_map.items():
            if column_name in self.column_data:
                current_dtype = self.dtypes_map[column_name]
                if current_dtype != target_dtype:
                    new_column_values = self._attempt_casting(column_name, target_dtype)
                    if new_column_values is not None:
                        self._update_column_dtype(
                            column_name, new_column_values, target_dtype
                        )
                    else:
                        warnings.warn(
                            f"Failed to cast column '{column_name}' "
                            f"from {current_dtype} to {target_dtype}. "
                            f"Keeping original dtype '{current_dtype}'."
                        )

    def _attempt_casting(
        self, column_name: str, target_dtype: type
    ) -> Optional[List[Any]]:
        new_column_values = []
        for value in self.column_data[column_name]:
            if value is None:
                new_column_values.append(None)
            else:
                try:
                    new_column_values.append(target_dtype(value))
                except (ValueError, TypeError):
                    return None
        return new_column_values

    def _update_column_dtype(
        self, column_name: str, new_column_values: List[Any], target_dtype: type
    ) -> None:
        self.column_data[column_name] = new_column_values
        self.dtypes_map[column_name] = target_dtype

    def apply_elementwise_column_(
        self,
        column_name: str,
        new_column_name: str,
        func: Callable,
        num_jobs: int = 0,
        method: str = "threads",
        func_kwargs: Dict[str, Any] = None,
    ) -> "DataFrame":
        self._check_column_exists(column_name)

        if func_kwargs is None:
            func_kwargs = {}

        apply_func = partial(func, **func_kwargs)

        column_values = self.column_data[column_name]
        new_column_values = parallelize(
            apply_func, column_values, num_jobs=num_jobs, method=method
        )

        new_data_frame = DataFrame(
            row_data=[
                {new_column_name: new_column_values[i]}
                for i in range(len(new_column_values))
            ]
        )
        new_data_frame._index = self._index.copy()

        return new_data_frame

    def apply_elementwise_column(
        self,
        column_name: str,
        new_column_name: str,
        func: Callable,
        num_jobs: int = 0,
        method: str = "threads",
        func_kwargs: Dict[str, Any] = None,
    ) -> "DataFrame":
        self._check_column_exists(column_name)

        if func_kwargs is None:
            func_kwargs = {}

        apply_func = partial(func, **func_kwargs)

        column_values = self.column_data[column_name]
        indexed_column_values = list(zip(self.index.values, column_values))
        new_indexed_column_values = parallelize_with_indices(
            apply_func, indexed_column_values, num_jobs=num_jobs, method=method
        )

        new_index, new_column_values = zip(*new_indexed_column_values)
        new_data = [
            {new_column_name: new_column_values[i]}
            for i in range(len(new_column_values))
        ]
        new_data_frame = DataFrame(row_data=new_data)
        new_data_frame._index = Index(list(new_index))

        return new_data_frame
