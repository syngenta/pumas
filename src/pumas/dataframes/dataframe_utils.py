from typing import List

from pumas.dataframes.dataframe import Any, DataFrame, Dict, Index


def convert_columnar_to_row_data(
    columnar_data: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    row_data = []
    keys = list(columnar_data.keys())
    num_rows = len(columnar_data[keys[0]])

    for i in range(num_rows):
        row = {key: columnar_data[key][i] for key in keys}
        row_data.append(row)

    return row_data


def concat(
    dataframes: List[DataFrame], join: str = "outer", axis: int = 0
) -> DataFrame:
    # TODO Implement outer join with mismatching columns, and manage duplicates
    # or add a control to avoid duplicates and missing columns/indexes
    # TODO re-Implement inner join

    if not dataframes:
        raise ValueError("No dataframes to concatenate")

    if join not in ["inner", "outer"]:
        raise ValueError("Join must be either 'inner' or 'outer'")

    if axis not in [0, 1]:
        raise ValueError("Axis must be either 0 or 1")

    if axis == 0:
        return _concat_rows(dataframes, join)
    else:
        return _concat_columns(dataframes, join)


def _concat_rows(dataframes: List[DataFrame], join: str) -> DataFrame:
    if join == "inner":
        """
        TODO: this code stub has to be reviewed
        common_columns = set.intersection(*(set(df.columns) for df in dataframes))
        """
        raise NotImplementedError("Inner join not implemented yet")
    else:
        common_columns = set.union(*(set(df.columns) for df in dataframes))

    concatenated_row_data = []
    for df in dataframes:
        for row in df.row_data:
            concatenated_row = {col: row.get(col, None) for col in common_columns}
            concatenated_row_data.append(concatenated_row)

    concatenated_df = DataFrame(
        row_data=concatenated_row_data, index=list(range(len(concatenated_row_data)))
    )

    return concatenated_df


def _concat_columns(dataframes: List[DataFrame], join: str) -> DataFrame:
    # Get index values as sets of Hashable
    index_sets = (set(df.index.values) for df in dataframes)

    # Perform set operation based on join type
    if join == "inner":
        common_indices = set.intersection(*index_sets)
        raise NotImplementedError("Inner join not implemented yet")
    else:
        common_indices = set.union(*index_sets)

    # Convert to sequence for sorting and further use
    common_indices_sequence = sorted(common_indices)

    concatenated_row_data = []
    for idx in common_indices_sequence:
        concatenated_row = {}
        for df in dataframes:
            if idx in df.index.values:
                row_idx = df.index.values.index(idx)
                concatenated_row.update(df.row_data[row_idx])
            else:
                concatenated_row.update({col: None for col in df.columns})
        concatenated_row_data.append(concatenated_row)

    concatenated_df = DataFrame(row_data=concatenated_row_data)
    concatenated_df._index = Index(list(common_indices_sequence))
    return concatenated_df
