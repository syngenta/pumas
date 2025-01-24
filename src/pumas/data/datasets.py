import csv
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from pumas.dataframes.dataframe import DataFrame


@dataclass(frozen=True)
class ExampleDataset:
    name: str
    description_file_path: Path
    data_file_path: Path

    @property
    def data(self):
        with open(self.data_file_path) as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            a = [dict(zip(header, row)) for row in reader]
        return a

    @property
    def data_frame(self):
        return DataFrame(row_data=self.data)

    @property
    def description(self):
        return self.description_file_path.read_text()


harrington_dataset = ExampleDataset(
    name="harrington",
    description_file_path=files("pumas.data.examples.harrington").joinpath("readme.md"),
    data_file_path=files("pumas.data.examples.harrington").joinpath(
        "harrington_dataset.tsv"
    ),
)
