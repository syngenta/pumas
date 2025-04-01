from typing import Dict, Generic, List, Mapping, Optional

from pydantic import BaseModel, Field

from pumas.scoring_framework.type_definitions import R, T


class ObjectPropertiesMap(BaseModel, Generic[T]):
    data: Dict[str, Optional[T]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            super().__init__(data=args[0])
        else:
            super().__init__(data=dict(*args, **kwargs))

    def get(self, key: str) -> Optional[T]:
        return self.data.get(key)

    def __getitem__(self, key: str) -> Optional[T]:
        return self.data[key]

    def __setitem__(self, key: str, value: Optional[T]) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def items(self):
        return self.data.items()

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def update(self, *args, **kwargs):
        self.data.update(*args, **kwargs)


class InputData(BaseModel, Generic[T]):
    data: Dict[str, ObjectPropertiesMap[T]]

    class Config:
        arbitrary_types_allowed = True

    def items(self):
        return self.data.items()

    def validate_objectives(self, required_objectives: List[str]) -> bool:
        return all(
            all(obj in obj_data.data for obj in required_objectives)
            for obj_data in self.data.values()
        )


class ScoringResult(BaseModel, Generic[R]):
    aggregated_score: Optional[R]
    desirability_scores: Mapping[str, Optional[R]]

    class Config:
        arbitrary_types_allowed = True


class ScoringResults(BaseModel, Generic[R]):
    results: Dict[str, ScoringResult[R]]

    def get_result_by_uid(self, uid: str) -> ScoringResult[R]:
        return self.results[uid]

    def __getitem__(self, uid: str) -> ScoringResult[R]:
        return self.results[uid]

    def items(self):
        return self.results.items()
