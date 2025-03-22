from pathlib import Path

from typing_extensions import Union, TypedDict

PATH_LIKE = Union[str, Path]


class HistoryDict(TypedDict):
    train_epoch: list[int]
    train: list[dict]
    validate_epoch: list[int]
    validate: list[dict]


class BestValidateDict(TypedDict):
    epoch: int
    value: int | float | None
    metrics: dict
