from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

from pydantic import BaseModel, Field

from astro.utilities.timing import get_timestamp
from astro.utilities.uids import create_uid

# General Path Type
StrPath: TypeAlias = str | Path
StrDict: TypeAlias = dict[str, str]
PathDict: TypeAlias = dict[str, Path]
ModelType = TypeVar("ModelType", bound="TraceableModel")


def _str_dict_to_path_dict(contents: StrDict) -> PathDict:
    try:
        return {key: Path(value).resolve() for key, value in contents.items()}
    except Exception as error:
        raise ValueError(
            "Error while converting conversation index dictionary values to paths"
        ) from error


def _path_dict_to_str_dict(contents: PathDict) -> StrDict:
    try:
        return {key: str(value) for key, value in contents.items()}
    except Exception as error:
        raise ValueError(
            "Error while converting conversation index dictionary values to paths"
        ) from error


def _type_name(obj: Any) -> str:
    return type(obj).__name__


def _options_to_str(values: Sequence[Any]) -> str:
    if len(values) == 1:
        return f"`{values[0]}`"
    else:
        return ", ".join(f"`{value}`" for value in values[:-2]) + f" or `{values[-1]}`"




class TraceableModel(BaseModel):
    """Base class for data models used in agents and modules.

    This class provides a common structure for data models, including metadata fields
    such as name, UID, and creation timestamp. It can be extended to create specific
    data models for inputs, outputs, and states in agents or modules.
    """

    uid: str = Field(
        default_factory=create_uid, description="Unique identifier for this instance."
    )
    created_at: str = Field(
        default_factory=get_timestamp,
        description="Creation timestamp of this instance.",
    )
