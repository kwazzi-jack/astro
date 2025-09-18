import base64
import json
import random
from abc import ABC
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypeVar

import blake3
from pydantic import BaseModel, ConfigDict
from sqlmodel import Field, SQLModel

from astro.utilities.timing import get_datetime_now

StrPath: TypeAlias = str | Path
StrDict: TypeAlias = dict[str, str]
PathDict: TypeAlias = dict[str, Path]
RecordableModelType = TypeVar(
    "RecordableModelType",
    bound="RecordableModel",
)
ImmutableRecordType = TypeVar(
    "ImmutableRecordType", bound="ImmutableRecord", covariant=True
)
HashableObject: TypeAlias = "RecordableModel | ImmutableRecord"
NamedDict: TypeAlias = dict[str, Any]


def secretify(value: Any) -> str:
    """Obscures the middle part of a string, showing only the ends."""
    value_str = str(value)
    length = len(value_str)

    if length <= 8:
        return "*" * 8

    if length > 15:
        prefix_len = 4
        suffix_len = 4
    else:
        # This logic covers lengths from 9 to 15, matching the original if-chain
        uncovered = length - 8
        prefix_len = uncovered // 2
        suffix_len = uncovered - prefix_len

    prefix = value_str[:prefix_len]
    suffix = value_str[length - suffix_len :]
    num_asterisks = length - prefix_len - suffix_len

    return f"{prefix}{'*' * num_asterisks}{suffix}"


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
    if isinstance(obj, type):
        return obj.__name__
    else:
        return type(obj).__name__


def options_to_str(values: Sequence[str]) -> str:
    if len(values) == 0:
        return ""
    elif len(values) == 1:
        return values[0]
    else:
        return ", ".join(values[:-1]) + " or " + values[-1]


def options_to_repr_str(values: Sequence[str]) -> str:
    return options_to_str(list(map(repr, values)))


def type_options(objects: Sequence[Any]) -> list[str]:
    return list(map(_type_name, objects))


def format_object(): ...


class PathKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    SOCKET = "socket"
    FIFO = "fifo"
    BLOCK_DEVICE = "block_device"
    CHARACTER_DEVICE = "character_device"
    MISSING = "missing"
    UNKNOWN = "unknown"


def get_path_type(path: Path) -> PathKind:
    if not path.exists():
        return PathKind.MISSING
    if path.is_file():
        return PathKind.FILE
    if path.is_dir():
        return PathKind.DIRECTORY
    if path.is_symlink():
        return PathKind.SYMLINK
    if path.is_socket():
        return PathKind.SOCKET
    if path.is_fifo():
        return PathKind.FIFO
    if path.is_block_device():
        return PathKind.BLOCK_DEVICE
    if path.is_char_device():
        return PathKind.CHARACTER_DEVICE
    return PathKind.UNKNOWN


def _datetime_json_encoder(dt: datetime) -> str:
    return dt.isoformat()


def _path_json_encoder(path: Path) -> str:
    return str(path)


class RecordableModel(BaseModel, frozen=True):
    """Base model for recordable objects with hashing and dictionary utilities.

    This class provides methods for generating hashes and converting to dictionaries,
    suitable for database storage or serialization.

    Attributes:
        None (inherits from BaseModel; placeholder for any custom attributes).
    """

    model_config = ConfigDict(
        frozen=True,
        # These ensure consistent serialization:
        use_enum_values=True,  # Convert enums to their values
        json_encoders={
            datetime: _datetime_json_encoder,  # Consistent datetime format
            Path: _path_json_encoder,  # Convert Path to string
        },
    )

    def _compute_stable_hash(self) -> bytes:
        """Compute a stable blake3 hash that's consistent across systems.

        Returns:
            bytes: The raw blake3 hash bytes (32 bytes).
        """

        # Get JSON-serializable representation
        model_dict = self.model_dump(
            mode="json",
            exclude_none=False,
            exclude_defaults=False,
        )

        # Create deterministic JSON string
        # Brian - If something is wrong with hashing, its probably here
        json_str = json.dumps(
            model_dict,
            sort_keys=True,  # IMPORTANT: for consistent key order
            separators=(",", ";"),  # No whitespace
            ensure_ascii=True,  # Avoid encoding variations
            default=str,  # Fallback conversion
        )

        # Generate black3 hash
        return blake3.blake3(json_str.encode("utf-8")).digest()

    # OVERRIDE: pydantic.BaseModel.__hash__
    def __hash__(self) -> int:
        """Return the hash value of the model based on its stable hash.

        Returns:
            int: The hash value computed from the stable hash.
        """
        return int.from_bytes(self._compute_stable_hash(), byteorder="big")

    # OVERRIDE: pydantic.BaseModel.__eq__
    def __eq__(self, other: Any) -> bool:
        """Override of BaseModel.__eq__ to compare models based on their stable hash.

        Two models are considered equal if they are of the same type and have the same hash value,
        ensuring consistent equality based on content rather than identity.

        Args:
            other (Any): The object to compare with this model instance.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """

        # Not the same type
        if not isinstance(other, type(self)):
            return False

        # Compare hashes
        return hash(self) == hash(other)

    @property
    def uid(self) -> str:
        """Return unique identifier based on the model's content hash."""
        return self.to_hex()

    def to_hex(self) -> str:
        return hex(hash(self))[2:]

    def to_base64(self) -> str:
        """Generate a base64-encoded representation of the model's stable hash.

        Returns:
            str: Base64-encoded string of the model's blake3 hash (44 characters).
        """
        return base64.b64encode(self._compute_stable_hash()).decode("ascii")

    def secret_uid(self) -> str:
        return secretify(self.uid)


class ImmutableRecord(ABC, SQLModel):
    """Base SQLModel for immutable database records.

    This class represents a persistent, immutable record with metadata for tracking access.

    Attributes:
        uid (int | None): Unique identifier for the record.
        record_hash (int): Hash value for indexing and uniqueness.
        created_at (datetime): Timestamp when the record was created.
        last_accessed_at (datetime | None): Timestamp of last access.
        access_count (int): Number of times the record has been accessed.
    """

    uid: int | None = Field(default=None, primary_key=True)
    record_hash: int = Field(index=True, unique=True, nullable=False)
    created_at: datetime = Field(default_factory=get_datetime_now)

    # Metadata for buffer system
    last_accessed_at: datetime | None = Field(default=None, nullable=True)
    access_count: int = Field(default=0)


class RecordConverter(Protocol[RecordableModelType, ImmutableRecordType]):
    """Protocol for converting between recordable models and immutable records.

    This Protocol defines methods for bidirectional conversion between a model type
    and its corresponding immutable record type, ensuring type safety.

    Attributes:
        None (this is a Protocol with no attributes).
    """

    @classmethod
    def from_model(cls, model: RecordableModelType) -> ImmutableRecordType:
        """Convert a recordable model to an immutable record.

        Args:
            model (RecordableModelType): The model instance to convert.

        Returns:
            ImmutableRecordType: The corresponding immutable record.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def to_model(self) -> RecordableModelType:
        """Convert an immutable record back to a recordable model.

        Returns:
            RecordableModelType: The reconstructed model instance.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


class ModelName(StrEnum):
    # OpenAI
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"

    # Anthropic
    CLAUDE_SONNET_4 = "claude-sonnet-4"

    # Ollama
    GEMMA3_4B = "gemma3:4b"
    LLAMA3_8B = "llama3:8b"
    DEEPSEEK_R1_8B = "deepseek-r1:8b"
    DEEPSEEK_R1_14B = "deepseek-r1:14b"
    MISTRAL_7B = "mistral:7b"
    CODELLAMA_7B = "codellama:7b"
    CODELLAMA_13B = "codellama:13b"
    CODELLAMA_34B = "codellama:34b"
    GPT_OSS_20B = "gpt-oss:20b"

    @property
    def provider(self) -> "ModelProvider":
        match self:
            # OpenAI
            case ModelName.GPT_4O | ModelName.GPT_4O_MINI:
                return ModelProvider.OPENAI
            # Anthropic
            case ModelName.CLAUDE_SONNET_4:
                return ModelProvider.ANTHROPIC
            # Ollama
            case (
                ModelName.GEMMA3_4B
                | ModelName.LLAMA3_8B
                | ModelName.DEEPSEEK_R1_8B
                | ModelName.DEEPSEEK_R1_14B
                | ModelName.MISTRAL_7B
                | ModelName.CODELLAMA_7B
                | ModelName.CODELLAMA_13B
                | ModelName.CODELLAMA_34B
                | ModelName.GPT_OSS_20B
            ):
                return ModelProvider.OLLAMA
            case _:
                # Brian - Annoying that we have to do this for type checker but oh well
                raise ValueError(f"Unsupported ModelName: {self!r}")

    @classmethod
    def available(cls, *inclusions: str) -> list[str]:
        if len(inclusions) == 0:
            return [model.value for model in cls]
        return [model.value for model in cls if model.value in inclusions]

    @classmethod
    def supports(cls, model_name: str) -> bool:
        normalized_value = (
            model_name.strip().upper().replace(":", "_").replace("-", "_")
        )
        return normalized_value in cls.__members__


class ModelProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

    @property
    def default(self) -> "ModelProvider":
        return ModelProvider.OLLAMA

    @classmethod
    def associated_with(cls, model_name: str) -> "ModelProvider":
        if not ModelName.supports(model_name):
            raise ValueError(f"The model {model_name} is not supported by Astro")

        model_name_enum = ModelName(model_name)
        return model_name_enum.provider

    @property
    def models(self) -> list[ModelName]:
        match self:
            # OpenAI
            case ModelProvider.OPENAI:
                return [ModelName.GPT_4O, ModelName.GPT_4O_MINI]

            # Anthropic
            case ModelProvider.ANTHROPIC:
                return [ModelName.CLAUDE_SONNET_4]

            # Ollama
            case ModelProvider.OLLAMA:
                return [
                    ModelName.GEMMA3_4B,
                    ModelName.LLAMA3_8B,
                    ModelName.DEEPSEEK_R1_8B,  # Brian - might be issue if we use langchain-deepseek
                    ModelName.DEEPSEEK_R1_14B,  # Brian - might be issue if we use langchain-deepseek
                    ModelName.MISTRAL_7B,
                    ModelName.CODELLAMA_7B,
                    ModelName.CODELLAMA_13B,
                    ModelName.CODELLAMA_34B,
                    ModelName.GPT_OSS_20B,  # Brian - OpenAI does not support this via API
                ]

            # Fail safe
            case default:
                raise ValueError(
                    f"Probably internal mistake that {default} is not assigned yet"
                )

    @classmethod
    def available(cls, *inclusions: str) -> list[str]:
        if len(inclusions) == 0:
            return [provider.value for provider in cls]
        return [provider.value for provider in cls if provider.value in inclusions]

    @classmethod
    def supports(cls, model_name: str) -> bool:
        normalized_value = model_name.strip().upper()
        return normalized_value in cls.__members__


def _model_provider_options(recommendations: dict[str, ModelName] | None) -> str:
    if recommendations is None:
        return options_to_str(ModelProvider.available())
    return options_to_str(ModelProvider.available(*recommendations.keys()))


def _model_name_options() -> str:
    return options_to_repr_str(ModelName.available())


def _available_provider_model_options(model_provider: ModelProvider) -> str:
    return options_to_repr_str([model.value for model in model_provider.models])


def _count_pydantic_fields(model_class: type[BaseModel]) -> int:
    return len(model_class.model_fields)


if __name__ == "__main__":

    class TestModel(RecordableModel, frozen=True):
        float_value: float = 0.000001
        path_value: Path = Path.cwd()
        date_value: datetime = get_datetime_now()
        str_value: str = "hey there"
        int_value: int = 22
        list_value: list[int] = [1, 2, 3]

    # Generate random values for model_1 and model_2 to make them different
    model_1 = TestModel(
        float_value=random.random(),
        int_value=random.randint(0, 100),
        str_value=f"random_str_{random.randint(0, 100)}",
        list_value=[random.randint(0, 10) for _ in range(3)],
    )
    model_2 = TestModel(
        float_value=random.random(),
        int_value=random.randint(0, 100),
        str_value=f"random_str_{random.randint(0, 100)}",
        list_value=[random.randint(0, 10) for _ in range(3)],
    )
    model_3 = model_1.model_copy(deep=True)

    # model_4 with exact same inputs as model_2
    model_4 = model_2.model_copy(deep=True)

    # model_5 as deep copy of model_1, but change a value
    model_5 = model_1.model_copy(update={"int_value": 999}, deep=True)

    # model_6 not same type
    model_6 = RecordableModel()

    print(f"{model_1=}")
    print(f"{model_2=}")
    print(f"{model_3=}")
    print(f"{model_4=}")
    print(f"{model_5=}")
    print(f"{model_6=}")
    print()
    print(f"{hash(model_1)=}")
    print(f"{hash(model_2)=}")
    print(f"{hash(model_3)=}")
    print(f"{hash(model_4)=}")
    print(f"{hash(model_5)=}")
    print(f"{hash(model_6)=}")
    print()
    print(f"{model_1.to_hex()=}")
    print(f"{model_2.to_hex()=}")
    print(f"{model_3.to_hex()=}")
    print(f"{model_4.to_hex()=}")
    print(f"{model_5.to_hex()=}")
    print(f"{model_6.to_hex()=}")
    print()
    print(f"{model_1.to_base64()=}")
    print(f"{model_2.to_base64()=}")
    print(f"{model_3.to_base64()=}")
    print(f"{model_4.to_base64()=}")
    print(f"{model_5.to_base64()=}")
    print(f"{model_6.to_base64()=}")
    print()
    print(f"{(model_1 is model_2)=:<20} (Expected: False)")
    print(f"{(model_1 is model_3)=:<20} (Expected: False)")
    print(f"{(model_1 is model_4)=:<20} (Expected: False)")
    print(f"{(model_1 is model_5)=:<20} (Expected: False)")
    print(f"{(model_1 is model_6)=:<20} (Expected: False)")
    print(f"{(model_2 is model_3)=:<20} (Expected: False)")
    print(f"{(model_2 is model_4)=:<20} (Expected: False)")
    print(f"{(model_2 is model_5)=:<20} (Expected: False)")
    print(f"{(model_2 is model_6)=:<20} (Expected: False)")
    print(f"{(model_3 is model_4)=:<20} (Expected: False)")
    print(f"{(model_3 is model_5)=:<20} (Expected: False)")
    print(f"{(model_3 is model_6)=:<20} (Expected: False)")
    print(f"{(model_4 is model_5)=:<20} (Expected: False)")
    print(f"{(model_4 is model_6)=:<20} (Expected: False)")
    print(f"{(model_5 is model_6)=:<20} (Expected: False)")
    print()
    print(f"{(model_1 == model_2)=:<20} (Expected: False)")
    print(f"{(model_1 == model_3)=:<20} (Expected: True)")
    print(f"{(model_1 == model_4)=:<20} (Expected: False)")
    print(f"{(model_1 == model_5)=:<20} (Expected: False)")
    print(f"{(model_1 == model_6)=:<20} (Expected: False)")
    print(f"{(model_2 == model_3)=:<20} (Expected: False)")
    print(f"{(model_2 == model_4)=:<20} (Expected: True)")
    print(f"{(model_2 == model_5)=:<20} (Expected: False)")
    print(f"{(model_2 == model_6)=:<20} (Expected: False)")
    print(f"{(model_3 == model_4)=:<20} (Expected: False)")
    print(f"{(model_3 == model_5)=:<20} (Expected: False)")
    print(f"{(model_3 == model_6)=:<20} (Expected: False)")
    print(f"{(model_4 == model_5)=:<20} (Expected: False)")
    print(f"{(model_4 == model_6)=:<20} (Expected: False)")
    print(f"{(model_5 == model_6)=:<20} (Expected: False)")
