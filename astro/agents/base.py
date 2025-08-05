"""
astro/agents/base.py

Core type definitions and base agent abstractions.

Author: Your Name
Date: 2025-07-27
License: MIT

Description:
    Provides protocols, type aliases, and abstract base classes for agent state and behavior.

Dependencies:
    - pydantic
"""

from abc import ABC, abstractmethod
from typing import Any, Hashable, Optional, TypeVar, Generic

from pydantic import BaseModel, Field, computed_field

from astro.utilities.timing import get_timestamp
from astro.utilities.uids import create_uid

StrMessage = str | list[str | dict[Any, Any]]


class DataModel(BaseModel):
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

    @computed_field
    @property
    def name_uid(self) -> str:
        """Returns a unique name identifier for this instance."""
        return f"DataModel#{self.__class__.__name__}#{self.uid}"


class Input(DataModel):
    """Base class for input data to an agent or module.

    This class extends DataModel to include a name and description for the input.
    It can be used as a base class for more specific input types.
    """


class Output(DataModel):
    """Base class for output data from an agent or module.

    This class extends DataModel to include a name and description for the output.
    It can be used as a base class for more specific output types.
    """


class State(DataModel):
    """Base class for state data in an agent or module.

    This class extends DataModel to include a name and description for the state.
    It can be used as a base class for more specific state types.
    """


KeyType = TypeVar("KeyType", bound=int | Hashable)
DataType = TypeVar("DataType", bound=DataModel)
InputType = TypeVar("InputType", bound=Input)
OutputType = TypeVar("OutputType", bound=Output)
StateType = TypeVar("StateType", bound=State)


class EffectModule(ABC, Generic[InputType, OutputType]):
    """
    Abstract base class for effect modules that transform inputs to outputs.

    An EffectModule represents a processing unit that takes inputs of a specific type
    and produces outputs of another type. This class uses generics to provide type
    safety for the input and output data.

    Type Parameters:
        InputType: The type of data that this effect module accepts as input
        OutputType: The type of data that this effect module produces as output

    Methods:
        invoke: Abstract method that must be implemented by subclasses to define
               the transformation logic from inputs to outputs
    """

    @abstractmethod
    def invoke(self, inputs: InputType) -> OutputType:
        raise NotImplementedError


class MemoryModule(ABC, Generic[KeyType, DataType]):


    @abstractmethod
    def __init__(self, values: Any) -> None:
        super().__init__()

    @property
    @abstractmethod
    def last(self) -> DataModel:
        raise NotImplementedError

    @property
    def memory_type(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def values(self) -> list[DataType]:
        raise NotImplementedError

    @abstractmethod
    def clear(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: KeyType) -> DataType:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, key: KeyType, value: DataType):
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

class Agent(ABC, Generic[InputType, OutputType]):

    def __init__(self, name: str, effect: EffectModule, memory: list[] uid: Optional[str] = None):
        self._name = name
        self.effect = effect
        self.uid = uid or create_uid()

    @property
    def name_uid(self) -> str:
        """Returns a unique name identifier for this instance."""
        return f"Agent#{self.__class__.__name__}#{self.uid}"


if __name__ == "__main__":
    input1 = Input()
    input2 = Input()
    input3 = Input(uid=input1.uid, created_at=input1.created_at)

    print(f"{input1=}")
    print(f"{input2=}")
    print(f"{input3=}")
