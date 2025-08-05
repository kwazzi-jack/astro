from abc import ABC, abstractmethod
from typing import Any


class CommModule(ABC):
    @abstractmethod
    def send(self, agent_id: str, message: Any): ...

    @abstractmethod
    def receive(self) -> list[dict[str, Any]]: ...
