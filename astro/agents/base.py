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

from pathlib import Path

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, computed_field

from astro.llms.base import LLMConfig
from astro.paths import CONV_DIR


class AgentState(BaseModel):
    uid: str
    config: LLMConfig

    @computed_field
    @property
    def file_path(self) -> Path:
        return CONV_DIR / f"{self.uid}.json"

    def save(self):
        json_str = self.model_dump_json(indent=2)  # pretty JSON
        if not self.file_path.parent.exists():
            raise FileNotFoundError(f"Cannot find agent state folder")
        self.file_path.write_text(json_str, encoding="utf-8")
