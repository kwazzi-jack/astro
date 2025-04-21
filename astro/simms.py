from enum import Enum, auto
from pathlib import Path

from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.agents.base_agent import (
    BaseAgent,
    BaseAgentConfig,
    BaseIOSchema,
)
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

from pydantic import Field


class SimmsInput(BaseIOSchema):
    """The input schema for the Simms Agent that contains the description
    of the empty CASA measurement set to generate using the `simms` CLI."""

    description: str = Field(
        ...,
        description="The detailed description of the type CASA measurement set the user wants created with `simms`.",
    )
    path: Path = Field(
        ...,
        description="The path object pointing to where the measurement set should go.",
    )


class SimmsState(Enum):
    START = auto()
    ROUTE = auto()
    PARSE = auto()
    VALIDATE = auto()
    EXECUTE = auto()
    FINALISE = auto()
    END = auto()


class SimmsOutput(BaseIOSchema):
    """Output schema at each state of the Simms Agent providing information in the process
    to create an empty CASA measurement set with `simms` thus far."""

    state: SimmsState = Field(
        ..., description="Current state of Simms Agent in the process."
    )
    ms_path: Path = Field(
        None, description="Path to where the measurement set is stored."
    )
    cli_outputs: list[]
