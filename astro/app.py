import enum
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.agents.base_agent import (
    BaseAgentInputSchema,
    BaseIOSchema,
    BaseAgentOutputSchema,
)
from pydantic import BaseModel, Field


class State(enum.Enum):
    """State enum for the Astro agent. Stipulates which state the agent is in at any moment."""

    START = enum.auto()
    END = enum.auto()


class Memory(BaseModel):  # Maybe replace with class variables?
    """Memory object for the Astro agent. Allows for persistent storage of objects across states."""

    name: str = Field(..., description="Name of the agent")


class InputSchema(BaseAgentInputSchema):
    """Input schema for Astro agent. Contains the user's latest message to respond too."""


class InnerSchema(BaseAgentOutputSchema):
    """Output schema for Astro agent for internal outputs. Contains Astro's last message
    in response to their previous state and the new state they should move too."""

    state: State


class OuterSchema(BaseAgentOutputSchema):
    """Output schema for Astro agent for final output. Contains the final message
    to be displayed"""


class Astro:
    def __init__(self):

        super().__init__(config)
