# --- Internal Imports ---
from collections.abc import Sequence
from types import NoneType

# --- External Imports ---
from pydantic_ai import (
    Agent,
    ModelSettings,
)
from pydantic_ai.models import Model

# --- Local Imports ---
from astro.llms import create_llm_model
from astro.llms.contexts import Context
from astro.logger import get_loggy

# --- Globals ---
_loggy = get_loggy(__file__)


# --- Agent Factory ---
def create_agent(
    identifier: str,
    instructions: str | Sequence[str] | None = None,
    model_settings: ModelSettings | None = None,
    context_type: type = NoneType,
    agent_name: str = "agent",
) -> Agent[Context, str]:
    try:
        model = create_llm_model(identifier)
    except Exception as error:
        raise _loggy.CreationError(object_type=Model, caused_by=error)
    return Agent(
        name=agent_name,
        model=model,
        instructions=instructions,
        deps_type=context_type,
        model_settings=model_settings,
    )
