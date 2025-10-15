# --- Internal Imports ---
from typing import Any

# --- Local Imports ---
from astro.agents.base import Agent, AgentConfig
from astro.llms.contexts import ChatContext
from astro.llms.prompts import PromptTemplate
from astro.loggings.base import get_loggy
from astro.typings import ModelName, ModelProvider

# --- Globals ---
loggy = get_loggy(__file__)


# --- Factory Functions for Specific Agent Types ---


def create_chat_agent(
    identifier: str | ModelName | ModelProvider,
    provider: str | ModelProvider | None = None,
    system_prompt_or_tag: str | PromptTemplate | None = None,
    welcome_prompt_or_tag: str | PromptTemplate | None = None,
    context_prompt_or_tag: str | PromptTemplate | None = None,
    context_type: type[ChatContext] | None = None,
    **overrides: Any,
) -> Agent:
    loggy.debug("Creating new agent")

    try:
        config = AgentConfig.for_chat(
            identifier,
            provider,
            system_prompt_or_tag,
            welcome_prompt_or_tag,
            context_prompt_or_tag,
            context_type,
            **overrides,
        )
    except Exception as error:
        raise loggy.CreationError(object_type=AgentConfig, caused_by=error)

    loggy.debug(
        f"Using AgentConfig {config.secret_uid} "
        f"for {config.llm_config.model_name.value!r} "
        f"from {config.llm_config.model_provider.value!r}"
    )

    return Agent(config)


def create_primary_chat_agent(
    identifier: str | ModelName | ModelProvider,
    provider: str | ModelProvider | None = None,
) -> Agent:
    """Create a primary chat agent with default settings.

    Args:
        identifier (str | ModelName | ModelProvider): Model name or provider identifier
        provider (str | ModelProvider | None): Optional model provider (inferred if not provided)

    Returns:
        Agent: Configured primary chat agent

    Examples:
        # Simple usage
        agent = create_primary_chat_agent("gpt-4o")

        # With provider specification
        agent = create_primary_chat_agent("gpt-4o", provider="openai")
    """
    return create_chat_agent(
        identifier=identifier,
        provider=provider,
        system_prompt_or_tag=PromptTemplate.CHAT_SYSTEM,
        welcome_prompt_or_tag=PromptTemplate.CHAT_WELCOME,
        context_prompt_or_tag=PromptTemplate.CHAT_CONTEXT,
        context_type=ChatContext,
    )


# --- Legacy Compatibility Class ---


if __name__ == "__main__":
    from astro.utilities.display import astro_md_print, md_print, user_md_input

    # Create a simple chat agent using the class method
    astro = create_primary_chat_agent("gpt-4o-mini")
    astro_md_print("# Astro Agent Demo (Chat)")
    astro_md_print(f"**Model:** {astro.config.model_name}")
    astro_md_print(f"**UID:** {astro.uid}")
    astro_md_print("## Conversation Start")
    for msg in astro.messages:
        astro_md_print(msg.content)
    while True:
        user_input = user_md_input()
        if user_input.lower() in {"exit", "quit", "q"}:
            md_print("Exiting chat. Goodbye!")
            break
        response = astro.act(user_input)
        astro_md_print(response)
