# --- Internal Imports ---

# --- External Imports ---
from pydantic_ai import (
    ModelSettings,
    RunContext,
)

# --- Local Imports ---
from astro.agents.base import create_agent, create_agent_stream
from astro.contexts import ChatContext
from astro.llms.prompts import create_assistant_message, get_prompt_template
from astro.logger import get_loggy
from astro.typings.base import MessageList
from astro.typings.callables import AgentFn, AgentFnSequence, StreamFn
from astro.utilities.timing import get_datetime_str

# --- Globals ---
_loggy = get_loggy(__file__)


# --- Chat Agent ---
def create_astro_stream(
    identifier: str = "ollama:llama3.1:latest",
    tools: AgentFn[ChatContext] | AgentFnSequence[ChatContext] | None = None,
) -> tuple[StreamFn, MessageList]:
    """Create the Astro chat agent and unified streaming adapter.

    Args:
        identifier (str): Model identifier used to initialise the agent.
        tools (AgentFn[ChatContext] | AgentFnSequence[ChatContext] | None): Optional tools for the agent.

    Returns:
        tuple[StreamFn, MessageList]: Stream function and shared
        message history list.
    """
    # Create base agent
    astro_agent = create_agent(
        identifier,
        context_type=ChatContext,
        tools=tools,
        output_type=str,
        model_settings=ModelSettings(temperature=0.7),
        agent_name="astro",
    )

    # Get agent context and templates
    context = ChatContext()
    system_template = get_prompt_template("#chat-system")
    welcome_template = get_prompt_template("#chat-welcome")

    # Main message list
    messages: MessageList = [create_assistant_message(welcome_template(context))]

    # System instructions
    @astro_agent.instructions
    async def get_system_instructions(ctx: RunContext[ChatContext]) -> str:
        return system_template(ctx.deps)

    return create_agent_stream(astro_agent, context, messages), messages


if __name__ == "__main__":
    from astro.config import setup_api_config

    setup_api_config()

    def generate_name() -> str:
        import random

        names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
        return random.choice(names)

    astro_chat, messages = create_astro_stream(
        "ollama:gpt-oss:latest", tools=generate_name
    )

    import asyncio

    async def test():
        async for output in astro_chat("Can you give me the a random name?"):
            print(f"{output=}")

    asyncio.run(test())
