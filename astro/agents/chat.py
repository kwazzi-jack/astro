# --- Internal Imports ---

# --- External Imports ---
from collections.abc import Sequence

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

# --- Globals ---
_loggy = get_loggy(__file__)


# --- Chat Agent ---
def _normalise_instruction_blocks(
    instructions: str | Sequence[str] | None,
) -> tuple[str, ...]:
    """Return validated instruction strings ready for prompt injection.

    Args:
        instructions (str | Sequence[str] | None): Optional instruction blocks
            supplied by the caller.

    Returns:
        tuple[str, ...]: Cleaned instruction strings preserving order.

    Raises:
        ExpectedVariableType: Raised when the provided instructions container
            is neither a string nor a sequence of strings.
        ExpectedElementTypeError: Raised when an item inside the instructions
            sequence is not a string instance.
    """

    if instructions is None:
        return ()
    if isinstance(instructions, str):
        candidates: Sequence[str] = (instructions,)
    elif isinstance(instructions, Sequence) and not isinstance(instructions, (str, bytes, bytearray)):
        candidates = instructions
    else:
        raise _loggy.ExpectedVariableType(
            var_name="instructions",
            expected=(str, Sequence),
            got=type(instructions),
            with_value=instructions,
        )

    cleaned: list[str] = []
    for index, block in enumerate(candidates):
        if not isinstance(block, str):
            raise _loggy.ExpectedElementTypeError(
                structure_var_name="instructions",
                expected=str,
                got=type(block),
                index_or_key=index,
                with_value=block,
            )
        stripped = block.strip()
        if stripped:
            cleaned.append(stripped)
    return tuple(cleaned)


def create_astro_stream(
    identifier: str = "ollama:llama3.1:latest",
    tools: AgentFn[ChatContext] | AgentFnSequence[ChatContext] | None = None,
    instructions: str | Sequence[str] | None = None,
) -> tuple[StreamFn, MessageList]:
    """Create the Astro chat agent and unified streaming adapter.

    Args:
        identifier (str): Model identifier used to initialise the agent.
        tools (AgentFn[ChatContext] | AgentFnSequence[ChatContext] | None): Optional tools for the agent.
        instructions (str | Sequence[str] | None): Optional extra instruction
            blocks appended to the system prompt.

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
    instruction_blocks = _normalise_instruction_blocks(instructions)
    welcome_template = get_prompt_template("#chat-welcome")

    # Main message list
    messages: MessageList = [create_assistant_message(welcome_template(context))]

    # System instructions
    @astro_agent.instructions
    async def get_system_instructions(ctx: RunContext[ChatContext]) -> str:
        base_prompt = system_template(ctx.deps)
        if not instruction_blocks:
            return base_prompt
        extra_text = "\n\n".join(instruction_blocks)
        return f"{base_prompt}\n\n{extra_text}"

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
