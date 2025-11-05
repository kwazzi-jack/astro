# --- Internal Imports ---
from collections.abc import AsyncIterator

# --- External Imports ---
from pydantic_ai import AgentRunResultEvent, AgentStreamEvent, ModelSettings, RunContext

# --- Local Imports ---
from astro.agents.base import create_agent
from astro.llms.contexts import ChatContext
from astro.llms.prompts import create_assistant_message, get_prompt_template
from astro.logger import get_loggy
from astro.typings import AsyncChatFunction, MessageList
from astro.utilities.timing import get_datetime_str

# --- Globals ---
loggy = get_loggy(__file__)


# --- Chat Agent ---
def create_astro_chat(
    identifier: str = "ollama:llama3.1:latest",
) -> tuple[
    AsyncChatFunction,
    MessageList,
]:
    # Create base agent
    astro_agent = create_agent(
        identifier,
        model_settings=ModelSettings(temperature=0.7),
        context_type=ChatContext,
        agent_name="astro-chat",
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

    # System tool for datetime
    @astro_agent.tool(docstring_format="google")
    async def get_system_datetime(ctx: RunContext[ChatContext]) -> str:
        """Get the current system datetime formatted as a string.

        Returns a formatted datetime string from the chat context's datetime attribute.

        Args:
            ctx (RunContext[ChatContext]): Runtime context containing chat state and dependencies.

        Returns:
            str: Formatted datetime string representing the current system time.
        """
        return get_datetime_str(dt=ctx.deps.datetime)

    async def chat_stream(
        prompt: str,
    ) -> AsyncIterator[AgentStreamEvent | AgentRunResultEvent[str]]:
        """Stream responses from the astro agent for a given prompt.

        Args:
            prompt (str): User prompt to send to the agent.

        Yields:
            (AgentStreamEvent | AgentRunResultEvent[str]): Streaming events from the agent.

        Raises:
            ExpectedTypeError: If prompt is not a string.
        """
        if not isinstance(prompt, str):
            raise loggy.ExpectedTypeError(
                expected=str, got=type(prompt), with_value=prompt
            )

        loggy.debug("Starting chat stream", prompt=prompt)
        nonlocal messages
        async for event in astro_agent.run_stream_events(
            prompt, message_history=messages, deps=context
        ):
            if event.event_kind == "agent_run_result":
                messages.extend(event.result.new_messages())
            yield event

    return chat_stream, messages
