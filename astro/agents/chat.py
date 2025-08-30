import json
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)

from astro.agents.base import AgentState
from astro.llms import ModelName, ModelProvider, create_conversational_model
from astro.llms.base import LLMConfig
from astro.llms.contexts import ChatContext
from astro.llms.prompts import (
    get_chat_context_prompt_generator,
    get_chat_system_prompt,
    get_chat_welcome_prompt,
)
from astro.utilities.uids import create_agent_uid


class AstroChatAgent:
    def __init__(
        self,
        identifier: str | ModelName | ModelProvider = "ollima",
        provider: str | ModelProvider | None = None,
        model_overrides: dict[str, Any] | None = None,
        system_prompt_tag: str = "chat-system",
        welcome_prompt_tag: str = "chat-welcome",
        context_prompt_tag: str | None = "chat-context",
    ) -> None:
        # Default attributes
        self._uid = create_agent_uid("astro-chat")
        self._identifier: str | ModelName | ModelProvider = identifier
        self._provider: str | ModelProvider | None = provider
        self._context = ChatContext()

        # Use prompt context messages
        if context_prompt_tag is not None:
            self._use_context_prompt = True

        # Parse prompt tags
        self._system_prompt_tag = system_prompt_tag
        self._welcome_prompt_tag = welcome_prompt_tag
        self._context_prompt_tag = context_prompt_tag
        self._system_prompt = get_chat_system_prompt(self._context)
        self._welcome_prompt = get_chat_welcome_prompt(self._context)
        self._context_prompt_generator = (
            get_chat_context_prompt_generator(self._context)
            if self._use_context_prompt
            else None
        )

        # Validate model overrides
        if model_overrides is None:
            model_overrides = {}

        # Create conversational llm bot
        self._bot, self._bot_config = create_conversational_model(
            identifier=self._identifier, provider=self._provider, **model_overrides
        )

        # Messages
        self._messages: list[BaseMessage] = [self._system_prompt, self._welcome_prompt]

    @property
    def messages(self) -> list[BaseMessage]:
        """Get the current list of messages in the conversation."""
        return self._messages

    @property
    def config(self) -> LLMConfig:
        """Get the LLM configuration for the agent."""
        return self._bot_config

    @property
    def uid(self) -> str:
        return self._uid

    def save(self, conv_dir: str | Path):
        if isinstance(conv_dir, str):
            conv_dir = Path(conv_dir)

        if not conv_dir.exists():
            raise FileNotFoundError(
                f"Cannot find conversation direction given `{conv_dir}`"
            )
        state = AgentState(uid=self.uid, config=self.config, messages=self.messages)

        

    @classmethod
    def load(cls, file_path: str | Path) -> "AstroChatAgent":
        """Load an agent's state from a JSON file.

        This class method reads a saved state file, reconstructs the agent
        with the original LLM configuration, and restores the message history.

        Args:
            file_path: The path to the file from which to load the state.

        Returns:
            An instance of AstroChatAgent initialized with the loaded state.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        with open(file_path) as file:
            state = json.load(file)

        # Reconstruct config and messages
        config = LLMConfig.model_validate(state["config"])
        messages = messages_from_dict(state["messages"])

        # Create a new agent instance with the loaded configuration
        model_overrides = config.model_dump()
        identifier = model_overrides.pop("model_name")
        provider = model_overrides.pop("provider")

        # The __init__ method handles prompt creation and bot setup
        agent = cls(
            identifier=identifier,
            provider=provider,
            model_overrides=model_overrides,
        )

        # Overwrite the initial messages with the loaded history
        agent._messages = messages

        return agent

    def _invoke_on_messages(self) -> BaseMessage:
        return self._bot.invoke(self._messages)

    def _get_context_prompt_message(self) -> BaseMessage:
        """Get the next context prompt message from the generator.

        Returns:
            The next context prompt message.

        Raises:
            RuntimeError: If context prompt generator is not available.
        """
        #
        if self._context_prompt_generator is None:
            raise RuntimeError("Context prompt generator is not available")
        return next(self._context_prompt_generator)

    def act(self, user_input: str) -> str:
        # Add user message
        user_message = HumanMessage(user_input)
        self._messages.append(user_message)

        # If set, add context message
        if self._use_context_prompt:
            context_message = self._get_context_prompt_message()
            self._messages.append(context_message)

        # Get and add assistant message
        response = self._invoke_on_messages()
        self._messages.append(response)

        # Cheap workaround for LangChain's BaseMessage.content typing...
        if not isinstance(response.content, str):
            raise NotImplementedError(
                f"Astro does not handle the funny LangChain content types yet, only strings: `{type(response.content).__name__}`"
            )

        # Return content of assistant response
        return response.content


if __name__ == "__main__":
    from astro.utilities.display import astro_md_print, md_print, user_md_input

    # --- DEMONSTRATION OF SAVE AND LOAD ---
    save_file = "astro_chat_session.json"
    md_print("### Initializing new agent and running a short conversation...")
    astro_initial = AstroChatAgent("openai")
    initial_response = astro_initial.act("Hi there, what's the capital of France?")
    astro_md_print(initial_response)

    md_print(f"\n### Saving agent state to `{save_file}`...")
    astro_initial.save(save_file)
    md_print("...Save complete.")

    md_print(f"\n### Loading agent from `{save_file}`...")
    astro_loaded = AstroChatAgent.load(save_file)
    md_print("...Load complete.")

    # Verify that the loaded agent has the same state
    assert astro_initial.config == astro_loaded.config
    assert astro_initial.messages == astro_loaded.messages
    md_print(
        "\n✅ **Verification successful:** Loaded agent state matches the original."
    )

    md_print("\n### Continuing conversation with the loaded agent...")
    follow_up_response = astro_loaded.act("What about the capital of Germany?")
    astro_md_print(follow_up_response)

    md_print("\n### Full conversation history from loaded agent:")
    for msg in astro_loaded.messages:
        if isinstance(msg, HumanMessage):
            print(f"👤 User: {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"🤖 Assistant: {msg.content}")

    # --- ORIGINAL INTERACTIVE CHAT LOOP ---
    md_print("\n--- Starting new interactive session ---")
    astro = AstroChatAgent("openai")
    astro_md_print(astro.messages[0].content)

    try:
        while True:
            user_input = user_md_input().strip()
            if user_input.lower()[0] == "q" or user_input.lower() == "exit":
                astro.save(save_file)
                md_print("\n**>> EXITING <<**")
                exit(0)

            response = astro.act(user_input)
            astro_md_print(response)
    except KeyboardInterrupt:
        astro.save(save_file)
        md_print("\n**>> INTERRUPTED <<**")
        exit(0)
