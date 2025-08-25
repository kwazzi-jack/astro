from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from astro.llms import ModelName, ModelProvider, create_conversational_model
from astro.llms.contexts import get_initial_chat_context


class ChatAgent:
    def __init__(
        self,
        system_prompt_tag: str,
        welcome_prompt_tag: str,
        identifier: str | ModelName | ModelProvider | None = None,
        provider: str | ModelProvider | None = None,
        model_overrides: dict[str, Any] | None = None,
    ) -> None:
        # Parse prompt tags
        self._system_prompt_tag = system_prompt_tag
        self._welcome_prompt_tag = welcome_prompt_tag

        # Parse identifier and provider
        self._identifier, self._provider = self._handle_identifier_provider(
            identifier, provider
        )

        # Validate model overrides
        if model_overrides is None:
            model_overrides = {}

        # Create conversational llm bot
        self._bot = create_conversational_model(
            identifier=self._identifier, provider=self._provider, **model_overrides
        )

    def _handle_identifier_provider(
        self,
        identifier: str | ModelName | ModelProvider | None = None,
        provider: str | ModelProvider | None = None,
    ) -> tuple[str | ModelName | ModelProvider, str | ModelProvider | None]:
        # Check if no identifier but provider given
        if identifier is None and provider is not None:
            raise ValueError(
                "Expected `identifier` to be not None if `provider` is not None"
            )

        # Identifier given with provider
        elif identifier is None:
            identifier = "ollama"  # Use recommended Ollama model

        # Return parsed identifier and provider
        # NOTE - B - Most validation is left for the astros.llm.base code
        return identifier, provider

    def _handle_initial_prompt_messages(self) -> tuple[BaseMessage, BaseMessage]:
        # Load context
        context_info = get_initial_chat_context()

        # Load system prompt
        system_message = get_chat_system_prompt(context_info)

    def act(self, user_input: str) -> str: ...
