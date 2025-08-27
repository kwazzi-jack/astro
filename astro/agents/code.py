from typing import Any

from astro.llms.base import ModelName, ModelProvider


class CodeAgent:
    def __init__(
        self,
        identifier: str | ModelName | ModelProvider = "ollama",
        provider: str | ModelProvider | None = None,
        model_overrides: dict[str, Any] | None = None,
        system_prompt_tag: str = "code-small-system",
    ) -> None:
        # Default attributes
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
