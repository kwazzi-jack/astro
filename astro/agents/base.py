"""
astro/agents/base.py

Core type definitions and base agent abstractions.

Author: Your Name
Date: 2025-07-27
License: MIT

Description:
    Provides protocols, type aliases, and abstract base classes for agent state and behavior.

Dependencies:
    - pydantic
"""

# --- Internal Imports ---
from types import NoneType
from typing import Any

# --- External Imports ---
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import field_serializer, field_validator

# --- Local Imports ---
from astro.llms.base import LLMConfig, create_llm_model
from astro.llms.contexts import ChatContext, Context
from astro.llms.prompts import PromptTemplate, RegisteredPromptTemplate
from astro.loggings.base import get_loggy
from astro.typings import (
    ModelName,
    ModelProvider,
    PromptGenerator,
    RecordableModel,
    type_name,
)
from astro.utilities.uids import create_agent_uid

# --- Globals ---
loggy = get_loggy(__file__)


class AgentConfig(RecordableModel, frozen=True):
    name: str = "agent"
    llm_config: LLMConfig
    context_type: type[Context] | None = None
    system_prompt_template: RegisteredPromptTemplate | None = None
    welcome_prompt_tag: RegisteredPromptTemplate | None = None
    context_prompt_tag: RegisteredPromptTemplate | None = None

    @field_serializer("context_type", mode="plain")
    def _serialize_context_type(self, context_type: type[Context] | None) -> str | None:
        if context_type is None:
            return None
        else:
            return f"{context_type.__module__}.{context_type.__name__}"

    @field_validator("context_type", mode="before")
    def _validate_context_type(cls, context_type: Any) -> type[Context] | None:
        # None -> do nothing
        if context_type is None:
            return None

        # Context type -> return as is
        elif isinstance(context_type, type) and issubclass(context_type, Context):
            return context_type

        # String -> try to import and return
        elif isinstance(context_type, str):
            try:
                module_name, class_name = context_type.rsplit(".", 1)
                module = __import__(module_name, fromlist=[class_name])
                context_cls = getattr(module, class_name)
                if isinstance(context_cls, type) and issubclass(context_cls, Context):
                    return context_cls
            except Exception as error:
                raise loggy.CreationError(
                    object_type=Context,
                    reason=f"Failed to import context class {context_type}",
                    caused_by=error,
                )

        # Anything else -> error
        else:
            raise loggy.ExpectedVariableType(
                var_name="context_type",
                expected=(type[Context], str, NoneType),
                got=type(context_type),
            )

    @classmethod
    def _parse_prompt_template(
        cls,
        var_name: str,
        prompt_template_or_tag: str | RegisteredPromptTemplate | None,
    ) -> RegisteredPromptTemplate | None:
        # Handle null cases
        if prompt_template_or_tag is None:
            return None
        elif not isinstance(prompt_template_or_tag, (str, RegisteredPromptTemplate)):
            raise loggy.ExpectedVariableType(
                var_name=var_name,
                expected_type=(str, RegisteredPromptTemplate, None),
                actual_type=type(prompt_template_or_tag),
            )
        elif isinstance(prompt_template_or_tag, str):
            try:
                return RegisteredPromptTemplate(prompt_template_or_tag)
            except Exception as error:
                raise loggy.CreationError(
                    object_type=AgentConfig,
                    reason=(
                        f"Failed to create RegisteredPromptTemplate for {var_name!r} "
                        f"from tag {prompt_template_or_tag!r}"
                    ),
                    caused_by=error,
                )
        else:
            return prompt_template_or_tag

    @classmethod
    def for_chat(
        cls,
        identifier: str | ModelName | ModelProvider,
        provider: str | ModelProvider | None = None,
        system_prompt_template_or_tag: str | RegisteredPromptTemplate | None = None,
        welcome_prompt_template_or_tag: str | RegisteredPromptTemplate | None = None,
        context_prompt_template_or_tag: str | RegisteredPromptTemplate | None = None,
        context_type: type[ChatContext] | None = None,
        **overrides: Any,
    ) -> "AgentConfig":
        # Create LLM config
        try:
            llm_config = LLMConfig.for_chat(
                identifier=identifier,
                provider=provider,
                **overrides,
            )
        except Exception as error:
            raise loggy.CreationError(
                object_type=AgentConfig,
                reason="Failed to create LLMConfig",
                caused_by=error,
            )
        try:
            system_prompt_template = (
                AgentConfig._parse_prompt_template(
                    "system_prompt_template_or_tag", system_prompt_template_or_tag
                )
                or RegisteredPromptTemplate.CHAT_SYSTEM
            )
            welcome_prompt_template = (
                AgentConfig._parse_prompt_template(
                    "welcome_prompt_template_or_tag", welcome_prompt_template_or_tag
                )
                or RegisteredPromptTemplate.CHAT_WELCOME
            )
            context_prompt_template = (
                AgentConfig._parse_prompt_template(
                    "context_prompt_template_or_tag", context_prompt_template_or_tag
                )
                or RegisteredPromptTemplate.CHAT_CONTEXT
            )

        except Exception as error:
            raise loggy.CreationError(
                object_type=RegisteredPromptTemplate,
                reason="Error occurred while making prompt templates for AgentConfig",
                caused_by=error,
            )

        return cls(
            name="chat-agent",
            llm_config=llm_config,
            context_type=context_type or ChatContext,
            system_prompt_template=system_prompt_template,
            welcome_prompt_tag=welcome_prompt_template,
            context_prompt_tag=context_prompt_template,
        )


class Agent:
    """A general purpose agent that can perform various tasks using LLM models.

    This is the base agent class that handles common functionality like message
    management, prompt templates, and LLM interaction. Specific agent behaviors
    are created through factory functions and configuration.
    """

    def __init__(
        self, config: AgentConfig, shared_context: Context | None = None
    ) -> None:
        """Initialize an Agent with the given configuration.

        Args:
            agent_config: Configuration for the agent including LLM settings and prompts
            context: Optional context instance, defaults to appropriate context type
        """
        # Core components
        self._config = config
        self._name = config.name
        self._context = shared_context or (
            config.context_type() if config.context_type else None
        )
        self._uid = create_agent_uid(self._name)

        # Create the chat model
        self._llm_model = create_llm_model(self._config.llm_config)

        # Messages
        self._messages: list[BaseMessage] = []

        # Create prompt templates from config
        self._system_template: PromptTemplate | None = None
        self._welcome_template: PromptTemplate | None = None
        self._context_template: PromptTemplate | None = None

        if self._config.system_prompt_template:
            self._system_template = PromptTemplate(self._config.system_prompt_template)

        if self._config.welcome_prompt_tag:
            self._welcome_template = PromptTemplate(self._config.welcome_prompt_tag)

        if self._config.context_prompt_tag:
            self._context_template = PromptTemplate(self._config.context_prompt_tag)

        # Create live generator for context updates
        self._context_generator: PromptGenerator | None = (
            self._context_template.generated_with(self._context)
            if self._context_template and self._context
            else None
        )

        # Initialize message history
        self._reset_messages()

    @property
    def messages(self) -> list[BaseMessage]:
        """Get the current list of messages in the conversation."""
        return self._messages

    @property
    def config(self) -> LLMConfig:
        """Get the LLM configuration for the agent."""
        return self._config.llm_config

    @property
    def uid(self) -> str:
        """Get the unique identifier for this agent."""
        return self._uid

    @property
    def context(self) -> Any:
        """Get the context used for prompt generation."""
        return self._context

    def _reset_messages(self) -> None:
        """Reset the message history, adding system and welcome prompts if enabled."""
        self._messages = []

        # Add system prompt
        if self._system_template is not None:
            system_message = self._system_template.formatted_with(self._context)
            self._messages.append(system_message)

        # Add welcome prompt
        if self._welcome_template is not None:
            welcome_message = self._welcome_template.formatted_with(self._context)
            self._messages.append(welcome_message)

    def _invoke_on_messages(self) -> BaseMessage:
        """Invoke the LLM on the current message history."""
        return self._llm_model.invoke(self._messages)

    def _add_context_message(self) -> None:
        """Add a fresh context message to the conversation if context prompts are enabled."""
        if self._context_generator is not None:
            context_message = next(self._context_generator)
            self._messages.append(context_message)

    def act(self, user_input: str) -> str:
        """Process user input and generate a response.

        Args:
            user_input: The user's message text.

        Returns:
            str: The agent's response text.

        Raises:
            NotImplementedError: If LangChain returns non-string content.
        """
        # Add user message
        user_message = HumanMessage(user_input)
        self._messages.append(user_message)

        # Add context message with live updates if enabled
        if self._context_template is not None:
            self._add_context_message()

        # Get and add assistant response
        response = self._invoke_on_messages()
        self._messages.append(response)

        # Handle LangChain's content typing
        if not isinstance(response.content, str):
            raise loggy.NotImplementedError(
                "Astro does not handle the funny LangChain content types "
                f"yet, only strings: {type_name(response.content)}",
                reponse_content_type=type(response.content),
            )

        return response.content

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        template_info = []
        if self._system_template:
            template_info.append("system")
        if self._welcome_template:
            template_info.append("welcome")
        if self._context_template:
            template_info.append("context")

        templates_str = f"[{', '.join(template_info)}]" if template_info else "[]"

        return (
            f"AstroAgent(uid={self.uid!r}, "
            f"model={self.config.model_name}, "
            f"templates={templates_str}, "
            f"messages={len(self.messages)})"
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"AstroAgent with {len(self.messages)} messages using {self.config.model_name}"


if __name__ == "__main__":
    from astro.utilities.display import astro_md_print, user_md_input

    agent_config = AgentConfig.for_chat(identifier="gpt-4o-mini")

    agent = Agent(config=agent_config)

    print(agent_config)
    print(agent)

    config_json = agent_config.model_dump_json(indent=4)
    print(config_json)

    config2 = AgentConfig.model_validate_json(config_json)
    print(config2)
    agent2 = Agent(config=config2)

    for message in agent.messages:
        astro_md_print(message.content)
    while user_input := user_md_input():
        if user_input.lower() in ("exit", "q", "quit"):
            break
        ai_output = agent2.act(user_input)
        astro_md_print(ai_output)
