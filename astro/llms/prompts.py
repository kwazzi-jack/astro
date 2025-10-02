"""
astro/llms/prompts.py

LLM-powered hybrid context management system for AI prompt generation.

Author(s):
    - Brian Welman
Date: 2025-08-14
License: MIT

Description:
    Simplified hybrid context system that uses LLMs to generate natural language
    descriptions from structured context data. Combines the best of structured
    data validation with AI-powered natural language generation.

"""

from enum import StrEnum
from pathlib import Path
from typing import Generic

from langchain.prompts import (
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import BaseMessage
from langchain_core.prompts.message import BaseMessagePromptTemplate

from astro.llms.contexts import ChatContext, Context
from astro.loggings import get_loggy
from astro.paths import get_module_dir, read_markdown_file
from astro.typings import BaseMessageType, MessageRole, PromptGenerator, options_to_str

# --- Globals ---
loggy = get_loggy(__file__)

# Path to prompts
_PROMPT_DIR = get_module_dir(__file__) / "prompt-templates"
if not _PROMPT_DIR.exists():
    raise FileNotFoundError(
        "Cannot find `prompt-templates` directory in package. Ensure `astro` is installed properly."
    )


# IMPORTANT: This is manually set to avoid unknown prompts
class RegisteredPromptTemplate(StrEnum):
    CHAT_SYSTEM = "chat-system"
    CHAT_WELCOME = "chat-welcome"
    CHAT_CONTEXT = "chat-context"

    @property
    def purpose(self) -> str:
        match self:
            # Chat related
            case (
                RegisteredPromptTemplate.CHAT_CONTEXT
                | RegisteredPromptTemplate.CHAT_SYSTEM
                | RegisteredPromptTemplate.CHAT_WELCOME
            ):
                return "chat"
            case default:
                raise loggy.ValueError(
                    f"Probably internal mistake that {default} is not assigned yet"
                )

    @property
    def file_path(self) -> Path:
        """Path to the prompt template file associated with this enum value.

        Returns:
            Path: The file path to the markdown template file.

        Raises:
            ValueError: If the enum value is not recognized (internal error).
        """
        match self:
            case RegisteredPromptTemplate.CHAT_CONTEXT:
                return _PROMPT_DIR / "chat-context.prompt.md"
            case RegisteredPromptTemplate.CHAT_SYSTEM:
                return _PROMPT_DIR / "chat-system.prompt.md"
            case RegisteredPromptTemplate.CHAT_WELCOME:
                return _PROMPT_DIR / "chat-welcome.prompt.md"
            case default:
                raise loggy.ValueError(
                    f"Probably internal mistake that {default} is not assigned yet"
                )

    @property
    def context_type(self) -> type[Context]:
        """The Context subclass associated with this prompt template.

        Returns:
            type[Context]: The Context subclass type.

        Raises:
            ValueError: If the enum value is not recognized (internal error).
        """
        match self:
            # Chat related
            case (
                RegisteredPromptTemplate.CHAT_CONTEXT
                | RegisteredPromptTemplate.CHAT_SYSTEM
                | RegisteredPromptTemplate.CHAT_WELCOME
            ):
                return ChatContext
            case default:
                raise loggy.ValueError(
                    f"Probably internal mistake that {default} is not assigned yet"
                )

    @property
    def role(self) -> MessageRole:
        """The LangChain message template type associated with this prompt.

        Returns:
            MessageRole: The LangChain message class.

        Raises:
            ValueError: If the enum value is not recognized (internal error).
        """
        match self:
            # System Message
            case (
                RegisteredPromptTemplate.CHAT_SYSTEM
                | RegisteredPromptTemplate.CHAT_CONTEXT
            ):
                return MessageRole.SYSTEM

            # AI Message
            case RegisteredPromptTemplate.CHAT_WELCOME:
                return MessageRole.AI

            # Catch-all for unrecognized enum values
            case default:
                raise loggy.ValueError(
                    f"Probably internal mistake that {default} is not assigned yet"
                )

    def exists(self) -> bool:
        """Checks if the prompt template file exists.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return self.file_path.exists()

    @classmethod
    def from_tag(cls, tag: str) -> "RegisteredPromptTemplate":
        """Creates a RegisteredPromptTemplate enum instance from a string tag.

        Args:
            tag (str): The tag corresponding to a registered prompt template.

        Returns:
            RegisteredPromptTemplate: The enum instance corresponding to the tag.

        Raises:
            ExpectedVariableType: If tag is not a string.
            NoEntryError: If tag does not correspond to a registered prompt template.
        """
        # Input validation
        if not isinstance(tag, str):
            raise loggy.ExpectedVariableType(
                var_name="tag", expected=str, got=type(tag), with_value=tag
            )
        if tag not in RegisteredPromptTemplate:
            raise loggy.NoEntryError(
                key_value=tag, sources="registered prompt templates"
            )
        return cls(tag)


class PromptTemplate[ContextType: Context]:
    """A wrapper class for managing and formatting prompt templates using LangChain.

    This class abstracts the process of loading raw prompt templates from files,
    determining the appropriate message type (e.g., system or AI message), and
    formatting them with contextual data. It serves as an intermediary between
    raw template files and formatted LangChain BaseMessage objects, making it
    easier to generate prompts dynamically.

    Attributes:
        tag: The RegisteredPromptTemplate enum value identifying the prompt.
        raw_template: The raw string content of the prompt template loaded from file.
        message_type: The type of LangChain message template to use (e.g., SystemMessagePromptTemplate).
    """

    def __init__(self, tag: str | RegisteredPromptTemplate) -> None:
        """Initialize the PromptTemplate with a registered tag.

        Loads the raw template from the associated file and determines the message type
        based on the tag. Validates that the tag is a valid RegisteredPromptTemplate
        instance and that the corresponding file exists.

        Args:
            tag (str | RegisteredPromptTemplate): The RegisteredPromptTemplate enum value for the prompt template.

        Raises:
            ExpectedVariableType: If tag is not a RegisteredPromptTemplate instance.
            FileNotFoundError: If the associated template file does not exist.
            LoadError: If an error occurs while reading the template file.
        """
        # Input validation
        if not isinstance(tag, (str, RegisteredPromptTemplate)):
            raise loggy.ExpectedVariableType(
                var_name="tag",
                expected=RegisteredPromptTemplate,
                got=type(tag),
                with_value=tag,
            )

        # Assign tag based on registered templates
        if isinstance(tag, str):
            try:
                self._template = RegisteredPromptTemplate.from_tag(tag)
            except Exception as error:
                raise loggy.CreationError(
                    object_type=RegisteredPromptTemplate,
                    reason=f"Failed to create from tag {tag!r}",
                    caused_by=error,
                )
        else:
            self._template = tag

        # Check if file exists
        if not self._template.exists():
            raise loggy.FileNotFoundError(
                f"Prompt template file not found for tag {self._template}",
                warning="Ensure `astro` is installed properly to correctly identify system files.",
                file_path=self._template.file_path,
            )

        # Load other attributes
        self._context_type: type[Context] = self._template.context_type
        self._role_type: MessageRole = self._template.role
        self._raw_template: str = self._load_raw_template()

    @property
    def template(self) -> RegisteredPromptTemplate:
        """The registered template tag identifying this prompt.

        Returns:
            RegisteredPromptTemplate: The enum value for this prompt template.
        """
        return self._template

    @property
    def raw_template(self) -> str:
        """The raw string content of the prompt template.

        Returns:
            str: The raw template string loaded from file.
        """
        return self._raw_template

    @property
    def context_type(self) -> type[Context]:
        """The Context subclass type associated with this prompt template.

        Returns:
            type[Context]: The Context subclass type.
        """
        return self._context_type

    @property
    def role_type(self) -> MessageRole:
        """The LangChain message template type (e.g., SystemMessagePromptTemplate).

        Returns:
            MessageRole: The message role type.
        """
        return self._role_type

    def _load_raw_template(self) -> str:
        """Load the raw template string from the file associated with the tag.

        Returns:
            str: The raw template content.

        Raises:
            FileNotFoundError: If the file does not exist.
            LoadError: If an error occurs during file reading.
        """
        file_path = self._template.file_path

        try:
            return read_markdown_file(file_path)
        except Exception as error:
            raise loggy.LoadError(
                path_or_uid=file_path,
                obj_or_key=self._template.value,
                load_from="markdown file",
                caused_by=error,
            )

    def formatted_with(self, context: ContextType) -> BaseMessage:
        """Format the prompt template with the provided context.

        Args:
            context (ContextType): The context data to format the template with.

        Returns:
            BaseMessageType: The formatted LangChain BaseMessage object.

        Raises:
            ExpectedVariableType: If context is not of the expected Context subclass type.
            CreationError: If an error occurs during message creation.
        """
        # Input validation
        if not isinstance(context, self.context_type):
            raise loggy.ExpectedVariableType(
                var_name="context",
                expected=self._context_type,
                got=type(context),
                with_value=context,
            )

        # Create appropriate message template
        message_template: BaseMessagePromptTemplate | None = None
        try:
            match self.role_type:
                case MessageRole.SYSTEM:
                    message_template = SystemMessagePromptTemplate.from_template(
                        self._raw_template
                    )
                case MessageRole.AI:
                    from langchain.prompts import AIMessagePromptTemplate

                    message_template = AIMessagePromptTemplate.from_template(
                        self._raw_template
                    )
                case default:
                    raise loggy.ValueError(
                        f"Probably internal mistake that {default} is not assigned yet"
                    )
        except Exception as error:
            raise loggy.CreationError(
                object_type=BaseMessagePromptTemplate,
                reason="Failed to create message template from raw template",
                caused_by=error,
                context_type=self.context_type,
                role_type=self.role_type,
                message_template=message_template,
            )

        # Format and return message
        context_dict = context.to_dict()
        try:
            # Validate input variables match context keys
            required_vars = set(message_template.input_variables)
            provided_vars = set(context_dict.keys())

            missing_vars = required_vars - provided_vars
            if missing_vars:
                raise loggy.ValueError(
                    f"Missing required template variables: {options_to_str(sorted(missing_vars))}",
                    required_variables=sorted(required_vars),
                    provided_variables=sorted(provided_vars),
                    missing_variables=sorted(missing_vars),
                )

            # Optional: Check for unused variables (warning only)
            # NOTE: commented out because often context will have extra info
            # unused_vars = provided_vars - required_vars
            # if unused_vars:
            #     loggy.warning(
            #         f"Context contains unused variables: {options_to_str(sorted(unused_vars))}",
            #         unused_variables=sorted(unused_vars),
            #     )

            return message_template.format(**context_dict)
        except Exception as error:
            raise loggy.CreationError(
                object_type=self._role_type,
                reason="Failed to format message template with context",
                caused_by=error,
                context_type=self.context_type,
                role_type=self.role_type,
                message_template=message_template,
                context=context,
                context_dict=context_dict,
            )

    def generated_with(self, context: ContextType) -> PromptGenerator:
        """Generator that yields formatted messages given the context.

        Args:
            context (ContextType): The context data to format the template with.

        Yields:
            PromptGenerator: A generator yielding formatted BaseMessage objects.
        """
        while True:
            yield self.formatted_with(context)

    def __repr__(self) -> str:
        return (
            f"PromptTemplate(tag={self.template.value!r}, "
            f"context_type={self.context_type.__name__}, "
            f"role_type={self.role_type})"
        )

    def __str__(self) -> str:
        return f"PromptTemplate for '{self.template.value}' with context type {self.context_type.__name__} and role {self.role_type}"


def get_prompt_template(tag: str) -> RegisteredPromptTemplate:
    """Retrieves the prompt template content for a given file tag.

    This function validates the input file tag, ensures it corresponds to a supported
    prompt template, and reads the associated markdown file. If successful, it returns
    the file's contents as a string. Otherwise, it raises appropriate exceptions for
    invalid inputs or loading issues.

    Args:
        file_tag (str): The tag identifying the prompt template file. Must be a string
            and correspond to a supported template.

    Returns:
        str: The contents of the markdown file associated with the file tag.

    Raises:
        ExpectedVariableType: If file_tag is not a string.
        NoEntryError: If file_tag does not correspond to a supported template.
        LoadError: If an error occurs while reading the markdown file.
    """

    # Input validation
    if not isinstance(tag, str):
        raise loggy.ExpectedVariableType(var_name="tag", got=type(tag), expected=str)

    # Return register prompt template
    try:
        return RegisteredPromptTemplate.from_tag(tag)
    except Exception as error:
        raise loggy.LoadError(
            obj_or_key=tag,
            load_from="registered prompt templates",
            caused_by=error,
        )


def get_chat_system_prompt(context: ChatContext | None = None) -> BaseMessage:
    if context is None:
        context = ChatContext()
    return PromptTemplate[ChatContext]("chat-system").formatted_with(context)


def get_chat_welcome_prompt(context: ChatContext | None = None) -> BaseMessage:
    if context is None:
        context = ChatContext()
    return PromptTemplate[ChatContext]("chat-welcome").formatted_with(context)


def get_chat_context_prompt_generator(
    context: ChatContext | None = None,
) -> PromptGenerator:
    if context is None:
        context = ChatContext()
    return PromptTemplate[ChatContext]("chat-context").generated_with(context)


def get_chat_prompts(
    context: ChatContext | None = None,
) -> tuple[BaseMessage, BaseMessage, PromptGenerator]:
    """Get the full set of chat prompts: system, context, and welcome.

    Args:
        context (ChatContext | None): The chat context to use. If None, a default context is created.

    Returns:
        list[BaseMessage]: A list of formatted BaseMessage objects for the chat prompts.
    """
    if context is None:
        context = ChatContext()

    system_prompt = get_chat_system_prompt(context)
    welcome_prompt = get_chat_welcome_prompt(context)
    context_generator = get_chat_context_prompt_generator(context)
    return system_prompt, welcome_prompt, context_generator


if __name__ == "__main__":
    from time import sleep

    from astro.utilities.display import md_print

    prompt_template = PromptTemplate[ChatContext]("chat-context")
    chat1 = ChatContext()
    chat2 = ChatContext(datetime_live=False)

    count = 3
    md_print("## ChatContext1")
    for message in PromptTemplate[ChatContext]("chat-context").generated_with(chat1):
        md_print(f"**{message.type} Message:**\n\n{message.content}\n\n")
        sleep(1)
        count -= 1
        if count <= 0:
            break

    md_print("## ChatContext2")
    count = 3
    for message in PromptTemplate[ChatContext]("chat-context").generated_with(chat2):
        md_print(f"**{message.type} Message:**\n\n{message.content}\n\n")
        sleep(1)
        count -= 1
        if count <= 0:
            break
