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

from collections.abc import Generator

from langchain_core.messages import BaseMessage
from langchain_core.prompts import AIMessagePromptTemplate, SystemMessagePromptTemplate

from astro.llms.contexts import ChatContext
from astro.paths import get_module_dir, read_markdown_file

# --- Paths ---
PROMPT_DIR = get_module_dir(__file__) / "prompt-templates"
if not PROMPT_DIR.exists():
    raise FileNotFoundError(
        "Cannot find `prompt-templates` directory in package. Ensure `astro` is installed properly."
    )

PROMPT_TEMPLATE_PATHS = {
    "chat-system": PROMPT_DIR / "chat-system.prompt.md",
    "chat-welcome": PROMPT_DIR / "chat-welcome.prompt.md",
    "chat-context": PROMPT_DIR / "chat-context.prompt.md",
}


def get_prompt_template(filetag: str) -> str:
    """Retrieves a prompt template string for the specified file tag.
    Args:
        filetag (str): The file tag identifier used to lookup the corresponding
            prompt template path in PROMPT_TEMPLATE_PATHS.
    Returns:
        str: The content of the markdown file containing the prompt template.
    Raises:
        KeyError: If the provided filetag does not exist in PROMPT_TEMPLATE_PATHS.
    """
    # Validate file tag input
    if filetag not in PROMPT_TEMPLATE_PATHS:
        raise KeyError(f"File tag `{filetag}` does not exist")

    # Return contents of prompt template
    return read_markdown_file(PROMPT_TEMPLATE_PATHS[filetag])


def get_chat_system_prompt(context: ChatContext | None = None) -> BaseMessage:
    if context is None:
        context = ChatContext()

    prompt_text = get_prompt_template(filetag="chat-system")

    return SystemMessagePromptTemplate.from_template(prompt_text).format(
        current_datetime=context.current_datetime(),
        current_platform=context.current_platform(),
        current_python_environment=context.current_python_environment(),
    )


def get_chat_welcome_prompt(context: ChatContext | None = None) -> BaseMessage:
    if context is None:
        context = ChatContext()

    prompt_text = get_prompt_template(filetag="chat-welcome")

    return AIMessagePromptTemplate.from_template(prompt_text).format(
        current_period=context.current_period()
    )


def get_chat_context_prompt_generator(
    context: ChatContext | None = None,
) -> Generator[BaseMessage, None, None]:
    if context is None:
        context = ChatContext()

    prompt_text = get_prompt_template(filetag="chat-context")
    prompt_template = SystemMessagePromptTemplate.from_template(prompt_text)

    while True:
        yield prompt_template.format(current_datetime=context.current_datetime())


if __name__ == "__main__":
    prompt = get_chat_system_prompt()
    prompt.pretty_print()
    prompt = get_chat_welcome_prompt()
    prompt.pretty_print()
