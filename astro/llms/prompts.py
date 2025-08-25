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

Dependencies:
    - pydantic
    - langchain-core
    - astro.llms.base
"""

from datetime import datetime

from langchain_core.messages import BaseMessage
from langchain_core.prompts import AIMessagePromptTemplate, SystemMessagePromptTemplate

from astro.llms.contexts import MainChatContext
from astro.paths import get_module_dir, read_markdown_file
from astro.utilities.system import (
    PlatformDetails,
    get_platform_details,
    get_platform_str,
)
from astro.utilities.timing import (
    get_datetime_now,
    get_datetime_str,
    get_period_str,
)

# --- Paths ---
PROMPT_DIR = get_module_dir(__file__) / "prompt-templates"
if not PROMPT_DIR.exists():
    raise FileNotFoundError(
        "Cannot find 'prompt-templates' directory in package. Ensure 'astro' is installed properly."
    )

PROMPT_TEMPLATE_PATHS = {
    "chat-system-1": PROMPT_DIR / "chat-system-1.prompt.md",
    "chat-welcome-1": PROMPT_DIR / "chat-welcome-1.prompt.md",
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
        raise KeyError(f"File tag '{filetag}' does not exist")

    # Return contents of prompt template
    return read_markdown_file(PROMPT_TEMPLATE_PATHS[filetag])


def get_main_chat_system_prompt(context: MainChatContext | None = None) -> BaseMessage:
    if context is None:
        context = MainChatContext()

    prompt_text = get_prompt_template(filetag="chat-system-1")

    return SystemMessagePromptTemplate.from_template(prompt_text).format(
        current_datetime=context.current_datetime,
        current_platform=context.current_platform,
    )


def get_main_chat_welcome_prompt(context: MainChatContext | None = None) -> BaseMessage:
    if context is None:
        context = MainChatContext()

    prompt_text = get_prompt_template(filetag="chat-welcome-1")

    return AIMessagePromptTemplate.from_template(prompt_text).format(
        current_period=context.current_period
    )


if __name__ == "__main__":
    prompt = get_main_chat_system_prompt()
    prompt.pretty_print()
    prompt = get_main_chat_welcome_prompt()
    prompt.pretty_print()
