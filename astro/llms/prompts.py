# --- Internal Imports ---
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

# --- External Imports ---
import frontmatter
from jinja2 import Environment, FileSystemLoader, Template, meta, select_autoescape
from pydantic_ai import ModelRequest, ModelResponse, SystemPromptPart, TextPart

# --- Local Imports ---
from astro.llms.contexts import ChatContext, Context, select_context_type
from astro.logger import get_loggy
from astro.paths import get_module_dir
from astro.typings import NamedDict, options_to_str

# --- Globals ---
_loggy = get_loggy(__file__)


# Path to prompts
_PROMPT_DIR = get_module_dir(__file__) / "prompt-templates"
if not _PROMPT_DIR.exists():
    raise FileNotFoundError(
        "Cannot find 'prompt-templates' directory in package. Ensure Astro is installed properly."
    )

# Setup Jinja2 Environment once
_jinja2_env = Environment(
    loader=FileSystemLoader(_PROMPT_DIR),
    autoescape=select_autoescape(("prompt.md")),
    trim_blocks=True,
    lstrip_blocks=True,
)

# Registered prompt templates
_PROMPT_TEMPLATE_PATHS = {
    "#chat-system": (_PROMPT_DIR / "chat-system.prompt.md").resolve(),
    "#chat-welcome": (_PROMPT_DIR / "chat-welcome.prompt.md").resolve(),
    "#chat-context": (_PROMPT_DIR / "chat-context.prompt.md").resolve(),
}
PromptTags = Literal["#chat-system", "#chat-welcome", "#chat-context"]

# Regex globals
ALPHA = r"[a-z]"
ALPHANUMERIC = r"[a-z0-9]"
SNAKE_CASE = rf"{ALPHA}+[_{ALPHANUMERIC}+]*"
STRING_FORMAT_VARIABLE = rf"{{({SNAKE_CASE})}}"
INPUT_VARIABLE_PATTERN = re.compile(STRING_FORMAT_VARIABLE)


def _is_prompt_file(file_path: Path) -> bool:
    return file_path.name.endswith(".prompt.md")


def _parse_prompt_template(source: str) -> tuple[Template, set[str]]:
    parsed_content = _jinja2_env.parse(source)
    variables = meta.find_undeclared_variables(parsed_content)
    template = _jinja2_env.from_string(source)
    return template, variables


def _variables_in_context(variables: set[str], context: type[Context]) -> bool:
    """Check if all variables are present in the given context.

    Args:
        variables (set[str]): Set of variable names to check.
        context (type[Context]): The context class to check against.

    Returns:
        bool: True if all variables are present in the context, False otherwise.
    """
    return all(context.contains(variable) for variable in variables)


def _load_prompt_file(
    tag: PromptTags,
) -> tuple[Template, NamedDict, type[Context]]:
    # Get file path and if valid prompt file
    file_path = _PROMPT_TEMPLATE_PATHS[tag]
    if not _is_prompt_file(file_path):
        raise _loggy.ValueError(
            f"Invalid file name '{file_path.name}' ('{file_path}'). "
            "Expected '.prompt.md' file extension"
        )

    # Check if file exists
    if not file_path.exists():
        raise _loggy.FileNotFoundError(
            f"Cannot find expected prompt file {file_path} from tag {tag!r}. "
            "Ensure Astro is installed correctly."
        )

    # Load file with frontmatter
    post = frontmatter.load(str(file_path))
    metadata = post.metadata
    source = post.content

    # Parse template and add variables to metadata
    template, variables = _parse_prompt_template(source)
    metadata["variables"] = variables

    # Check if context type is present and extract
    if "context_type" not in metadata:
        raise _loggy.ValueError(f"Expected context_type entry in {tag!r} frontmatter. ")
    context_type = select_context_type(str(metadata["context_type"]))

    # Validate variables against context-type
    if not _variables_in_context(variables, context_type):
        missing_variables = [
            variable for variable in variables if not context_type.contains(variable)
        ]
        missing_str = options_to_str(missing_variables, with_repr=True)
        raise _loggy.ValueError(
            "Context missing fields for these prompt "
            f"template variables: {missing_str}",
            variables=variables,
            context_type=context_type,
            tag=tag,
        )

    # Return extracted values
    return template, metadata, context_type


def get_prompt_template(tag: PromptTags) -> Callable[[Context], str]:
    template, meta, context_type = _load_prompt_file(tag)
    variables = meta["variables"]

    def formatter(context: Context) -> str:
        if not isinstance(context, context_type):
            raise _loggy.ExpectedVariableType(
                var_name="context", expected=context_type, got=type(context)
            )
        context_values = context.to_formatted()
        return template.render(
            {variable: context_values[variable] for variable in variables}
        )

    return formatter


def create_assistant_message(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def create_system_message(text: str) -> ModelRequest:
    return ModelRequest(parts=[SystemPromptPart(content=text)])
