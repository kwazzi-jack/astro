import json
import math
import textwrap
import shutil
from typing import Any, Optional

from rich import print as rprint

BASE_INDENT = 4
SAST_DISPLAY_PATTERN = "%Y/%m/%d %H:%M:%S %Z (%z)"


def get_terminal_shape() -> tuple[int, int]:
    return shutil.get_terminal_size()


def get_terminal_width() -> int:
    terminal_width, _ = get_terminal_shape()
    return terminal_width if terminal_width >= 40 else 40


def get_ideal_terminal_width() -> int:
    """Calculate the ideal terminal width for display purposes.

    This function determines an optimal terminal width by taking 60% of the current
    terminal width, with a minimum threshold of 40 characters to ensure readability.

    Returns:
        int: The ideal terminal width in characters. Always returns at least 40
             characters, or 60% of the current terminal width, whichever is larger.

    Examples:
        >>> # Assuming terminal width is 100 characters
        >>> get_ideal_terminal_width()
        60

        >>> # Assuming terminal width is 50 characters
        >>> get_ideal_terminal_width()
        40

    Note:
        This function depends on `get_terminal_shape()` to retrieve the current
        terminal dimensions. The 60% ratio is chosen to provide comfortable
        reading width while leaving space for other content.
    """
    terminal_width, _ = get_terminal_shape()
    ideal_width = math.ceil(terminal_width * 0.6)
    if ideal_width <= 40:
        return 40
    else:
        return ideal_width


def text_wrap(text: str, width: int = 79, indent: int = 0) -> str:
    # Adjust width to account for indentation
    adjusted_width = width - indent

    # Wrap the text
    wrapped_text = textwrap.fill(text, width=adjusted_width)

    # Add indentation to each line
    indented_lines = [" " * indent + line for line in wrapped_text.split("\n")]

    # Join the lines back together
    return "\n".join(indented_lines)


def adaptive_text_wrap(text: str, indent: int = 0) -> str:
    """
    Wrap text to the current terminal width with specified indentation.

    Args:
        text (str): The input text to format
        indent (int): Number of spaces to indent each line

    Returns:
        str: Formatted text with indentation, adapted to terminal width
    """
    terminal_width, _ = get_terminal_shape()
    return text_wrap(text, width=terminal_width, indent=indent)


def field_to_text(
    *values: str,
    name: str,
    indent: int = BASE_INDENT,
) -> str:
    return (
        adaptive_text_wrap(name, indent)
        + ":\n"
        + "\n".join(
            adaptive_text_wrap(str(value), indent + len(name) + 1) for value in values
        )
    )


def format_box(
    fields: dict[str, str],
    label_width: int = 13,
    field_padding: int = 2,
    max_line_length: Optional[int] = None,
) -> str:
    """
    Format agent information in a pretty box.

    Args:
        name: The name of the agent
        fields: Dictionary of field labels and values (None values will be skipped)
        label_width: Width to right-justify labels
        field_padding: Padding between label and value
        max_line_length: Override default width calculation

    Returns:
        A formatted string with box borders
    """
    # Determine width
    width = max_line_length if max_line_length else get_ideal_terminal_width()

    # Box characters
    border_char = "─"
    edge_char = "│"

    # Create parts list
    parts = []

    # Extract uid
    uid = fields.get("uid")

    # Header
    header = f"┌{''.center(width - 2, border_char)}┐"
    if uid is not None:
        title = f"{edge_char} {uid.center(width - 4)} {edge_char}"
        separator = f"├{''.center(width - 2, border_char)}┤"
        parts.extend([header, title, separator])
    else:
        parts.extend([header])

    # Add each field
    for label, value in fields.items():
        if label == "uid":
            continue

        value_str = str(value)
        value_space = (
            width - label_width - field_padding - 4
        )  # 4 for edge chars and spaces

        wrapped_lines = text_wrap(value_str, width=value_space).split("\n")
        line = f"{edge_char} {label.upper().rjust(label_width)}{' ' * field_padding}{wrapped_lines[0].ljust(value_space)} {edge_char}"
        parts.append(line)
        indent = " " * (label_width + field_padding)
        for wrapped_line in wrapped_lines[1:]:
            line = f"{edge_char} {indent}{wrapped_line.ljust(value_space)} {edge_char}"
            parts.append(line)

    # Footer
    footer = f"└{''.center(width - 2, border_char)}┘"
    parts.append(footer)

    return "\n".join(parts)


def prettify_json_string(json_string: str) -> str:
    """Prettify a JSON string for better readability.

    Args:
        json_string (str): The JSON string to prettify

    Returns:
        str: A formatted JSON string with indentation
    """
    try:
        parsed_json = json.loads(json_string)
        return json.dumps(parsed_json, indent=4, ensure_ascii=False)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON string: {error}") from error


# Example usage
if __name__ == "__main__":
    example_text = "This is a long sentence that we need to be able to wrap with our function. If the sentence is very long and does not fit, then we need to make sure it fits in a way that the user can see it."

    print("Using fixed width (79 chars):")
    print(text_wrap(example_text, indent=4))

    print("\nUsing adaptive width based on terminal:")
    print(adaptive_text_wrap(example_text, indent=4))

    # Show the detected terminal width
    width, _ = shutil.get_terminal_size()
    print(f"\nDetected terminal width: {width} characters")

    # Example of the box formatting
    print("\nExample agent box:")
    print(
        format_box(
            {
                "name": "TestAgent",
                "uid": "test-agent-12345",
                "description": "This is a test agent for demonstration",
                "conv. id": "conv12345",
                "created": "2025/04/11 16:45:30 SAST (+0200)",
            },
        )
    )
