"""
scripts/prompt-refiner.py

Enhanced LLM prompt testing and refinement script.

Author(s):
    - Brian Welman
Date: 2025-08-16
License: MIT

Description:
    A comprehensive script for testing system prompts with various LLM models.
    Supports experiment tracking, versioning, and detailed output generation
    with timing and token usage statistics.

Dependencies:
    - click
    - rich
    - tqdm
    - langchain-core
    - astro.llms.base
"""

import re
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

import click
import dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from tqdm import tqdm

from astro.llms.base import create_llm_model

# Load environment variables
dotenv.load_dotenv()

# Console for rich output
console = Console()

# Hard-coded test messages (from prompts.py)
TEST_MESSAGES: list[str] = [
    "I need to implement W-projection for wide-field imaging with MeerKAT data. What's the best approach?",
    "Calculate the theoretical noise for a 4-hour observation with 16 antennas at 1.4 GHz, 856 MHz bandwidth, dual polarization",
    "How do I fix direction-dependent calibration errors? Please model this using the RIME with mathematics",
    "How does RATT fit into radio astronomy? And what does it stand for?",
    "Write a function called `to_dirty_image` that converts visibilities from a CASA measurement set to an image I can view",
]

# Welcome message for AI assistant
WELCOME_MESSAGE = "How may I assist?"


def fix_latex_delimiters(text: str) -> str:
    """Convert LaTeX delimiters to standard dollar sign format.

    Converts:
    - Display math: \\[...\\] -> $$...$$
    - Inline math: \\(...\\) -> $...$

    Args:
        text: Input text containing LaTeX delimiters.

    Returns:
        Text with converted delimiters.
    """
    # Display math: \[...\] -> $$...$$
    text = re.sub(r"\\?\\\[(.*?)\\?\\\]", r"$$\1$$", text, flags=re.DOTALL)

    # Inline math: \(...\) -> $...$
    text = re.sub(r"\\?\\\((.*?)\\?\\\)", r"$\1$", text, flags=re.DOTALL)

    return text


def extract_model_name(full_model_name: str) -> str:
    """Extract the model name from provider::model format.

    Args:
        full_model_name: Full model name in format "provider::model".

    Returns:
        Just the model name part.

    Examples:
        >>> extract_model_name("openai::gpt-4o")
        'gpt-4o'
    """
    parts = full_model_name.split("::")
    return parts[1] if len(parts) > 1 else full_model_name


def get_next_version(output_dir: Path, experiment_name: str, model_name: str) -> int:
    """Determine the next version number for an experiment.

    Args:
        output_dir: Directory containing existing experiment files.
        experiment_name: Name of the experiment.
        model_name: Name of the model being tested.

    Returns:
        Next available version number.
    """
    pattern = f"{experiment_name}-{model_name}-v*.md"
    existing_files = list(output_dir.glob(pattern))

    if not existing_files:
        return 1

    # Extract version numbers from existing files
    versions = []
    for file in existing_files:
        match = re.search(rf"{experiment_name}-{model_name}-v(\d+)\.md", file.name)
        if match:
            versions.append(int(match.group(1)))

    return max(versions) + 1 if versions else 1


def get_change_description(version: int) -> str:
    """Interactively get a description of changes from the user.

    Args:
        version: Version number being created.

    Returns:
        User-provided change description.
    """
    if version == 1:
        return "Initial experiment version"

    return Prompt.ask(
        f"\n[bold yellow]What changes were made for version {version}?[/bold yellow]",
        default="No description provided",
    )


def write_experiment_header(
    file,
    experiment_name: str,
    model_name: str,
    version: int,
    change_description: str,
    model_config: dict[str, str | int | float],
) -> None:
    """Write the experiment metadata header to the output file.

    Args:
        file: Open file handle to write to.
        experiment_name: Name of the experiment.
        model_name: Name of the model being tested.
        version: Version number.
        change_description: Description of changes.
        model_config: Model configuration parameters.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"# {experiment_name.title()} - {model_name} - Version {version}", file=file)
    print("", file=file)
    print("## Experiment Metadata", file=file)
    print(f"- **Experiment Name:** {experiment_name}", file=file)
    print(f"- **Model:** {model_config['name']}", file=file)
    print(f"- **Version:** {version}", file=file)
    print(f"- **Timestamp:** {timestamp}", file=file)
    print(f"- **Changes:** {change_description}", file=file)
    print("", file=file)

    print("## Model Configuration", file=file)
    for key, value in model_config.items():
        if key != "name":  # Already shown above
            print(f"- **{key.replace('_', ' ').title()}:** {value}", file=file)
    print("", file=file)


def write_system_prompt(file, prompt_content: str) -> None:
    """Write the system prompt section to the output file.

    Args:
        file: Open file handle to write to.
        prompt_content: Content of the system prompt.
    """
    print("## System Prompt", file=file)
    print("```markdown", file=file)
    print(prompt_content, file=file)
    print("```", file=file)
    print("", file=file)


def write_welcome_message(file, welcome_message: str) -> None:
    """Write the welcome message section to the output file.

    Args:
        file: Open file handle to write to.
        welcome_message: Content of the welcome message.
    """
    print("## Welcome Message", file=file)
    print("```", file=file)
    print(welcome_message, file=file)
    print("```", file=file)
    print("", file=file)


def write_test_result(
    file: TextIO,
    test_num: int,
    user_message: str,
    ai_response: AIMessage,
    response_time: float,
    reasoning: bool,
) -> None:
    """Write a single test result to the output file.

    Args:
        file (TextIO): Open file handle to write to.
        test_num (int): Test number.
        user_message (str): User's test message.
        ai_response (AIMessage): AI's response message.
        response_time (float): Time taken for response in seconds.
        reasoning (bool): Whether reasoning mode was used.
    """
    print(f"# Test {test_num}", file=file)
    print(f"## {test_num} User", file=file)
    print(user_message, file=file)
    print(f"## {test_num} Reasoning", file=file)
    if reasoning and ai_response.additional_kwargs.get("reasoning_content"):
        print(
            fix_latex_delimiters(
                ai_response.additional_kwargs.get("reasoning_content")  # type: ignore
            ),
            file=file,
        )
    else:
        print("Reasoning mode not enabled for this run.", file=file)
    print(f"## {test_num} Assistant", file=file)
    print(fix_latex_delimiters(ai_response.content), file=file)  # type: ignore

    # Add timing and token information
    print("", file=file)
    print(f"**Response Time:** {response_time:.2f}s", file=file)

    # Try to extract token usage information
    try:
        if hasattr(ai_response, "usage_metadata") and ai_response.usage_metadata:
            metadata = ai_response.usage_metadata
            input_tokens = metadata.get("input_tokens", "N/A")
            output_tokens = metadata.get("output_tokens", "N/A")
            total_tokens = metadata.get("total_tokens", "N/A")

            print(f"**Input Tokens:** {input_tokens}", file=file)
            print(f"**Output Tokens:** {output_tokens}", file=file)
            print(f"**Total Tokens:** {total_tokens}", file=file)

            if output_tokens != "N/A" and response_time > 0:
                tokens_per_sec = output_tokens / response_time
                print(f"**Tokens/Second:** {tokens_per_sec:.2f}", file=file)
    except Exception:
        print("**Token Usage:** Not available", file=file)

    print("", file=file)


def read_input_file(input_file: TextIO) -> list[str]:
    lines = [line for line in input_file.readlines() if line]
    result: list[str] = []
    for line in lines:
        line = line.strip()
        if not line.startswith(">>>"):
            continue
        result.append(line[3:])
    return result  # type: ignore


@click.command()
@click.argument("model_name", type=str)
@click.option(
    "--prompt-file",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the system prompt file to test.",
)
@click.option(
    "--experiment-name",
    "-e",
    type=str,
    required=True,
    help="Name of the experiment (used for versioning).",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="./outputs",
    help="Output directory for experiment results. Defaults to './outputs'.",
)
@click.option(
    "-i",
    "--input-file",
    type=click.Path(path_type=Path, exists=True, file_okay=True),
    default=None,
    help="Optional path to file of input prompts file.",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=0.8,
    help="Model temperature (0.0-1.0). Defaults to 0.8.",
)
@click.option(
    "--top-p",
    type=float,
    default=1.0,
    help="Top-p sampling parameter (0.0-1.0). Defaults to 1.0.",
)
@click.option(
    "--seed",
    type=int,
    default=666,
    help="Random seed for reproducible generation. Defaults to 666.",
)
@click.option(
    "--include-welcome",
    "-w",
    is_flag=True,
    default=False,
    help="Whether to include welcome message.",
)
@click.option(
    "--reasoning-mode",
    "-r",
    is_flag=True,
    default=False,
    help="Whether to run with reasoning mode",
)
def main(
    model_name: str,
    prompt_file: Path,
    experiment_name: str,
    output_dir: Path,
    input_file: Path | None,
    temperature: float,
    top_p: float,
    seed: int,
    include_welcome: bool,
    reasoning_mode: bool,
) -> None:
    """Test and refine system prompts using various language models.

    This script runs a standardized set of test messages against a system prompt
    using the specified model. Results are saved with comprehensive metadata for
    analysis and comparison across different models and prompt versions.

    Args:
        model_name: Model name in format "provider::model" (e.g., "openai::gpt-4o").
        prompt_file: Path to the markdown file containing the system prompt.
        experiment_name: Name for the experiment (used in output filename).
        output_dir (Path): Directory where results will be saved.
        input_file (Path | None): Optional path to input prompts file.
        temperature: Model temperature parameter.
        top_p: Top-p sampling parameter.
        seed: Random seed for reproducible results.
        include_welcome: Whether to include welcome message
        reasoning_mode: Whether to run with reasoning mode
    """
    global TEST_MESSAGES

    # Get test messages if provided
    use_default_tests = True
    test_origin = "DEFAULTS"
    if input_file:
        try:
            use_default_tests = False
            with open(input_file) as file:
                TEST_MESSAGES = read_input_file(file)
                test_origin = "INPUT-FILE"
        except Exception as error:
            console.print(f"[red]✗ Error reading input file: {error}[/red]")
            return

    # Check tests are present
    test_count = len(TEST_MESSAGES)
    if test_count == 0 and use_default_tests:
        console.print("[red]✗ No tests were provided in default[/red]")
        return
    if test_count == 0 and not use_default_tests:
        console.print(f"[red]✗ No tests were provided in '{input_file}'[/red]")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract model name for filename
    model_only = extract_model_name(model_name)

    # Determine version number
    version = get_next_version(output_dir, experiment_name, model_only)

    # Get change description from user
    change_description = get_change_description(version)

    # Create output filename
    output_file = (
        output_dir / f"{experiment_name}-{model_only.replace(':', '-')}-v{version}.md"
    )

    # Display experiment info
    console.print(
        Panel.fit(
            f"[bold blue]Experiment:[/bold blue] {experiment_name}\n"
            f"[bold blue]Model:[/bold blue] {model_name}\n"
            f"[bold blue]Version:[/bold blue] {version}\n"
            f"[bold blue]Output:[/bold blue] {output_file}\n"
            f"[bold blue]Inputs:[/bold blue] {test_count} messages (origin={test_origin})",
            title="Prompt Testing Session",
            border_style="blue",
        )
    )

    # Read the system prompt
    try:
        prompt_content = prompt_file.read_text(encoding="utf-8")
        console.print(f"✓ Loaded system prompt from: {prompt_file}")
    except Exception as error:
        console.print(f"[red]✗ Error reading prompt file: {error}[/red]")
        return

    # Create the chat model with progress indicator

    try:
        model = create_llm_model(
            name=model_name,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            reasoning=reasoning_mode,
        )
    except Exception as error:
        console.print(f"[red]✗ Error creating model: {error}[/red]")
        return

    # Model configuration for metadata
    model_config = {
        "name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "include_welcome": include_welcome,
        "reasoning_mode": reasoning_mode,
        "input_file": input_file,
    }

    # Write file header
    with open(output_file, "w", encoding="utf-8") as file:
        write_experiment_header(
            file,
            experiment_name,
            model_only.replace(":", "-"),
            version,
            change_description,
            model_config,
        )
        write_system_prompt(file, prompt_content)

        # Write welcome message section if enabled
        if include_welcome:
            write_welcome_message(file, WELCOME_MESSAGE)

    # Run tests with progress bar
    console.print("\n[bold green]Running tests...[/bold green]")

    for i, user_message in tqdm(
        enumerate(TEST_MESSAGES, 1),
        total=len(TEST_MESSAGES),
        desc="Testing",
        unit="test",
    ):
        # Create conversation messages
        messages: list[BaseMessage] = [SystemMessage(prompt_content)]

        # Include welcome message
        if include_welcome:
            messages.append(AIMessage(WELCOME_MESSAGE))

        # Add user message
        messages.append(HumanMessage(user_message))

        # Time the model response
        start_time = time.time()
        try:
            ai_response = model.invoke(messages)
            response_time = time.time() - start_time

            # Ensure we have an AIMessage for type safety
            if isinstance(ai_response, AIMessage):
                # Write test result to file
                with open(output_file, "a", encoding="utf-8") as file:
                    write_test_result(
                        file,
                        i,
                        user_message,
                        ai_response,
                        response_time,
                        reasoning_mode,
                    )
            else:
                raise ValueError(f"Expected AIMessage, got {type(ai_response)}")

        except Exception as error:
            console.print(f"[red]✗ Error in test {i}: {error}[/red]")
            # Write error to file
            with open(output_file, "a", encoding="utf-8") as file:
                print(f"# Test {i}", file=file)
                print("## User", file=file)
                print(user_message, file=file)
                print("## Assistant", file=file)
                print(f"**Error:** {error}", file=file)
                print("", file=file)

    # Success message
    console.print(
        Panel.fit(
            f"[bold green]✓ Experiment completed successfully![/bold green]\n"
            f"Results saved to: {output_file}",
            title="Experiment Complete",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
