# Astro

> **Beta Release** — Astro is currently in active development. Core functionality is stable, but the API may evolve before the 1.0 release.

Astro is a chat-first CLI and Python library for interacting with AI agents powered by multiple LLM backends (OpenAI, Anthropic, Ollama). It provides an interactive terminal interface with rich formatting, model selection, and custom tool integration capabilities.

## Installation

Astro requires Python >=3.11 and is distributed via PyPI. We recommend using [`uv`](https://github.com/astral-sh/uv) for installation and dependency management, but any packaging tool should be usable.

### As a Command-Line Tool

Install Astro globally as a CLI application.

Using `uv`:

```bash
uv tool install astro
```

Using `pip`:

```bash
pip install astro
```

Using `pipx`:

```bash
pipx install astro
```

Once installed, launch the interactive shell:

```bash
astro-cli
```

### As a Python Library

Add Astro to your project dependencies.

Using `uv` with `pyproject.toml`:
```bash
uv add astro
```

Using `uv` without `pyproject.toml` (bare `.venv`):

```bash
uv pip install astro
```

Using `pip`:

```bash
pip install astro
```

## Environment Configuration

Astro loads LLM API keys from multiple sources (in order of precedence):

1. **Local `.env` file** — Place a `.env` file in your working directory:
   ```bash
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   ```

2. **User secrets directory** — Store keys in `$HOME/.astro/.secrets`:
   ```bash
   mkdir -p ~/.astro # Or Astro will generate one
   echo "OPENAI_API_KEY=sk-..." >> ~/.astro/.secrets
   ```

3. **Shell environment variables** — Export keys in your shell configuration (e.g., `.bashrc`, `.zshrc`) or manually in the terminal:
   ```bash
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

### Ollama Setup

For local models via Ollama, ensure the Ollama service is running:

```bash
# Install Ollama (see https://ollama.ai)
ollama serve

# Check instance status
ollama ps
```

Astro will automatically detect local Ollama models, but it will only use what has been manually pulled. To use a specific model, use:

```bash
ollama pull llama3.1:latest # Or some other model
```

See [ollama.com/library](https://ollama.com/library) for different models you can pull.

## Usage

Astro offers three primary usage patterns depending on your needs:

### 1. Interactive CLI

Launch the interactive shell after installation:

```bash
# If installed as a tool
astro-cli

# Or run directly with uv
uv run astro-cli
```

Inside the shell, you can:
- Chat with AI models using natural language
- Switch models with `/model`
- View available commands with `/help`
- Exit with `/quit` or `Ctrl+C`

> NOTE: Hashtag commands, e.g. `#history`, exist but are not implemented yet. At they moment, they do nothing.

### 2. Custom Tools Integration

Use `run_astro_with` to inject your own Python functions as agent tools with a chat stream:

```python
from astro import run_astro_with

def calculate_orbit(period_days: float, semi_major_axis_au: float) -> dict:
    """Calculate orbital parameters for a celestial body."""
    # Your implementation here
    return {"period": period_days, "axis": semi_major_axis_au}

def list_telescopes() -> list[str]:
    """Return available telescope identifiers."""
    return ["VLT", "ALMA", "JWST"]

# Launch interactive CLI with custom tools
run_astro_with(
    items=[calculate_orbit, list_telescopes],
    instructions="You are an astronomy assistant with access to orbital calculations."
)
```

See `examples/custom_cli` for an example.

### 3. Direct Agent API

Build custom applications using the agent primitives directly:

```python
from astro.agents.chat import create_astro_stream
from astro.agents.base import create_agent
from astro.contexts import ChatContext

# Create a streaming chat agent
stream_fn, message_history = create_astro_stream(
    identifier="openai:gpt-4o",
    tools=None,  # or pass your tools
    instructions="Custom system instructions here"
)

# Stream responses
async def chat():
    async for output in stream_fn("What is the diameter of Mars?"):
        # Handle different output types (text, tool calls, etc.)
        print(output)

# Or create a custom agent from scratch
agent = create_agent(
    identifier="ollama:llama3.1:latest",
    context_type=ChatContext,
    tools=[your_tools],
    agent_name="my_agent"
)
```

## Examples

The `examples/` directory contains working demonstrations:

### Custom Tools Demo

Clone the repository and run the observatory tools example:

```bash
git clone https://github.com/kwazzi-jack/astro.git
cd astro
uv sync
uv run python examples/custom_tools/run_custom_cli.py
```

This demo shows how to:
- Define custom tool functions
- Register observatories with site metadata
- Inject tools into the Astro CLI
- Use natural language to query domain-specific data

Try these prompts in the demo CLI:
1. "List available observatories and their specialties"
2. "Describe the summit-array site"
3. "Schedule a 40-minute observation of Vega with two exposures"

## Beta Limitations

This is an early beta release focused on core functionality:

- ✅ Multi-model LLM support (OpenAI, Anthropic, Ollama)
- ✅ Interactive CLI with rich formatting
- ✅ Custom tool integration
- ✅ Direct agent API access
- 🚧 Full API documentation (TODO)
- 🚧 Comprehensive test coverage (planned)
- 🚧 CI/CD automation (planned)
