# Custom Tools Example

This example demonstrates how to expose plain Python functions and modules to the Astro chat agent.

## Files

- `observatory.py` defines simple data objects shared by the tools.
- `run_custom_cli.py` launches the Astro CLI with `observatory.py` tools and instructions attached.

## Usage

With `uv` (recommended):

```bash
uv run run_custom_cli.py
```

With native `python`:

```bash
python run_custom_cli.py
```

The script calls `astro.run_astro_with(...)`, pointing it at the helper module. Astro automatically loads each public callable, so you can add more helper functions without changing the bootstrap code.

## Sample Prompts

Use the following prompts in the chat to experiment with the `observatory.py` script:

- "List the available observatories and tell me which one is best for spectroscopic work."
- "Describe the Aurora Ridge site and summarize its instruments."
- "Schedule a 40-minute observation of Vega at the demo station with two exposures."
- "What instruments are installed at the Summit Array?"
- "I need to observe Jupiter tonight. Can you plan a one-hour block at the coastal site?"
