# Contributing to Astro

**TODO**: Full contribution guidelines are being developed.

## Quick Start for Contributors

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/kwazzi-jack/astro.git
   cd astro
   ```

2. Install dependencies with `uv`:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

### Running Tests

```bash
uv run pytest
```

### Code Style

This project follows:
- Python 3.13+ conventions
- Google-style docstrings
- Type annotations (checked with mypy)
- Black formatting
- Ruff linting

Run checks:
```bash
uv run pytest --ruff --black --mypy
```

## TODO

- [ ] Complete contribution workflow documentation
- [ ] PR review guidelines
- [ ] Issue templates
- [ ] Release process documentation
- [ ] Developer API reference
- [ ] Architecture overview

For now, please open an issue to discuss significant changes before starting work.