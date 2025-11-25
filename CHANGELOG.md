# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0b1] - 2025-11-25

### Added
- Initial beta release of Astro
- Interactive CLI shell with model selection and streaming responses
- Support for multiple LLM backends (OpenAI, Anthropic, Ollama)
- Custom tool integration via `run_astro_with` helper
- Direct agent API access via `create_agent` and `create_astro_stream`
- Environment configuration from `.env`, `$HOME/.astro/.secrets`, or shell exports
- Rich terminal UI with syntax highlighting and completions
- Dynamic versioning using git tags via `hatch-vcs`
- Type annotations with `py.typed` marker

### Notes
- Requires Python >=3.13
- LaTeX and mathematical tools are experimental (optional extras)
- Full API documentation pending (marked as TODO)
- CI/CD and comprehensive test coverage in progress

[Unreleased]: https://github.com/kwazzi-jack/astro/compare/v0.1.0b1...HEAD
[0.1.0b1]: https://github.com/kwazzi-jack/astro/releases/tag/v0.1.0b1
