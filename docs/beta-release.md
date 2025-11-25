# Beta Release Checklist

This document outlines the steps to publish the `0.1.0b1` beta release of Astro.

## Pre-Release Verification

- [x] Updated `pyproject.toml` with beta classifiers and metadata
- [x] Added `py.typed` marker for type distribution
- [x] Created `CHANGELOG.md` with initial beta entry
- [x] Updated README with installation, configuration, and usage instructions
- [x] Updated CONTRIBUTING.md with TODO markers
- [x] Verified package build excludes `examples/`, `tests/`, `trash/`
- [x] Confirmed console script renamed to `astro-cli`
- [x] Organized dependencies (core + optional extras)

## Release Steps

### 1. Create Beta Tag

Tag the current commit as `v0.1.0b1` to trigger version generation:

```bash
git add .
git commit -m "Prepare v0.1.0b1 beta release"
git tag -a v0.1.0b1 -m "Beta release v0.1.0b1"
git push origin pydantic-ai-switch
git push origin v0.1.0b1
```

### 2. Verify Version Generation

After tagging, verify the version is correctly generated:

```bash
uv run python -c "from astro.__version__ import __version__; print(__version__)"
```

Expected output: `0.1.0b1`

### 3. Build Distribution

Build source distribution and wheel:

```bash
uv build
```

Verify artifacts in `dist/`:
- `astro-0.1.0b1.tar.gz`
- `astro-0.1.0b1-py3-none-any.whl`

### 4. Test Installation Locally

Test the built wheel locally before publishing:

```bash
# Create a test environment
uv venv test-env
source test-env/bin/activate

# Install from local wheel
uv pip install dist/astro-0.1.0b1-py3-none-any.whl

# Test CLI entry point
astro-cli --help

# Test import
python -c "from astro import run_astro_with; print('✓ Import successful')"

# Cleanup
deactivate
rm -rf test-env
```

### 5. Publish to PyPI

**Test PyPI First** (recommended):

```bash
# Configure test PyPI
uv publish --token <test-pypi-token> --publish-url https://test.pypi.org/legacy/

# Install from test PyPI to verify
uv pip install --index-url https://test.pypi.org/simple/ astro==0.1.0b1
```

**Production PyPI**:

```bash
uv publish --token <pypi-token>
```

### 6. Post-Release

- Create GitHub Release from tag `v0.1.0b1`
- Attach built artifacts (`.tar.gz` and `.whl`)
- Copy CHANGELOG entry into release notes
- Announce beta in relevant channels (if applicable)

## Version Management

### Current Setup

- **Dynamic versioning** via `hatch-vcs`
- Reads git tags to generate version strings
- Development builds: `0.1.dev<N>+g<commit>`
- Tagged releases: `0.1.0b1`, `0.1.0b2`, etc.

### Future Beta Releases

For subsequent beta versions:

```bash
# Update CHANGELOG.md with new entry
git add CHANGELOG.md
git commit -m "Prepare v0.1.0b2 beta release"
git tag -a v0.1.0b2 -m "Beta release v0.1.0b2"
git push origin pydantic-ai-switch
git push origin v0.1.0b2
uv build
```

### Transitioning to Stable

When ready for `0.1.0` stable release:

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

Update classifiers in `pyproject.toml` from `Development Status :: 4 - Beta` to `Development Status :: 5 - Production/Stable`.

## Notes

- Beta releases signal stability for basic functionality while indicating API may evolve
- Users installing via `uv tool install astro` will receive beta releases if no stable versions exist
- Specify version explicitly to install specific beta: `uv tool install astro==0.1.0b1`
- Monitor GitHub Issues for beta feedback
- Update documentation based on user feedback before stable release
