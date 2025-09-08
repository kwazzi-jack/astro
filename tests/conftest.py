"""
Pytest configuration and shared fixtures.

This module provides common fixtures and configuration for the astro test suite,
including temporary directory management, mock model creation, and test data setup.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

from astro.typings import RecordableModel


class MockTraceableModel(RecordableModel):
    """Mock implementation of TraceableModel for testing."""

    name: str = Field(description="Name of the mock model")
    value: int = Field(default=42, description="Test value")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class AnotherMockModel(RecordableModel):
    """Another mock model for testing type validation."""

    title: str = Field(description="Title of the model")
    description: str = Field(default="", description="Description")


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for testing."""
    return tmp_path


@pytest.fixture
def mock_model() -> MockTraceableModel:
    """Create a mock TraceableModel instance for testing."""
    return MockTraceableModel(name="test_model", value=100, metadata={"type": "test"})


@pytest.fixture
def mock_model_2() -> MockTraceableModel:
    """Create a second mock TraceableModel instance for testing."""
    return MockTraceableModel(
        name="test_model_2", value=200, metadata={"type": "test2"}
    )


@pytest.fixture
def another_mock_model() -> AnotherMockModel:
    """Create a different type of mock model for testing type validation."""
    return AnotherMockModel(title="Test Title", description="Test Description")


@pytest.fixture
def mock_model_dict(mock_model: MockTraceableModel) -> dict[str, Any]:
    """Get the dictionary representation of a mock model."""
    return mock_model.model_dump(mode="json")


@pytest.fixture
def populated_temp_dir(tmp_path: Path, mock_model: MockTraceableModel) -> Path:
    """Provide a temporary directory with pre-existing model files and index."""
    # Create model file
    model_file = tmp_path / mock_model.uid
    with open(model_file, "w") as f:
        json.dump(mock_model.model_dump(mode="json"), f)

    # Create index file
    index_file = tmp_path / "index"
    index_data = {mock_model.uid: str(model_file)}
    with open(index_file, "w") as f:
        json.dump(index_data, f)

    return tmp_path


@pytest.fixture
def corrupted_temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory with corrupted files for error testing."""
    # Create corrupted index file
    index_file = tmp_path / "index"
    with open(index_file, "w") as f:
        f.write("invalid json content")

    return tmp_path
