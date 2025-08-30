"""
Examples of pytest features and testing patterns.

This module demonstrates various pytest features that are useful for testing:
- Built-in fixtures (tmp_path, monkeypatch, capsys, etc.)
- Parametrized tests
- Fixtures with different scopes
- Testing exceptions and side effects
"""

import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from astro.paths import ModelFileStore
from tests.conftest import MockTraceableModel


class TestPytestBuiltinFixtures:
    """Demonstrate pytest's built-in fixtures."""

    def test_tmp_path_fixture(self, tmp_path: Path):
        """Test using pytest's tmp_path fixture."""
        # tmp_path is a Path object to a temporary directory
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello pytest!")

        assert test_file.read_text() == "Hello pytest!"
        assert test_file.parent == tmp_path

    def test_monkeypatch_env_vars(self, monkeypatch: pytest.MonkeyPatch):
        """Test environment variable mocking with monkeypatch."""
        monkeypatch.setenv("TEST_VAR", "test_value")

        assert os.getenv("TEST_VAR") == "test_value"

    def test_monkeypatch_attributes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Test attribute patching with monkeypatch."""
        store = ModelFileStore(tmp_path, MockTraceableModel)

        # Mock the _save_index method
        mock_save = Mock()
        monkeypatch.setattr(store, "_save_index", mock_save)

        # Now when we call setitem, it should use our mock
        store["test"] = tmp_path / "test"
        mock_save.assert_called_once()

    def test_capsys_capture_output(self, capsys: pytest.CaptureFixture[str]):
        """Test stdout/stderr capture with capsys."""
        print("Hello stdout")
        print("Error message", file=__import__("sys").stderr)

        captured = capsys.readouterr()
        assert "Hello stdout" in captured.out
        assert "Error message" in captured.err


class TestParametrizedTests:
    """Demonstrate parametrized testing."""

    @pytest.mark.parametrize(
        "name,value,expected_type",
        [
            ("model1", 100, int),
            ("model2", 200, int),
            ("model3", 42, int),
        ],
    )
    def test_mock_model_creation(self, name: str, value: int, expected_type: type):
        """Test creating models with different parameters."""
        model = MockTraceableModel(name=name, value=value)

        assert model.name == name
        assert model.value == value
        assert isinstance(model.value, expected_type)

    @pytest.mark.parametrize(
        "invalid_type",
        [
            123,
            "string",
            [],
            {},
            None,
        ],
    )
    def test_store_invalid_types(self, tmp_path: Path, invalid_type):
        """Test that ModelFileStore rejects invalid model types."""
        with pytest.raises((AttributeError, ValueError)):
            ModelFileStore(tmp_path, invalid_type)  # type: ignore

    @pytest.mark.parametrize("file_count", [1, 5, 10, 25])
    def test_store_multiple_models(self, tmp_path: Path, file_count: int):
        """Test store performance with different numbers of models."""
        store = ModelFileStore(tmp_path, MockTraceableModel)
        models = [
            MockTraceableModel(name=f"model_{i}", value=i) for i in range(file_count)
        ]

        for model in models:
            store.add_model(model)

        assert len(store) == file_count

        # Verify all models can be retrieved
        for model in models:
            retrieved = store.get_model(model.uid)
            assert retrieved is not None
            assert retrieved.name == model.name


class TestFixtureScopes:
    """Demonstrate different fixture scopes."""

    @pytest.fixture(scope="class")
    def class_scoped_store(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> ModelFileStore:
        """Class-scoped fixture that persists across all tests in this class."""
        temp_dir = tmp_path_factory.mktemp("class_scoped")
        return ModelFileStore(temp_dir, MockTraceableModel)

    @pytest.fixture(scope="function")
    def sample_model(self) -> MockTraceableModel:
        """Function-scoped fixture (default) - new instance for each test."""
        return MockTraceableModel(name="sample", value=999)

    def test_class_fixture_first(
        self, class_scoped_store: ModelFileStore, sample_model: MockTraceableModel
    ):
        """First test using class-scoped fixture."""
        class_scoped_store.add_model(sample_model)
        assert len(class_scoped_store) == 1

    def test_class_fixture_second(self, class_scoped_store: ModelFileStore):
        """Second test - should see data from first test due to class scope."""
        # The store should still contain the model from the previous test
        assert len(class_scoped_store) == 1


class TestExceptionHandling:
    """Demonstrate testing exception scenarios."""

    def test_exception_with_match(self, tmp_path: Path):
        """Test exception with message matching."""
        store = ModelFileStore(tmp_path, MockTraceableModel)

        with pytest.raises(KeyError, match="No entry for key"):
            _ = store["nonexistent"]

    def test_exception_context_manager(self, tmp_path: Path):
        """Test exception using context manager for additional assertions."""
        store = ModelFileStore(tmp_path, MockTraceableModel)

        with pytest.raises(ValueError) as exc_info:
            store[123] = tmp_path / "test"  # type: ignore

        # Can make additional assertions about the exception
        assert "Expected `key` to be `str`" in str(exc_info.value)
        assert "Got `int` instead" in str(exc_info.value)

    def test_no_exception_raised(self, tmp_path: Path):
        """Test that no exception is raised in normal operation."""
        store = ModelFileStore(tmp_path, MockTraceableModel)
        model = MockTraceableModel(name="test", value=42)

        # This should not raise any exception
        store.add_model(model)
        retrieved = store.get_model(model.uid)

        assert retrieved is not None


class TestMockingPatterns:
    """Demonstrate different mocking patterns."""

    def test_mock_with_side_effect(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test using mock with side effects."""
        store = ModelFileStore(tmp_path, MockTraceableModel)

        # Mock the open function to raise an error
        mock_open = Mock(side_effect=IOError("Disk full"))
        monkeypatch.setattr("builtins.open", mock_open)

        model = MockTraceableModel(name="test", value=42)

        # This should raise an error because our mock open will fail
        with pytest.raises(
            IOError, match="Error occurred while loading from index file"
        ):
            store[model.uid] = tmp_path / model.uid

    def test_mock_return_value(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test using mock with return values."""
        # Mock Path.exists to always return True
        mock_exists = Mock(return_value=True)
        monkeypatch.setattr(Path, "exists", mock_exists)

        fake_path = Path("/fake/path")
        assert fake_path.exists()  # Will return True due to our mock
        mock_exists.assert_called_once()


class TestDataDrivenTests:
    """Demonstrate data-driven testing patterns."""

    # Test data as class attribute
    MODEL_TEST_DATA = [
        {"name": "alpha", "value": 1, "metadata": {"type": "test"}},
        {"name": "beta", "value": 2, "metadata": {"type": "prod"}},
        {"name": "gamma", "value": 3, "metadata": {"env": "dev"}},
    ]

    @pytest.mark.parametrize("model_data", MODEL_TEST_DATA)
    def test_model_serialization(self, tmp_path: Path, model_data: dict):
        """Test model serialization with different data sets."""
        model = MockTraceableModel(**model_data)
        store = ModelFileStore(tmp_path, MockTraceableModel)

        store.add_model(model)
        retrieved = store.get_model(model.uid)

        assert retrieved is not None
        assert retrieved.name == model_data["name"]
        assert retrieved.value == model_data["value"]
        assert retrieved.metadata == model_data["metadata"]

    def test_json_roundtrip(self, tmp_path: Path):
        """Test JSON serialization roundtrip."""
        original_model = MockTraceableModel(
            name="roundtrip_test", value=999, metadata={"nested": {"key": "value"}}
        )

        # Serialize to JSON
        json_data = original_model.model_dump(mode="json")
        json_str = json.dumps(json_data)

        # Deserialize from JSON
        loaded_data = json.loads(json_str)
        restored_model = MockTraceableModel.model_validate(loaded_data)

        assert restored_model.name == original_model.name
        assert restored_model.value == original_model.value
        assert restored_model.metadata == original_model.metadata


# Demonstrate pytest markers
@pytest.mark.unit
class TestMarkerExamples:
    """Demonstrate pytest markers."""

    @pytest.mark.slow
    def test_marked_as_slow(self):
        """This test is marked as slow."""
        # Simulate slow operation
        import time

        time.sleep(0.01)  # Very short sleep for demo
        assert True

    @pytest.mark.filesystem
    def test_marked_filesystem(self, tmp_path: Path):
        """This test is marked as filesystem-dependent."""
        test_file = tmp_path / "marker_test.txt"
        test_file.write_text("marker test")
        assert test_file.exists()

    @pytest.mark.integration
    def test_marked_integration(self, tmp_path: Path):
        """This test is marked as integration test."""
        # Integration test example
        store = ModelFileStore(tmp_path, MockTraceableModel)
        model = MockTraceableModel(name="integration", value=123)

        # Full workflow test
        store.add_model(model)
        assert model.uid in store

        retrieved = store.get_model(model.uid)
        assert retrieved.name == "integration"

        store.remove_model(model)
        assert model.uid not in store
