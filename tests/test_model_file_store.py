"""
Comprehensive tests for ModelFileStore class.

This module tests all functionality of the ModelFileStore class including:
- Initialization and validation
- CRUD operations
- File system interactions
- Error handling and edge cases
- Index file management
- Type safety and validation
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from astro.paths import ModelFileStore
from tests.conftest import AnotherMockModel, MockTraceableModel


class TestModelFileStoreInitialization:
    """Test ModelFileStore initialization and basic properties."""

    @pytest.mark.unit
    def test_init_creates_root_directory(self, temp_dir: Path):
        """Test that initialization creates root directory if it doesn't exist."""
        new_dir = temp_dir / "new_store"
        assert not new_dir.exists()

        ModelFileStore(new_dir, MockTraceableModel)

        assert new_dir.exists()
        assert new_dir.is_dir()

    @pytest.mark.unit
    def test_init_with_existing_directory(self, temp_dir: Path):
        """Test initialization with existing directory."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        assert store.root_dir == temp_dir
        assert store.model_type == MockTraceableModel
        assert store.name == temp_dir.name

    @pytest.mark.unit
    def test_init_creates_empty_index_file(self, temp_dir: Path):
        """Test that initialization creates an empty index file."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        assert store.index_file.exists()
        assert store.index_file == temp_dir / "index"

        # Check index file content
        with open(store.index_file, "r") as f:
            content = json.load(f)
        assert content == {}

    @pytest.mark.unit
    def test_init_loads_existing_index(
        self, populated_temp_dir: Path, mock_model: MockTraceableModel
    ):
        """Test that initialization loads existing index file."""
        store = ModelFileStore(populated_temp_dir, MockTraceableModel)

        assert len(store) == 1
        assert mock_model.uid in store

    @pytest.mark.unit
    def test_init_filters_nonexistent_files(self, temp_dir: Path):
        """Test that initialization filters out non-existent files from index."""
        # Create index with non-existent file
        index_file = temp_dir / "index"
        fake_path = temp_dir / "nonexistent_file"
        index_data = {"fake_uid": str(fake_path)}

        with open(index_file, "w") as f:
            json.dump(index_data, f)

        store = ModelFileStore(temp_dir, MockTraceableModel)

        assert len(store) == 0
        assert "fake_uid" not in store

    @pytest.mark.unit
    def test_init_invalid_model_type_raises_error(self, temp_dir: Path):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(
            AttributeError, match="'str' object has no attribute '__name__'"
        ):
            ModelFileStore(temp_dir, dict)  # type: ignore

        with pytest.raises(
            AttributeError, match="'str' object has no attribute '__name__'"
        ):
            ModelFileStore(temp_dir, "not_a_type")  # type: ignore

    @pytest.mark.unit
    def test_properties(self, temp_dir: Path):
        """Test all property accessors."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        assert store.name == temp_dir.name
        assert store.root_dir == temp_dir
        assert store.model_type == MockTraceableModel
        assert store.index_file == temp_dir / "index"


class TestModelFileStoreIndexOperations:
    """Test index file operations."""

    @pytest.mark.filesystem
    def test_save_index_creates_file(
        self, temp_dir: Path, mock_model: MockTraceableModel
    ):
        """Test that _save_index creates index file correctly."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        store._index_map[mock_model.uid] = temp_dir / mock_model.uid

        store._save_index()

        assert store.index_file.exists()
        with open(store.index_file, "r") as f:
            content = json.load(f)
        assert mock_model.uid in content

    @pytest.mark.filesystem
    def test_load_index_reads_file(
        self, populated_temp_dir: Path, mock_model: MockTraceableModel
    ):
        """Test that _load_index reads index file correctly."""
        store = ModelFileStore(populated_temp_dir, MockTraceableModel)
        loaded_index = store._load_index()

        assert mock_model.uid in loaded_index
        assert isinstance(loaded_index[mock_model.uid], Path)

    @pytest.mark.unit
    def test_save_index_io_error(self, temp_dir: Path):
        """Test _save_index handles IO errors."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        with patch("builtins.open", side_effect=IOError("Permission denied")):
            with pytest.raises(
                IOError, match="Error occurred while loading from index file"
            ):
                store._save_index()

    @pytest.mark.unit
    def test_load_index_io_error(self, corrupted_temp_dir: Path):
        """Test _load_index handles corrupted files."""
        with pytest.raises(IOError, match="Error occurred while saving to index file"):
            ModelFileStore(corrupted_temp_dir, MockTraceableModel)


class TestModelFileStoreObjectOperations:
    """Test model object save/load operations."""

    @pytest.mark.filesystem
    def test_save_object(self, temp_dir: Path, mock_model: MockTraceableModel):
        """Test _save_object saves model correctly."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        store[mock_model.uid] = temp_dir / mock_model.uid

        store._save_object(mock_model)

        file_path = temp_dir / mock_model.uid
        assert file_path.exists()

        with open(file_path, "r") as f:
            content = json.load(f)
        assert content["name"] == mock_model.name
        assert content["value"] == mock_model.value

    @pytest.mark.filesystem
    def test_load_object(self, temp_dir: Path, mock_model: MockTraceableModel):
        """Test _load_object loads model correctly."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        store[mock_model.uid] = temp_dir / mock_model.uid
        store._save_object(mock_model)

        loaded_model = store._load_object(mock_model.uid)

        assert isinstance(loaded_model, MockTraceableModel)
        assert loaded_model.uid == mock_model.uid
        assert loaded_model.name == mock_model.name
        assert loaded_model.value == mock_model.value

    @pytest.mark.unit
    def test_save_object_wrong_type(
        self, temp_dir: Path, another_mock_model: AnotherMockModel
    ):
        """Test _save_object rejects wrong model type."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        with pytest.raises(ValueError, match="Expected `obj` to be"):
            store._save_object(another_mock_model)  # type: ignore

    @pytest.mark.unit
    def test_load_object_invalid_key_type(self, temp_dir: Path):
        """Test _load_object rejects non-string keys."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        with pytest.raises(ValueError, match="Expected `key` to be"):
            store._load_object(123)  # type: ignore

    @pytest.mark.unit
    def test_save_object_io_error(self, temp_dir: Path, mock_model: MockTraceableModel):
        """Test _save_object handles IO errors."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        store[mock_model.uid] = temp_dir / mock_model.uid

        with patch("builtins.open", side_effect=IOError("Disk full")):
            with pytest.raises(IOError, match=f"Error while saving.*{mock_model.uid}"):
                store._save_object(mock_model)

    @pytest.mark.unit
    def test_load_object_io_error(self, temp_dir: Path):
        """Test _load_object handles IO errors."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        fake_key = "nonexistent_key"
        store[fake_key] = temp_dir / "nonexistent_file"

        with pytest.raises(IOError, match=f"Error while loading {fake_key}"):
            store._load_object(fake_key)


class TestModelFileStoreDictInterface:
    """Test dictionary-like interface operations."""

    @pytest.mark.unit
    def test_getitem(self, temp_dir: Path, mock_model: MockTraceableModel):
        """Test __getitem__ retrieves paths correctly."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        expected_path = temp_dir / mock_model.uid
        store[mock_model.uid] = expected_path

        result = store[mock_model.uid]
        assert result == expected_path

    @pytest.mark.unit
    def test_getitem_invalid_key_type(self, temp_dir: Path):
        """Test __getitem__ rejects non-string keys."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        with pytest.raises(
            ValueError, match="Expected `key` to be `str`. Got `int` instead"
        ):
            _ = store[123]  # type: ignore

    @pytest.mark.unit
    def test_getitem_missing_key(self, temp_dir: Path):
        """Test __getitem__ raises KeyError for missing keys."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        with pytest.raises(KeyError, match="No entry for key"):
            _ = store["nonexistent_key"]

    @pytest.mark.filesystem
    def test_setitem(self, temp_dir: Path):
        """Test __setitem__ stores paths correctly."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        test_key = "test_key"
        test_path = temp_dir / "test_file"

        store[test_key] = test_path

        assert store[test_key] == test_path
        assert store.index_file.exists()

    @pytest.mark.unit
    def test_setitem_invalid_key_type(self, temp_dir: Path):
        """Test __setitem__ rejects non-string keys."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        with pytest.raises(
            ValueError, match="Expected `key` to be `str`. Got `int` instead"
        ):
            store[123] = temp_dir / "file"  # type: ignore

    @pytest.mark.unit
    def test_setitem_invalid_value_type(self, temp_dir: Path):
        """Test __setitem__ rejects non-Path values."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        with pytest.raises(
            ValueError, match="Expected `value` to be `Path`. Got `str` instead"
        ):
            store["key"] = "not_a_path"  # type: ignore

    @pytest.mark.filesystem
    def test_delitem(self, temp_dir: Path, mock_model: MockTraceableModel):
        """Test __delitem__ removes entries and files."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        file_path = temp_dir / mock_model.uid

        # Create file and add to store
        file_path.touch()
        store[mock_model.uid] = file_path
        assert file_path.exists()

        del store[mock_model.uid]

        assert mock_model.uid not in store
        assert not file_path.exists()

    @pytest.mark.unit
    def test_delitem_invalid_key_type(self, temp_dir: Path):
        """Test __delitem__ rejects non-string keys."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        with pytest.raises(
            ValueError, match="Expected `key` to be `str`. Got `int` instead"
        ):
            del store[123]  # type: ignore

    @pytest.mark.unit
    def test_delitem_missing_key(self, temp_dir: Path):
        """Test __delitem__ raises KeyError for missing keys."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        with pytest.raises(KeyError, match="No entry for key"):
            del store["nonexistent_key"]

    @pytest.mark.unit
    def test_iter(
        self,
        temp_dir: Path,
        mock_model: MockTraceableModel,
        mock_model_2: MockTraceableModel,
    ):
        """Test __iter__ iterates over keys."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        store[mock_model.uid] = temp_dir / mock_model.uid
        store[mock_model_2.uid] = temp_dir / mock_model_2.uid

        keys = list(store)
        assert len(keys) == 2
        assert mock_model.uid in keys
        assert mock_model_2.uid in keys

    @pytest.mark.unit
    def test_contains_with_string(self, temp_dir: Path, mock_model: MockTraceableModel):
        """Test __contains__ with string keys."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        store[mock_model.uid] = temp_dir / mock_model.uid

        assert mock_model.uid in store
        assert "nonexistent_key" not in store

    @pytest.mark.unit
    def test_contains_with_model(
        self,
        temp_dir: Path,
        mock_model: MockTraceableModel,
        mock_model_2: MockTraceableModel,
    ):
        """Test __contains__ with model objects."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        store[mock_model.uid] = temp_dir / mock_model.uid

        assert mock_model in store
        assert mock_model_2 not in store

    @pytest.mark.unit
    def test_contains_invalid_type(self, temp_dir: Path):
        """Test __contains__ rejects invalid types."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        with pytest.raises(ValueError, match="Expected `key_or_obj` to be"):
            _ = 123 in store  # type: ignore

    @pytest.mark.unit
    def test_len(
        self,
        temp_dir: Path,
        mock_model: MockTraceableModel,
        mock_model_2: MockTraceableModel,
    ):
        """Test __len__ returns correct count."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        assert len(store) == 0

        store[mock_model.uid] = temp_dir / mock_model.uid
        assert len(store) == 1

        store[mock_model_2.uid] = temp_dir / mock_model_2.uid
        assert len(store) == 2


class TestModelFileStoreViewMethods:
    """Test dictionary view methods."""

    @pytest.mark.unit
    def test_keys(
        self,
        temp_dir: Path,
        mock_model: MockTraceableModel,
        mock_model_2: MockTraceableModel,
    ):
        """Test keys() returns correct view."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        store[mock_model.uid] = temp_dir / mock_model.uid
        store[mock_model_2.uid] = temp_dir / mock_model_2.uid

        keys = store.keys()
        assert mock_model.uid in keys
        assert mock_model_2.uid in keys
        assert len(keys) == 2

    @pytest.mark.unit
    def test_values(self, temp_dir: Path, mock_model: MockTraceableModel):
        """Test values() returns correct view."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        expected_path = temp_dir / mock_model.uid
        store[mock_model.uid] = expected_path

        values = store.values()
        assert expected_path in values
        assert len(values) == 1

    @pytest.mark.unit
    def test_items(self, temp_dir: Path, mock_model: MockTraceableModel):
        """Test items() returns correct view."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        expected_path = temp_dir / mock_model.uid
        store[mock_model.uid] = expected_path

        items = store.items()
        assert (mock_model.uid, expected_path) in items
        assert len(items) == 1


class TestModelFileStorePublicMethods:
    """Test public methods of ModelFileStore."""

    @pytest.mark.filesystem
    def test_add_model(self, temp_dir: Path, mock_model: MockTraceableModel):
        """Test add() stores model correctly."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        store.add_model(mock_model)

        assert mock_model.uid in store
        assert len(store) == 1

        # Verify file was created
        expected_path = temp_dir / mock_model.uid
        assert expected_path.exists()

        # Verify content
        with open(expected_path, "r") as f:
            content = json.load(f)
        assert content["name"] == mock_model.name

    @pytest.mark.unit
    def test_add_invalid_type(
        self, temp_dir: Path, another_mock_model: AnotherMockModel
    ):
        """Test add() rejects wrong model type."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        with pytest.raises(
            ValueError,
            match="Expected `value` to be `MockTraceableModel`. Got `AnotherMockModel` instead",
        ):
            store.add_model(another_mock_model)  # type: ignore

    @pytest.mark.filesystem
    def test_remove_by_key(self, temp_dir: Path, mock_model: MockTraceableModel):
        """Test remove() with string key."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        store.add_model(mock_model)

        file_path = temp_dir / mock_model.uid
        assert file_path.exists()

        store.remove_model(mock_model.uid)

        assert mock_model.uid not in store
        assert not file_path.exists()

    @pytest.mark.filesystem
    def test_remove_by_model(self, temp_dir: Path, mock_model: MockTraceableModel):
        """Test remove() with model object."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        store.add_model(mock_model)

        file_path = temp_dir / mock_model.uid
        assert file_path.exists()

        store.remove_model(mock_model)

        assert mock_model.uid not in store
        assert not file_path.exists()

    @pytest.mark.unit
    def test_remove_invalid_type(self, temp_dir: Path):
        """Test remove() rejects invalid types."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        with pytest.raises(ValueError, match="Expected `key_or_obj` to be"):
            store.remove_model(123)  # type: ignore

    @pytest.mark.filesystem
    def test_get_model_existing(self, temp_dir: Path, mock_model: MockTraceableModel):
        """Test get_model() retrieves existing model."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        store.add_model(mock_model)

        retrieved = store.get_model(mock_model.uid)

        assert isinstance(retrieved, MockTraceableModel)
        assert retrieved.uid == mock_model.uid
        assert retrieved.name == mock_model.name

    @pytest.mark.unit
    def test_get_model_nonexistent(self, temp_dir: Path):
        """Test get_model() returns default for nonexistent key."""
        store = ModelFileStore(temp_dir, MockTraceableModel)

        result = store.get_model("nonexistent")
        assert result is None

        result = store.get_model("nonexistent", "default")
        assert result == "default"

    @pytest.mark.unit
    def test_get_model_io_error(self, temp_dir: Path, mock_model: MockTraceableModel):
        """Test get_model() returns default on IO error."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        store[mock_model.uid] = temp_dir / "nonexistent_file"

        result = store.get_model(mock_model.uid, "default")
        assert result == "default"

    @pytest.mark.filesystem
    def test_clear(
        self,
        temp_dir: Path,
        mock_model: MockTraceableModel,
        mock_model_2: MockTraceableModel,
    ):
        """Test clear() removes all entries and files."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        store.add_model(mock_model)
        store.add_model(mock_model_2)

        file1 = temp_dir / mock_model.uid
        file2 = temp_dir / mock_model_2.uid
        assert file1.exists()
        assert file2.exists()
        assert len(store) == 2

        store.clear()

        assert len(store) == 0
        assert not file1.exists()
        assert not file2.exists()


class TestModelFileStoreStringRepresentation:
    """Test string representation methods."""

    @pytest.mark.unit
    def test_str(self, temp_dir: Path):
        """Test __str__ method."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        result = str(store)

        assert "ModelFileStore" in result
        assert "MockTraceableModel" in result
        assert str(temp_dir) in result

    @pytest.mark.unit
    def test_repr(self, temp_dir: Path):
        """Test __repr__ method."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        result = repr(store)

        assert result.startswith("ModelFileStore(")
        assert "root_dir=" in result
        assert "model_type=" in result


class TestModelFileStoreEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.filesystem
    def test_concurrent_access_simulation(
        self, temp_dir: Path, mock_model: MockTraceableModel
    ):
        """Test behavior with simulated concurrent access."""
        store1 = ModelFileStore(temp_dir, MockTraceableModel)
        ModelFileStore(temp_dir, MockTraceableModel)  # store2 for simulation

        store1.add_model(mock_model)

        # New instance should see the change after reloading
        store3 = ModelFileStore(temp_dir, MockTraceableModel)
        assert mock_model.uid in store3

    @pytest.mark.filesystem
    def test_persistence_across_instances(
        self, temp_dir: Path, mock_model: MockTraceableModel
    ):
        """Test that data persists across ModelFileStore instances."""
        # First instance
        store1 = ModelFileStore(temp_dir, MockTraceableModel)
        store1.add_model(mock_model)
        del store1

        # Second instance should load existing data
        store2 = ModelFileStore(temp_dir, MockTraceableModel)
        assert len(store2) == 1
        assert mock_model.uid in store2

        retrieved = store2.get_model(mock_model.uid)
        assert retrieved.name == mock_model.name

    @pytest.mark.filesystem
    def test_large_dataset_performance(self, temp_dir: Path):
        """Test performance with larger dataset."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        models = [MockTraceableModel(name=f"model_{i}", value=i) for i in range(100)]

        # Add all models
        for model in models:
            store.add_model(model)

        assert len(store) == 100

        # Verify all can be retrieved
        for model in models:
            retrieved = store.get_model(model.uid)
            assert retrieved is not None
            assert retrieved.name == model.name

    @pytest.mark.filesystem
    @pytest.mark.slow
    def test_file_permissions(self, temp_dir: Path, mock_model: MockTraceableModel):
        """Test behavior with file permission issues."""
        store = ModelFileStore(temp_dir, MockTraceableModel)
        store.add_model(mock_model)

        # Make index file read-only
        store.index_file.chmod(0o444)

        # Should raise IOError when trying to modify
        with pytest.raises(IOError):
            store.add_model(MockTraceableModel(name="another", value=999))

        # Restore permissions for cleanup
        store.index_file.chmod(0o644)
