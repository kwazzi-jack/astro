"""
astro/stores.py

Modern buffer system for RecordableModel objects with async database persistence.

Author(s):
    - Brian Welman
Date: 2025-09-02
License: MIT

Description:
    This module provides a buffer system that serves as a staging area between application
    usage and database storage. It leverages the improved hashing system from RecordableModel
    and provides asynchronous database synchronization with proper locking mechanisms.

    The Store class is generic and supports whitelisted RecordableModel types that can be
    converted to ImmutableRecord database entries via the RecordConverter protocol.

Dependencies:
    - asyncio
    - sqlmodel
    - sqlalchemy
    - threading
"""

import asyncio
import threading
from collections.abc import Iterator, KeysView, ValuesView
from typing import Generic

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from astro.errors import ExpectedVarType
from astro.loggings import get_logger
from astro.typings import (
    ImmutableRecord,
    ImmutableRecordType,
    RecordableModel,
    RecordableModelType,
    RecordConverter,
    type_name,
)
from astro.utilities.timing import get_datetime_now

# Global logger
_logger = get_logger(__file__)


class Store(Generic[RecordableModelType, ImmutableRecordType]):
    """Generic buffer system for RecordableModel objects with async database persistence.

    Provides an in-memory staging area for CRUD operations on RecordableModel objects,
    with asynchronous background synchronization to a SQLModel database. Uses hash-based
    identification and supports proper concurrency control.

    Attributes:
        model_type: The RecordableModel subclass this store manages
        record_type: The ImmutableRecord subclass for database storage
        name: Unique name identifier for this store

    Args:
        model_type: RecordableModel subclass to store
        record_type: ImmutableRecord subclass for database persistence
        name: Unique identifier for this store instance
        flush_interval: Seconds between automatic database flushes (default: 30)
        batch_size: Maximum number of records to process per flush (default: 100)
    """

    def __init__(
        self,
        model_type: type[RecordableModelType],
        record_type: type[ImmutableRecordType],
        name: str,
        flush_interval: float = 30.0,
        batch_size: int = 100,
    ):
        _logger.debug(f"Initializing Store '{name}' for {model_type.__name__}")

        # Validate types
        if not (
            isinstance(model_type, type) and issubclass(model_type, RecordableModel)
        ):
            raise ExpectedVarType(
                var_name="model_type",
                got=model_type,
                expected=RecordableModel,
            )

        if not (
            isinstance(record_type, type) and issubclass(record_type, ImmutableRecord)
        ):
            raise ExpectedVarType(
                var_name="record_type",
                got=record_type,
                expected=ImmutableRecord,
            )

        # Validate that record_type implements RecordConverter protocol methods
        if not (
            hasattr(record_type, "from_model") and hasattr(record_type, "to_model")
        ):
            raise ValueError(
                f"Record type {type_name(record_type)} must implement `RecordConverter` protocol "
                f"(`from_model` and `to_model` methods)"
            )

        if not isinstance(name, str):
            raise ValueError("Store name must be a non-empty string")

        if len(name.strip()) == 0:
            raise ValueError("Store name cannot be empty")

        # Core attributes
        self._model_type = model_type
        self._record_type = record_type
        self._name = name.strip()

        # Buffer configuration
        self._flush_interval = max(1.0, flush_interval)
        self._batch_size = max(1, batch_size)

        # In-memory buffer - keyed by hash hex string
        self._buffer: dict[str, RecordableModelType] = {}
        self._dirty_keys: set[str] = set()  # Track which objects need database sync

        # Concurrency control
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()

        # Background synchronization state
        self._sync_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        self._last_flush = get_datetime_now()

        _logger.info(f"Store `{self._name}` initialized for {type_name(model_type)}")

    @property
    def model_type(self) -> type[RecordableModelType]:
        """The RecordableModel subclass this store manages."""
        return self._model_type

    @property
    def record_type(self) -> type[ImmutableRecordType]:
        """The ImmutableRecord subclass for database storage."""
        return self._record_type

    @property
    def name(self) -> str:
        """Unique name identifier for this store."""
        return self._name

    def _get_key(self, model: RecordableModelType) -> str:
        """Get hash-based key for a model object."""
        return model.to_hex()

    def __len__(self) -> int:
        """Return the number of objects in the buffer."""
        with self._lock:
            return len(self._buffer)

    def __contains__(self, key_or_model: str | RecordableModelType) -> bool:
        """Check if a key or model is present in the buffer."""
        if isinstance(key_or_model, str):
            key = key_or_model
        elif isinstance(key_or_model, self._model_type):
            key = self._get_key(key_or_model)
        else:
            raise ExpectedVarType(
                var_name="key_or_model",
                got=type(key_or_model),
                expected=(str, self._model_type),
            )

        with self._lock:
            return key in self._buffer

    def __iter__(self) -> Iterator[str]:
        """Iterate over the hash keys in the buffer."""
        with self._lock:
            return iter(list(self._buffer.keys()))

    def keys(self) -> KeysView[str]:
        """Return a view of the hash keys in the buffer."""
        with self._lock:
            return self._buffer.keys()

    def values(self) -> ValuesView[RecordableModelType]:
        """Return a view of the models in the buffer."""
        with self._lock:
            return self._buffer.values()

    def __str__(self) -> str:
        """Return a user-friendly string representation."""
        return (
            f"Store[{self._model_type.__name__}](name='{self._name}', size={len(self)})"
        )

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        return (
            f"Store(model_type={self._model_type.__name__}, "
            f"record_type={self._record_type.__name__}, "
            f"name='{self._name}', "
            f"size={len(self)})"
        )

    def get(self, key: str, default=None) -> RecordableModelType | None:
        """Get a model object by its hash key.

        Args:
            key: The hash hex string key
            default: Value to return if key not found

        Returns:
            The model object if found, otherwise default
        """
        if not isinstance(key, str):
            raise ExpectedVarType(var_name="key", got=type(key), expected=str)

        with self._lock:
            return self._buffer.get(key, default)

    def add(self, model: RecordableModelType) -> str:
        """Add a model object to the buffer.

        Args:
            model: The RecordableModel instance to add

        Returns:
            The hash key for the added model

        Raises:
            ValueError: If model is not the correct type
        """
        if not isinstance(model, self._model_type):
            raise ExpectedVarType(
                var_name="model", got=type(model), expected=self._model_type
            )

        key = self._get_key(model)
        _logger.debug(f"Adding model to `{self._name}` store with key: `{key}`")

        with self._lock:
            self._buffer[key] = model
            self._dirty_keys.add(key)

        _logger.info(f"Added model `{key}` to `{self._name}` store")
        return key

    def update(self, model: RecordableModelType) -> str:
        """Update or add a model object in the buffer.

        Args:
            model: The RecordableModel instance to update

        Returns:
            The hash key for the updated model
        """
        # update is the same as add for hash-based systems
        # since the key is derived from content
        return self.add(model)

    def remove(
        self, key_or_model: str | RecordableModelType
    ) -> RecordableModelType | None:
        """Remove a model object from the buffer.

        Args:
            key_or_model: Either the hash key or the model object itself

        Returns:
            The removed model object if found, otherwise None
        """
        if isinstance(key_or_model, str):
            key = key_or_model
        elif isinstance(key_or_model, self._model_type):
            key = self._get_key(key_or_model)
        else:
            raise ExpectedVarType(
                var_name="key_or_model",
                got=type(key_or_model),
                expected=(str, self._model_type),
            )

        _logger.debug(f"Removing model from {self._name} store with key: {key}")

        with self._lock:
            model = self._buffer.pop(key, None)
            self._dirty_keys.discard(key)

        if model:
            _logger.info(f"Removed model `{key}` from `{self._name}` store")
        else:
            _logger.warning(
                f"Model `{key}` not found in `{self._name}` store for removal"
            )

        return model

    def clear(self):
        """Clear all objects from the buffer."""
        _logger.debug(f"Clearing all objects from `{self._name}` store")

        with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            self._dirty_keys.clear()

        _logger.info(f"Cleared `{count}` objects from `{self._name}` store")

    def get_dirty_count(self) -> int:
        """Get the number of objects that need database synchronization."""
        with self._lock:
            return len(self._dirty_keys)

    def mark_clean(self, key: str):
        """Mark a specific object as synchronized with the database."""
        with self._lock:
            self._dirty_keys.discard(key)

    def get_dirty_objects(
        self, limit: int | None = None
    ) -> list[tuple[str, RecordableModelType]]:
        """Get objects that need database synchronization.

        Args:
            limit: Maximum number of objects to return

        Returns:
            List of (key, model) tuples for dirty objects
        """
        with self._lock:
            dirty_keys = list(self._dirty_keys)
            if limit:
                dirty_keys = dirty_keys[:limit]

            return [
                (key, self._buffer[key]) for key in dirty_keys if key in self._buffer
            ]

    def set_database_engine(self, engine):
        """Set the SQLAlchemy engine for database operations."""
        self._database_engine = engine
        _logger.debug(f"Database engine set for `{self._name}` store")

    def start_background_sync(self):
        """Start the background synchronization task."""
        if self._sync_task and not self._sync_task.done():
            _logger.warning(f"Background sync already running for `{self._name}` store")
            return

        self._shutdown_event.clear()
        self._sync_task = asyncio.create_task(self._sync_worker())
        _logger.info(f"Started background sync for `{self._name}` store")

    async def stop_background_sync(self):
        """Stop the background synchronization task."""
        if not self._sync_task or self._sync_task.done():
            _logger.debug(f"No background sync running for `{self._name}` store")
            return

        _logger.debug(f"Stopping background sync for `{self._name}` store")
        self._shutdown_event.set()

        try:
            await asyncio.wait_for(self._sync_task, timeout=10.0)
        except asyncio.TimeoutError:
            _logger.warning(
                f"Background sync for {self._name} store did not stop gracefully"
            )
            self._sync_task.cancel()

        _logger.info(f"Stopped background sync for {self._name} store")

    async def _sync_worker(self):
        """Background worker that periodically syncs dirty objects to database."""
        _logger.debug(f"Background sync worker started for {self._name} store")

        while not self._shutdown_event.is_set():
            try:
                # Check if we need to sync
                dirty_count = self.get_dirty_count()
                if dirty_count > 0:
                    _logger.debug(
                        f"Syncing {dirty_count} dirty objects from {self._name} store"
                    )
                    await self._sync_to_database()

                # Wait for next sync interval or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=self._flush_interval
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue syncing

            except Exception as error:
                _logger.error(
                    f"Error in background sync for {self._name} store: {error}"
                )
                # Wait a bit before retrying to avoid tight error loops
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=5.0)
                    break
                except asyncio.TimeoutError:
                    continue

        _logger.debug(f"Background sync worker stopped for {self._name} store")

    async def _sync_to_database(self):
        """Synchronize dirty objects to the database."""
        if not hasattr(self, "_database_engine") or not self._database_engine:
            _logger.warning(f"No database engine set for {self._name} store")
            return

        async with self._async_lock:
            # Get dirty objects in batches
            dirty_objects = self.get_dirty_objects(limit=self._batch_size)
            if not dirty_objects:
                return

            _logger.debug(
                f"Syncing {len(dirty_objects)} objects to database for {self._name} store"
            )

            # Convert models to records and sync to database
            try:
                with Session(self._database_engine) as session:
                    for key, model in dirty_objects:
                        await self._sync_single_object(session, key, model)

                    session.commit()
                    _logger.info(
                        f"Successfully synced {len(dirty_objects)} objects for {self._name} store"
                    )

            except Exception as error:
                _logger.error(f"Failed to sync objects for {self._name} store: {error}")
                raise

    async def _sync_single_object(
        self, session: Session, key: str, model: RecordableModelType
    ):
        """Sync a single object to the database."""
        try:
            # Convert model to database record using the RecordConverter protocol
            # The record_type is validated to implement from_model in __init__
            record = self._record_type.from_model(model)  # type: ignore[attr-defined]

            # Check if record already exists by hash
            existing = session.exec(
                select(self._record_type).where(
                    self._record_type.record_hash == record.record_hash
                )
            ).first()

            if existing:
                # Update existing record's access metadata
                existing.last_accessed_at = get_datetime_now()
                existing.access_count += 1
                session.add(existing)
            else:
                # Add new record
                session.add(record)

            # Mark as clean in memory
            self.mark_clean(key)

        except IntegrityError as error:
            _logger.warning(
                f"Integrity error syncing {key} for {self._name} store: {error}"
            )
            # Still mark as clean since the record exists in database
            self.mark_clean(key)
        except Exception as error:
            _logger.error(f"Error syncing {key} for {self._name} store: {error}")
            raise

    async def flush_to_database(self):
        """Manually flush all dirty objects to database."""
        _logger.debug(f"Manual flush requested for {self._name} store")
        await self._sync_to_database()


# Global store registry and management
_STORE_REGISTRY: dict[str, Store] = {}
_DATABASE_ENGINE = None


def setup_database_engine(database_url: str):
    """Set up the global database engine for all stores.

    Args:
        database_url: SQLAlchemy database URL
    """
    global _DATABASE_ENGINE
    from sqlalchemy import create_engine
    from sqlmodel import SQLModel

    _DATABASE_ENGINE = create_engine(database_url)

    # Create all tables
    SQLModel.metadata.create_all(_DATABASE_ENGINE)

    _logger.info(f"Database engine setup complete for: {database_url}")


def register_store(
    model_type: type[RecordableModelType],
    record_type: type[ImmutableRecordType],
    name: str | None = None,
    **kwargs,
) -> Store[RecordableModelType, ImmutableRecordType]:
    """Register and create a new store for the given model and record types.

    Args:
        model_type: RecordableModel subclass to store
        record_type: ImmutableRecord subclass for database persistence
        name: Optional name for the store (defaults to model_type.__name__)
        **kwargs: Additional arguments passed to Store constructor

    Returns:
        The created Store instance

    Raises:
        ValueError: If a store with the same name already exists
    """
    store_name = name or model_type.__name__

    if store_name in _STORE_REGISTRY:
        raise ValueError(f"Store '{store_name}' is already registered")

    # Create the store
    store = Store(
        model_type=model_type, record_type=record_type, name=store_name, **kwargs
    )

    # Set database engine if available
    if _DATABASE_ENGINE:
        store.set_database_engine(_DATABASE_ENGINE)

    # Register the store
    _STORE_REGISTRY[store_name] = store
    _logger.info(f"Registered store '{store_name}' for {model_type.__name__}")

    return store


def get_store(name: str) -> Store:
    """Get a registered store by name.

    Args:
        name: The name of the store to retrieve

    Returns:
        The Store instance

    Raises:
        KeyError: If no store with the given name exists
    """
    if name not in _STORE_REGISTRY:
        raise KeyError(
            f"No store registered with name '{name}'. Available stores: {list(_STORE_REGISTRY.keys())}"
        )

    return _STORE_REGISTRY[name]


def get_store_for_model(model_type: type[RecordableModel]) -> Store:
    """Get the registered store for a specific model type.

    Args:
        model_type: The RecordableModel subclass

    Returns:
        The Store instance for the model type

    Raises:
        KeyError: If no store is registered for the model type
    """
    store_name = model_type.__name__
    return get_store(store_name)


def list_stores() -> list[str]:
    """List all registered store names.

    Returns:
        List of store names
    """
    return list(_STORE_REGISTRY.keys())


def start_all_background_sync():
    """Start background synchronization for all registered stores."""
    _logger.info("Starting background sync for all stores")
    for store in _STORE_REGISTRY.values():
        if _DATABASE_ENGINE:
            store.set_database_engine(_DATABASE_ENGINE)
        store.start_background_sync()


async def stop_all_background_sync():
    """Stop background synchronization for all registered stores."""
    _logger.info("Stopping background sync for all stores")
    tasks = []
    for store in _STORE_REGISTRY.values():
        tasks.append(store.stop_background_sync())

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def flush_all_stores():
    """Manually flush all registered stores to database."""
    _logger.info("Flushing all stores to database")
    tasks = []
    for store in _STORE_REGISTRY.values():
        tasks.append(store.flush_to_database())

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def clear_all_stores():
    """Clear all objects from all registered stores."""
    _logger.warning("Clearing all stores")
    for store in _STORE_REGISTRY.values():
        store.clear()


class StoreManager:
    """High-level manager for store operations and lifecycle."""

    def __init__(self, database_url: str):
        """Initialize the store manager with a database connection.

        Args:
            database_url: SQLAlchemy database URL
        """
        self.database_url = database_url
        setup_database_engine(database_url)
        _logger.info("StoreManager initialized")

    def register_store(
        self,
        model_type: type[RecordableModelType],
        record_type: type[ImmutableRecordType],
        name: str | None = None,
        **kwargs,
    ) -> Store[RecordableModelType, ImmutableRecordType]:
        """Register a new store."""
        return register_store(model_type, record_type, name, **kwargs)

    def get_store(self, name: str) -> Store:
        """Get a registered store by name."""
        return get_store(name)

    def get_store_for_model(self, model_type: type[RecordableModel]) -> Store:
        """Get the store for a specific model type."""
        return get_store_for_model(model_type)

    def list_stores(self) -> list[str]:
        """List all registered stores."""
        return list_stores()

    def start_all_sync(self):
        """Start background sync for all stores."""
        start_all_background_sync()

    async def stop_all_sync(self):
        """Stop background sync for all stores."""
        await stop_all_background_sync()

    async def flush_all(self):
        """Flush all stores to database."""
        await flush_all_stores()

    def clear_all(self):
        """Clear all stores."""
        clear_all_stores()

    async def shutdown(self):
        """Gracefully shutdown all stores and cleanup."""
        _logger.info("Shutting down StoreManager")
        await self.stop_all_sync()
        await self.flush_all()
        _logger.info("StoreManager shutdown complete")


# Test example using LLMConfig
if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import track
    from rich.table import Table

    from astro.databases.models import LLMConfigRecord
    from astro.llms.base import LLMConfig

    console = Console()

    async def test_store_system():
        """Test the Store system with LLMConfig example."""
        console.print(Panel.fit("🚀 Store System Test", style="bold blue"))

        # Initialize store manager with SQLite database
        db_path = Path("test_store.db")
        if db_path.exists():
            db_path.unlink()  # Clean start

        console.print("📦 Initializing StoreManager with SQLite database...")
        manager = StoreManager(f"sqlite:///{db_path}")

        # Register LLMConfig store
        store = manager.register_store(
            model_type=LLMConfig,
            record_type=LLMConfigRecord,
            flush_interval=5.0,  # 5 seconds for testing
            batch_size=10,
        )

        console.print(f"✅ Registered store: [bold green]{store}[/bold green]")
        console.print(f"📋 Available stores: {manager.list_stores()}")

        # Create some LLMConfig instances
        console.print("\n🏗️  Creating LLMConfig instances...")
        config1 = LLMConfig.for_conversational("openai")
        config2 = LLMConfig.for_conversational("ollama")
        config3 = config1.model_copy(update={"temperature": 0.5}, deep=True)

        # Create a table for config details
        config_table = Table(title="Created Configurations")
        config_table.add_column("Config", style="cyan", no_wrap=True)
        config_table.add_column("UID", style="magenta")
        config_table.add_column("Provider", style="green")
        config_table.add_column("Model", style="yellow")
        config_table.add_column("Temperature", justify="right", style="blue")

        config_table.add_row(
            "Config 1",
            config1.uid[:12] + "...",
            str(config1.model_provider),
            str(config1.model_name),
            str(config1.temperature),
        )
        config_table.add_row(
            "Config 2",
            config2.uid[:12] + "...",
            str(config2.model_provider),
            str(config2.model_name),
            str(config2.temperature),
        )
        config_table.add_row(
            "Config 3",
            config3.uid[:12] + "...",
            str(config3.model_provider),
            str(config3.model_name),
            str(config3.temperature),
        )

        console.print(config_table)

        # Add configs to store
        console.print("\n💾 Adding configurations to store...")
        configs = [config1, config2, config3]
        keys = []

        for config in track(configs, description="Adding configs..."):
            key = store.add(config)
            keys.append(key)
            await asyncio.sleep(0.1)  # Small delay for visual effect

        # Store status
        store_status = Table(title="Store Status")
        store_status.add_column("Metric", style="cyan")
        store_status.add_column("Value", style="green")

        store_status.add_row("Store Size", str(len(store)))
        store_status.add_row("Dirty Objects", str(store.get_dirty_count()))
        store_status.add_row("Store Type", store.model_type.__name__)

        console.print(store_status)

        # Test retrieval
        console.print("\n🔍 Testing object retrieval...")
        retrieved1 = store.get(keys[0])
        if retrieved1 == config1:
            console.print("✅ Retrieved config matches original", style="green")
        else:
            console.print("❌ Retrieved config doesn't match", style="red")

        # Start background sync and wait a bit
        console.print("\n🔄 Starting background synchronization...")
        manager.start_all_sync()

        # Progress bar for waiting
        for _ in track(range(6), description="Waiting for sync..."):
            await asyncio.sleep(1)

        console.print(
            f"📊 Dirty objects after sync: [bold yellow]{store.get_dirty_count()}[/bold yellow]"
        )

        # Test manual flush
        console.print("\n⚡ Testing manual flush...")
        config4 = LLMConfig.for_conversational("anthropic")
        store.add(config4)

        console.print(
            f"📈 Dirty objects before flush: [red]{store.get_dirty_count()}[/red]"
        )
        await store.flush_to_database()
        console.print(
            f"📉 Dirty objects after flush: [green]{store.get_dirty_count()}[/green]"
        )

        # Test removal
        console.print("\n🗑️  Testing object removal...")
        removed = store.remove(keys[0])
        if removed == config1:
            console.print("✅ Successfully removed config", style="green")
        else:
            console.print("❌ Failed to remove config", style="red")

        console.print(
            f"📦 Store size after removal: [bold blue]{len(store)}[/bold blue]"
        )

        # Final cleanup
        console.print("\n🧹 Shutting down store system...")
        await manager.shutdown()

        # Cleanup test database
        if db_path.exists():
            db_path.unlink()

        console.print(
            Panel.fit(
                "🎉 Store System Test Completed Successfully!", style="bold green"
            )
        )

    # Run the test
    asyncio.run(test_store_system())

if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import track
    from rich.table import Table

    from astro.databases.models import LLMConfigRecord
    from astro.llms.base import LLMConfig

    console = Console()

    async def test_store_system():
        """Test the Store system with LLMConfig example."""
        console.print(Panel.fit("🚀 Store System Test", style="bold blue"))

        # 1. Initialize store manager with a clean SQLite database
        db_path = Path("test_store.db")
        if db_path.exists():
            db_path.unlink()

        console.print("📦 Initializing StoreManager with SQLite database...")
        manager = StoreManager(f"sqlite:///{db_path.resolve().as_posix()}")

        # 2. Register a store for LLMConfig with a short flush interval for testing
        store = manager.register_store(
            model_type=LLMConfig,
            record_type=LLMConfigRecord,
            flush_interval=5.0,  # Sync every 5 seconds
            batch_size=10,
        )
        console.print(f"✅ Registered store: [bold green]{store}[/bold green]")

        # 3. Create some LLMConfig instances to work with
        console.print("\n🏗️  Creating LLMConfig instances...")
        configs = [
            LLMConfig.for_conversational("openai"),
            LLMConfig.for_conversational("ollama"),
            LLMConfig.for_conversational("anthropic"),
        ]

        # 4. Add configs to the store, making them "dirty"
        console.print("\n💾 Adding configurations to the store...")
        keys = [store.add(c) for c in track(configs, description="Adding...")]
        console.print(
            f"📊 Store size: {len(store)}, Dirty objects: {store.get_dirty_count()}"
        )

        # 5. Start background sync and wait for it to flush the dirty objects
        console.print("\n🔄 Starting background synchronization...")
        manager.start_all_sync()
        for _ in track(range(6), description="Waiting for auto-sync..."):
            await asyncio.sleep(1)
        console.print(
            f"📊 Dirty objects after sync: [bold yellow]{store.get_dirty_count()}[/bold yellow]"
        )

        # 6. Test manual flush
        console.print("\n⚡ Testing manual flush...")
        new_config = LLMConfig.for_conversational("openai", temperature=0.9)
        store.add(new_config)
        console.print(
            f"📈 Dirty objects before flush: [red]{store.get_dirty_count()}[/red]"
        )
        await manager.flush_all()
        console.print(
            f"📉 Dirty objects after flush: [green]{store.get_dirty_count()}[/green]"
        )

        # 7. Test object retrieval and removal
        console.print("\n🔍 Testing object retrieval and removal...")
        retrieved = store.get(keys[0])
        if retrieved:
            console.print(f"✅ Retrieved: {retrieved.model_name}", style="green")
        removed = store.remove(keys[0])
        if removed:
            console.print(f"✅ Removed: {removed.model_name}", style="green")
        console.print(
            f"📦 Store size after removal: [bold blue]{len(store)}[/bold blue]"
        )

        # 8. Gracefully shut down the manager
        console.print("\n🧹 Shutting down store system...")
        await manager.shutdown()

        # 9. Clean up the test database file
        if db_path.exists():
            db_path.unlink()

        console.print(
            Panel.fit(
                "🎉 Store System Test Completed Successfully!", style="bold green"
            )
        )

    # Run the asynchronous test
    asyncio.run(test_store_system())
