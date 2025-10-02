# --- Internal Imports ---
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

# --- External Imports ---
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine, inspect, select

from astro.agents.base import AgentConfig

# --- Local Imports ---
from astro.databases.models import AgentConfigRecord, ChatRecord, LLMConfigRecord
from astro.llms.base import LLMConfig
from astro.loggings.base import get_loggy
from astro.paths import REPOSITORY_DIR
from astro.typings import (
    ImmutableRecord,
    ImmutableRecordType,
    RecordableModel,
    RecordableModelType,
    StrPath,
    options_to_str,
    type_name,
)

# --- Globals ---
_loggy = get_loggy(__file__)


# --- Helper Functions ---
def _database_exists(db_path: Path) -> bool:
    db_file = db_path.expanduser().resolve()
    return db_file.exists() and db_file.is_file()


def _contains_tables(engine: Engine, tables: Sequence[type[ImmutableRecord]]) -> bool:
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    return all(table.__tablename__ in existing_tables for table in tables)


class RepositoryManager:
    def __init__(
        self,
        db_name: str,
        verbose: bool = False,
    ) -> None:
        _loggy.debug(f"Creating new RepositoryManager for database {db_name!r}")
        if REPOSITORY_DIR is None:
            raise _loggy.CreationError(
                object_type=RepositoryManager,
                reason="Respository directory is not set",
            )

        self._name = db_name
        self._file_path = REPOSITORY_DIR / f"{db_name}.db"
        self._db_path = f"sqlite:///{self._file_path}"
        # NB: Manually set the list of record tables here
        # Brian: Not sure how to handle type annotation of `_record_tables`
        # to handle generics too
        self._record_tables: list[type[ImmutableRecord[Any]]] = [
            ChatRecord,
            AgentConfigRecord,
            LLMConfigRecord,
        ]
        self._record_table_names: set[str] = {
            "chatrecord",
            "agentconfigrecord",
            "llmconfigrecord",
        }

        if not _database_exists(self._file_path):
            _loggy.info(
                f"Database {self._file_path.name!r} does not exist. Creating new one."
            )

        self._engine = self._create_db_engine(echo=verbose)
        self._create_tables()

    def _create_db_engine(self, echo: bool = False) -> Engine:
        return create_engine(self._db_path, echo=echo)

    def _create_tables(self) -> None:
        _loggy.debug(f"Creating tables for database {self._name!r}")
        try:
            SQLModel.metadata.create_all(self._engine)
        except Exception as error:
            raise _loggy.CreationError(
                object_type=RepositoryManager,
                reason="Failed to create tables",
                caused_by=error,
            )

    def _select_existing_record_hashes(
        self, session: Session, *records: ImmutableRecord[Any]
    ) -> set[int]:
        existing_hashes = set()
        for record in records:
            record_hash = record.record_hash
            statement = (
                select(type(record))
                .where(type(record).record_hash == record_hash)
                .limit(1)
            )
            result = session.exec(statement).first()
            if result is not None:
                existing_hashes.add(record_hash)
        return existing_hashes

    def store_llm_records(self, *models: LLMConfig) -> None:
        # Input validation
        if len(models) == 0:
            raise _loggy.ValueError("#TODO Implement proper error handling here")

        for i, model in enumerate(models):
            if not isinstance(model, LLMConfig):
                raise _loggy.ExpectedElementTypeError(
                    collection_var_name="models",
                    expected=LLMConfigRecord,
                    got=type(model),
                    index_or_key=i,
                    with_value=model,
                )

        duplicate_pairs: list[tuple[int, int, int]] = []
        for i, model in enumerate(models):
            for j, other_model in enumerate(models, start=i + 1):
                if model == other_model:
                    duplicate_pairs.append((i, j, hash(model)))

        if len(duplicate_pairs) > 0:
            msg = options_to_str([f"arg {i} == arg {j} ({hash_value})" for i, j, hash_value in duplicate_pairs])
            _loggy.warning(f"Found duplicate pairs while creating LLMConfig records: {msg}")

        with Session(self._engine) as session:
            try:
                # Check for existing records to avoid duplicates
                existing_hashes =
                new_records = [
                    record
                    for record in models
                    if record.record_hash not in existing_hashes
                ]

                if len(new_records) == 0:
                    _loggy.debug(f"All {len(models)} records already exist in database")
                    return

                session.add_all(new_records)
                session.commit()
            except Exception as error:
                raise _loggy.SQLIntegrityError(
                    operation="create_records",
                    reason="Failed to create records",
                    database=self._name,
                    table=self._record_tables,
                    caused_by=error,
                )

    def retrieve_llm_records(self, *record_hashes: int) -> dict[int, LLMConfig]:
        # Input validation
        if len(record_hashes) == 0:
            raise _loggy.ValueError("#TODO Implement proper error handling here")

        for i, record_hash in enumerate(record_hashes):
            if not isinstance(record_hash, int):
                raise _loggy.ExpectedElementTypeError(
                    collection_var_name="record_hashes",
                    expected=int,
                    got=type(record_hash),
                    index_or_key=i,
                    with_value=record_hash,
                )

        # Fetch all LLMConfigRecord that match hashes
        retrieved_records: dict[int, LLMConfigRecord] = {}
        with Session(self._engine) as session:
            try:
                for record_hash in record_hashes:
                    # Where hashes match, select first
                    statement = select(LLMConfigRecord).where(
                        LLMConfigRecord.record_hash == record_hash
                    )
                    result = session.exec(statement).first()
                    if result is not None:
                        retrieved_records[result.record_hash] = result

            except Exception as error:
                raise _loggy.SQLRetrievalError(
                    operation="retrieve_records",
                    reason="Failed to retrieve records",
                    database=self._name,
                    table=LLMConfigRecord,
                    caused_by=error,
                )

        # Convert to LLMConfigs
        retrieved_llm_configs: dict[int, LLMConfig] = {}
        for record_hash, record in retrieved_records.items():
            try:
                llm_config = record.to_model()
            except Exception as error:
                raise _loggy.ValueError(
                    "#TODO Implement proper error handling here", caused_by=error
                )

            # Check if built LLMConfig matches original record hash
            if hash(llm_config) != record_hash:
                raise _loggy.ValueError("#TODO Implement proper error handling here")

            # Add to result
            retrieved_llm_configs[record_hash] = llm_config

        # Return result
        return retrieved_llm_configs

    def retrieve_llm_record(self, record_hash: int) -> LLMConfig:
        llm_configs = self.retrieve_llm_records(record_hash)
        return llm_configs[record_hash]

    def retrieve_agent_records(self, *record_hashes: int) -> dict[int, AgentConfig]:
        # Input validation
        if len(record_hashes) == 0:
            raise _loggy.ValueError("#TODO Implement proper error handling here")

        for i, record_hash in enumerate(record_hashes):
            if not isinstance(record_hash, int):
                raise _loggy.ExpectedElementTypeError(
                    collection_var_name="record_hashes",
                    expected=int,
                    got=type(record_hash),
                    index_or_key=i,
                    with_value=record_hash,
                )

        # Fetch all LLMConfigRecord that match hashes
        retrieved_records: dict[int, AgentConfigRecord] = {}
        with Session(self._engine) as session:
            try:
                for record_hash in record_hashes:
                    # Where hashes match, select first
                    statement = select(AgentConfigRecord).where(
                        AgentConfigRecord.record_hash == record_hash
                    )
                    result = session.exec(statement).first()
                    if result is not None:
                        retrieved_records[result.record_hash] = result

            except Exception as error:
                raise _loggy.SQLRetrievalError(
                    operation="retrieve_records",
                    reason="Failed to retrieve records",
                    database=self._name,
                    table=LLMConfigRecord,
                    caused_by=error,
                )

        # Depends on LLMConfigs, fetch all by hashes
        llm_config_hashes = [
            retrieved_record.llm_config_hash
            for retrieved_record in retrieved_records.values()
        ]
        llm_configs = self.retrieve_llm_records(*llm_config_hashes)

        # Convert to AgentConfigs (with LLMConfigs)
        retrieved_agent_configs: dict[int, AgentConfig] = {}
        for record_hash, record in retrieved_records.items():
            try:
                llm_config = llm_configs[record.llm_config_hash]
                agent_config = record.to_model(llm_config)
            except Exception as error:
                raise _loggy.ValueError(
                    "#TODO Implement proper error handling here", caused_by=error
                )

            # Check if built LLMConfig matches original record hash
            if hash(llm_config) != record_hash:
                raise _loggy.ValueError("#TODO Implement proper error handling here")

            # Add to result
            retrieved_agent_configs[record_hash] = agent_config

        # Return result
        return retrieved_agent_configs

    def retrieve_agent_record(self, record_hash: int) -> AgentConfig:
        agent_configs = self.retrieve_agent_records(record_hash)
        return agent_configs[record_hash]


if __name__ == "__main__":
    from langchain_core.messages import AIMessage

    from astro.agents.base import AgentConfig
    from astro.app.handler import Chat

    agent = AgentConfig.for_chat("openai")
    chat = Chat(agent_config=agent, messages=[AIMessage("Hello, world!")])
    llm = agent.llm_config
    llm_record = LLMConfigRecord.from_model(llm)
    agent_record = AgentConfigRecord.from_model(agent)
    chat_record = ChatRecord.from_model(chat)
    print(llm_record)
    print(agent_record)
    print(chat_record)
    repo = RepositoryManager("test_db", verbose=True)

    repo.create_records(llm_record, agent_record, chat_record)
    records = repo.retrieve_llm_records(llm_record.record_hash)
    print("\n\n")
    for record in records:
        print(record)

    llm2 = records[0]
    print("\n\n")
    print(f"{(llm == llm2)=}")
