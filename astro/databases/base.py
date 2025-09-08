# from sqlmodel import

from typing import Protocol, TypeVar

from astro.typings import ImmutableRecordType, RecordableModelType

if __name__ == "__main__":
    # Create all tables
    from pathlib import Path

    from sqlalchemy import create_engine
    from sqlmodel import Session, SQLModel, create_engine, select

    from astro.databases.models import LLMConfigRecord
    from astro.errors import DBIntegrityError
    from astro.llms.base import LLMConfig
    from astro.typings import ModelName, ModelProvider

    db_path = Path("store.db")
    engine = create_engine(f"sqlite:///{db_path.resolve().as_posix()}")

    SQLModel.metadata.create_all(engine)

    config_1 = LLMConfig.for_conversational("openai")
    config_2 = LLMConfig.for_conversational("ollama")
    config_3 = config_1.model_copy(deep=True)
    config_4 = LLMConfig.for_conversational("ollama")
    config_5 = config_1.model_copy(update={"model_name": ModelName.GPT_4O}, deep=True)

    record_1 = LLMConfigRecord.from_model(config_1)
    record_2 = LLMConfigRecord.from_model(config_2)
    record_3 = LLMConfigRecord.from_model(config_3)
    record_4 = LLMConfigRecord.from_model(config_4)
    record_5 = LLMConfigRecord.from_model(config_5)

    # Insert into DB
    # with Session(engine) as session:
    #     session.add(record_1)
    #     session.add(record_2)
    #     # session.add(record_3)
    #     # session.add(record_4)
    #     session.add(record_5)
    #     session.commit()

    # Query and display all entries from LLMConfigRecord
    with Session(engine) as session:
        results = session.exec(
            select(LLMConfigRecord).where(
                LLMConfigRecord.model_provider == ModelProvider.OPENAI
            )
        ).all()
        for i, record in enumerate(results):
            print(f"{i + 1}. Record {record.uid}.")
            model = record.to_model()
            print(f"    > Created model {model.to_hex()}")

            if model == config_1:
                print("    > Matches config 1")
                print("        > Original ")
                print("        ", model)
                print()
                print("        > Config 1 ")
                print("        ", config_1)
                print()

            if model == config_2:
                print("    > Matches config 2")
                print("        > Original ")
                print("        ", model)
                print()
                print("        > Config 2 ")
                print("        ", config_2)
                print()

            if model == config_3:
                print("    > Matches config 3")
                print("        > Original ")
                print("        ", model)
                print()
                print("        > Config 3 ")
                print("        ", config_3)
                print()

            if model == config_4:
                print("    > Matches config 4")
                print("        > Original ")
                print("        ", model)
                print()
                print("        > Config 4 ")
                print("        ", config_4)
                print()

            if model == config_5:
                print("    > Matches config 5")
                print("        > Original ")
                print("        ", model)
                print()
                print("        > Config 5 ")
                print("        ", config_5)
                print()
            print()
