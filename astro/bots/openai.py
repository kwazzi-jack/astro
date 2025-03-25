import os
from typing import Iterator

from langchain.globals import set_llm_cache
from langchain_core.messages.base import BaseMessageChunk
from langchain_community.cache import SQLiteCache
from langchain_openai.chat_models import ChatOpenAI


class Bot:
    def __init__(self, name: str = "astro"):

        # Verify OPENAI_API_KEY is set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Could not find OPEN_API_KEY environment variable.")

        # Setup/load response cache
        set_llm_cache(SQLiteCache(database_path=".astro.db"))

        # Proceed to create object
        self.name = name
        self.bot = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=1024,
            timeout=None,
            max_retries=2,
            streaming=True,
        )

    def chat(self, messages: str | list[str] | list[dict]) -> str:
        return self.bot.invoke(messages).content

    def chat_stream(
        self, messages: str | list[str] | list[dict]
    ) -> Iterator[BaseMessageChunk]:
        return self.bot.stream(messages)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    testbot = Bot()
    result = testbot.chat("What is the first 10 decimals of pi?")
    print(f"{result=}")
