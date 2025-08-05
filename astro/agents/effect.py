from typing import Literal, Optional, Self, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import Field

from astro.agents.base import DataModel, Input, Output, EffectModule, StrMessage


class LLMMessage(DataModel):
    """A message for the LLMModule, containing a sequence of messages."""

    content: StrMessage = Field(
        ..., description="Message to be sent to the language model."
    )
    role: Literal["human", "ai", "system", "tool"] = Field(
        ..., description="Role of the message sender."
    )

    def to_langchain(self) -> BaseMessage:
        """Convert to a LangChain BaseMessage."""
        if self.role == "human":
            return HumanMessage(content=self.content)
        elif self.role == "ai":
            return AIMessage(content=self.content)
        elif self.role == "system":
            return SystemMessage(content=self.content)
        elif self.role == "tool":
            return ToolMessage(content=self.content, tool_call_id=self.uid)
        else:
            raise ValueError(f"Unknown role: {self.role}")

    @classmethod
    def from_langchain(cls, message: BaseMessage) -> Self:
        """Create an LLMMessage from a LangChain BaseMessage."""
        if isinstance(message, HumanMessage):
            return cls(content=message.content, role="human")
        elif isinstance(message, AIMessage):
            return cls(content=message.content, role="ai")
        elif isinstance(message, SystemMessage):
            return cls(content=message.content, role="system")
        elif isinstance(message, ToolMessage):
            return cls(content=message.content, role="tool", uid=message.tool_call_id)
        else:
            raise ValueError(f"Unknown LangChain message type: {type(message)}")

    def __str__(self) -> str:
        """Return string representation of the message."""
        return f"{self.role.upper()}: {self.content}"


class LLMInput(Input):
    """Input for the LLMModule, containing a sequence of messages."""

    messages: Sequence[LLMMessage] = Field(
        default_factory=list, description="List of LLM (MTD) Messages"
    )

    def to_langchain(self) -> list[BaseMessage]:
        """Convert to a list of LangChain BaseMessages."""
        return [msg.to_langchain() for msg in self.messages]


class LLMOutput(Output):
    """Output from the LLMModule, containing the AI's response."""

    message: LLMMessage = Field(
        ..., description="Response from AI based on input messages, if any."
    )


class LLMParams(Input):
    """LLM Model Parameters"""

    model_name: str = "openai:gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2048
    max_retries: int = 3
    timeout: int = 60


class LLMModule(EffectModule[LLMInput, LLMOutput]):
    """An EffectModule that invokes a LangChain compatible language model."""

    def __init__(self, params: Optional[LLMParams] = None):
        self.params = params or LLMParams()
        self.llm = init_chat_model(
            model=self.params.model_name,
            temperature=self.params.temperature,
            max_tokens=self.params.max_tokens,
            max_retries=self.params.max_retries,
            timeout=self.params.timeout,
        )

    def invoke(self, inputs: LLMInput) -> LLMOutput:
        """
        Invokes the language model with the given inputs.

        Args:
            inputs: An LLMInput object containing the messages to send to the model.

        Returns:
            An LLMOutput object containing the model's response.
        """
        outputs = self.llm.invoke(inputs.to_langchain())
        return LLMOutput(message=LLMMessage.from_langchain(outputs))


if __name__ == "__main__":
    input1 = LLMInput(messages=[LLMMessage(content="Hello, world!", role="ai")])
    input2 = LLMInput(
        messages=[LLMMessage(content=[{"text": "How are you?"}], role="human")]
    )
    input3 = LLMInput(messages=input1.messages)

    print(f"{input1=}")
    print(f"{input2=}")
    print(f"{input3=}")

    output1 = LLMOutput(message=LLMMessage(content="I'm fine, thank you!", role="ai"))
    output2 = LLMOutput(
        message=LLMMessage(content=["This is a list message."], role="human")
    )
    output3 = LLMOutput(message=output1.message)

    print(f"{output1=}")
    print(f"{output2=}")
    print(f"{output3=}")
