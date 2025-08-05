from typing import Any, Optional
import uuid

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AnyMessage,
    ToolMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from astro.agents.base import Input, Output, StrMessage
from astro.agents.effect import LLMInput, LLMModule, LLMOutput, LLMParams
from astro.agents.memory import ChatMemory
from astro.utilities.uids import create_uid, named_uid_factory




class MessageOutput(Output):
    """AI message output"""

    # Metadata
    data_name: str = Field("MessageOutput", description="Name of the message output")
    data_uid: str = Field(
        default_factory=named_uid_factory("MessageOutput"),
        description="Unique identifier for the message output",
    )
    data_description: str = Field(
        "Message output from the agent",
        description="Description of the message output",
    )

    # Attributes
    message: str = Field(..., description="Agent message to the actor")


class ReactiveAgent:
    def __init__(
        self,
        model_name: str,
        model_params: Optional[LLMParams] = None,
        system_prompt: Optional[StrMessage] = None,
        name: str = "ReactiveAgent",
        description: Optional[str] = None,
        uid: Optional[str] = None,
    ):
        self.model_name = model_name
        if system_prompt is None:
            self.system_prompt = "You are a helpful AI assistant."
        else:
            self.system_prompt = system_prompt
        self.name = name
        self.description = description or ""
        self.uid = uid or create_uid()
        self.llm = LLMModule(

        )
        self.memory = ChatMemory()

    def act(self, inputs: MessageInput) -> MessageOutput:
        self.memory.clear()
        self.memory.add_system_message(self.system_prompt)
        self.memory.add_user_message(inputs.message)

        llm_outputs = self.llm.invoke(LLMInput(messages=self.memory._messages))
        self.memory.add(llm_outputs.message)
        outputs = self.memory.last
        if isinstance(outputs, str):
            return MessageOutput(message=outputs)
        elif isinstance(outputs, list) and len(outputs):
            return MessageOutput(message=". ".join(map(str, outputs)))
        else:
            return MessageOutput(message=str(outputs))


class StaticChatAgent:
    """
    Includes:
    > LLM
    > Chat Memory

    INPUT = MessageInput, OUTPUT = MessageOutput
    """

    def __init__(
        self,
        model_name: str,
        model_params: Optional[LLMParams] = None,
        system_prompt: Optional[StrMessage] = None,
        welcome_message: Optional[StrMessage] = None,
        name: str = "static-chat-agent",
        description: Optional[str] = None,
        uid: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_params = model_params or LLMParams()
        if system_prompt is None:
            self.system_prompt = "You are a helpful AI assistant."
        else:
            self.system_prompt = system_prompt
        if welcome_message is None:
            self.welcome_message = "Hi, how can I help?"
        self.name = name
        self.description = description or ""
        self.uid = uid or f"{self.__class__.__name__}{str(uuid.uuid4())[:12]}"
        self.llm = LLMModule(
            init_chat_model(
                model=model_name,
                temperature=self.model_params.temperature,
                max_tokens=self.model_params.max_tokens,
                max_retries=self.model_params.max_retries,
                timeout=self.model_params.timeout,
            )
        )
        self.memory = ChatMemory()
        self.memory.add_system_message(self.system_prompt)
        self.memory.add_ai_message(self.welcome_message)

    def act(self, inputs: MessageInput) -> MessageOutput:
        self.memory.add_user_message(inputs.message)
        llm_outputs = self.llm.invoke(LLMInput(messages=self.memory._messages))
        self.memory.add(llm_outputs.message)
        outputs = self.memory.last
        if isinstance(outputs, str):
            return MessageOutput(message=outputs)
        elif isinstance(outputs, list) and len(outputs):
            return MessageOutput(message=". ".join(map(str, outputs)))
        else:
            return MessageOutput(message=str(outputs))


class ChatAgent:
    """
    Includes:
    > LLM
    > Chat Memory
    > Tools

    INPUT = MessageInput, OUTPUT = MessageOutput

    Optional:
    > Work Memory
    > Short Memory
    > Long Memory
    """


class ReactiveTextAgent:
    """
    Includes:
    > LLM

    INPUT = TextInput, OUTPUT = StructuredOutput
    """


class BasicTextAgent:
    """
    Includes:
    > LLM
    > Chat Memory

    INPUT = TextInput, OUTPUT = StructuredOutput
    """


class TextAgent:
    """
    Includes:
    > LLM
    > Chat Memory
    > Tools

    INPUT = TextInput, OUTPUT = StructuredOutput

    Optional:
    > Work Memory
    > Short Memory
    > Long Memory
    """


class ChatWorker:
    """
    > LLM
    > Chat Memory
    > Tools
    > Communication
    > Optional: Work Memory, Short Memory, Long Memory
    """


class StructuredAgent:
    pass


# class ChatAgentBuilder:
#     name: Optional[str] = None
#     description: Optional[str] = None
#     effect: Optional[EffectModule] = None
#     chat_memory: Optional[ChatMemory] = None
#     input_schema: Optional[InputType] = None
#     output_schema: Optional[OutputType] = None

#     def __init__(self, name: Optional[str] = None):
#         self.name = name or "Agent"

#     def add_effect(self, module: EffectModule) -> Self:
#         if self.effect is None:
#             self.effect = module
#         else:
#             raise ValueError(
#                 f"There already exists an effect module to add: {self.effect.__class__.__name__}"
#             )
#         return self

#     def add_llm(
#         self,
#         model: str,
#         temperature: Optional[float] = None,
#         max_tokens: Optional[int] = None,
#         max_retries: Optional[int] = None,
#         timeout: Optional[int] = None,
#     ) -> Self:
#         # Already exists some effect module so cannot add an LLM
#         if self.effect is not None:
#             raise ValueError(
#                 f"Cannot add LLM. There already exists an effect module to add: {self.effect.__class__.__name__}"
#             )
#         # Somehow, we have chat_memory but no LLM
#         if self.chat_memory is not None:
#             raise RuntimeError(
#                 "Something went wrong - No effect module but chat memory exists"
#             )

#         # Add llm model
#         llm = init_chat_model(
#             model=model,
#             temperature=temperature or 0.7,
#             max_tokens=max_tokens or 2048,
#             max_retries=max_retries or 3,
#             timeout=timeout or 60,
#         )
#         self.effect = LLMModule(llm)

#         # Add chat memory
#         self.chat_memory = ChatMemory()

#         return self

#     def add_welcome_message(self, content: StrMessage) -> Self:
#         if self.chat_memory is None:
#             raise ValueError(
#                 "Cannot add a welcome message if there is no chat memory. Add an LLM first with `.add_llm(...)`"
#             )
#         # TODO: Add warning if this is the first message, i.e. no system messsages
#         self.chat_memory.add_ai_message(content)
#         return self

#     def add_system_prompt(self, content: StrMessage) -> Self:
#         if self.chat_memory is None:
#             raise ValueError(
#                 "Cannot add a system prompt if there is no chat memory. Add an LLM first with `.add_llm(...)`"
#             )
#         self.chat_memory.add_system_message(content)
#         return self

#     def set_input(self, input_schema: InputType) -> Self:
#         if self.input_schema is not None:
#             raise ValueError(
#                 f"There already exists an input schema to add: {self.input_schema}"
#             )
#         self.input_schema = input_schema
#         return self

#     def set_output(self, output_schema: OutputType) -> Self:
#         if self.input_schema is not None:
#             raise ValueError(
#                 f"There already exists an output schema to add: {self.output_schema}"
#             )
#         self.output_schema = output_schema
#         return self

#     def build(self) -> Agent: ...


class ChatState(BaseModel):
    """State object for `ChatAgent`"""

    messages: list[AnyMessage] = Field(
        default_factory=list, description="Stores messages for conversation."
    )

    def add_system_message(self, message: StrMessage):
        """Add system-role message to list"""
        self.messages.append(SystemMessage(message))

    def add_user_message(self, message: StrMessage):
        """Add user-role message to list."""
        self.messages.append(HumanMessage(message))

    def add_bot_message(self, message: StrMessage):
        """Add assistant-role message to list."""
        self.messages.append(AIMessage(message))

    def add_tool_message(self, message: StrMessage):
        """Add tool-role message to list"""
        self.messages.append(ToolMessage(message))

    def last_message(self, n=1) -> StrMessage:
        """Return last message or last n-messages."""
        return self.messages[-n].content


# class ChatAgent:
#     """Agent dedicated to primary chat functions."""

#     def __init__(self, model: str = "openai:gpt-4o-mini"):
#         self.llm = init_chat_model(model=model, temperature=0.7)
#         self.graph = self._build_graph()
#         self.state = ChatState()
#         self.state.add_system_message(
#             f"""
#             You are Astro.,
#             You are a radio interferometry and astronomy software and data science assistant.,
#             You are South African.
#             You are a member of the centre for *Radio Astronomy Techniques and Technologies* (RATT) group.
#             They are based in the Physics & Electronics Department at Rhodes University, Makhanda, South Africa.,
#             Your primary language will be english, but if the user responds in another, do so accordingly.,
#             You will only use metric standard units.,
#             You will be helping with software-related problems and running software for users.,
#             You will be helping with educating the science, physics and maths behind radio astronomy and radio interferometry,

#             Here is contextual information:

#             - The date and time is .
#             """
#         )
#         self.state.add_bot_message(f"Good, Brian. How can I assist?")

#     def _build_graph(self) -> CompiledStateGraph:
#         """Build chat graph"""
#         graph = StateGraph(ChatState)
#         graph.add_node("chat", self._chat_node)
#         graph.add_edge(START, "chat")
#         graph.add_edge("chat", END)
#         return graph.compile()

#     def _chat_node(self, state: ChatState) -> ChatState:
#         """Generate response using LLM"""
#         response = self.llm.invoke(state.messages)
#         state.add_bot_message(response.content)
#         return state

#     def chat(self, message: str) -> StrMessage:
#         """Chat with the agent"""
#         self.state.add_user_message(message)
#         self.state = ChatState(**self.graph.invoke(self.state))
#         return self.last_message()

#     def last_message(self) -> StrMessage:
#         """Get last message"""
#         return self.state.last_message()


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    basic_chat_agent = StaticChatAgent("openai:gpt-4o-mini")

    try:
        print("====ReactiveAgent====")
        agent = ReactiveAgent("openai:gpt-4o-mini")
        user_input = input("User: ")
        while user_input.strip() != "q":
            ai_output = agent.act(MessageInput(message=user_input.strip())).message
            print("ReactiveAgent:", ai_output)
            user_input = input("User: ")
    except KeyboardInterrupt:
        print("Interrupted. Exiting...")
        exit(0)

    try:
        print("====BasicChatAgent====")
        agent = StaticChatAgent("openai:gpt-4o-mini")
        user_input = input("User: ")
        while user_input.strip() != "q":
            ai_output = agent.act(MessageInput(message=user_input.strip())).message
            print("BasicChatAgent:", ai_output)
            user_input = input("User: ")
    except KeyboardInterrupt:
        print("Interrupted. Exiting...")
        exit(0)
