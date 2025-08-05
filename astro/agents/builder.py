from typing import Any, Generic, Optional, Self

from langchain.chat_models import init_chat_model

from astro.agents.base import (
    Input,
    Output,
    EffectModule,
    InputType,
    OutputType,
    StrMessage,
)
from astro.agents.effect import LLMModule
from astro.agents.memory import ChatMemory


# class AgentBuilder(Generic[InputType, OutputType]):
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
