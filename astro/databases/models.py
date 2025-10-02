# --- Internal Imports ---
import json
from typing import Self

# --- External Imports ---
from langchain_core.messages import messages_from_dict, messages_to_dict
from sqlmodel import Column, Field, ForeignKey, Integer, Text

# --- Local Imports ---
from astro.agents.base import AgentConfig
from astro.app.handler import Chat
from astro.llms.base import LLMConfig
from astro.llms.prompts import RegisteredPromptTemplate
from astro.loggings import get_loggy
from astro.typings import (
    ImmutableRecord,
    ModelName,
    ModelProvider,
    RecordableModel,
    get_class_from_import_path,
    get_class_import_path,
    secretify,
)

# --- Global ---
_loggy = get_loggy(__file__)


# --- Database Models ---


# Brian - Not recommended to store API-key -> Will be loaded when llm-model is created
class LLMConfigRecord(ImmutableRecord[LLMConfig], table=True):
    model_name: ModelName
    model_provider: ModelProvider
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=128)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    streaming: bool = Field(default=False)
    reasoning: bool = Field(default=False)
    thinking: bool = Field(default=False)
    max_retries: int = Field(default=3, gt=0, le=20)
    timeout: float = Field(default=60.0, gt=0.0, le=180.0)
    seed: int | None = Field(default=None, gt=0)
    count: int | None = Field(default=None, gt=0)
    context_size: int = Field(default=2048, ge=128)
    gpu_count: int | None = Field(default=None, gt=0)
    thread_count: int | None = Field(default=None, gt=0)
    keep_alive: bool = Field(default=False)

    @classmethod
    def from_model(cls, model: LLMConfig) -> "LLMConfigRecord":
        """Create record from LLMConfig"""
        _loggy.debug(f"Creating LLMConfigRecord from LLMConfig ({model.secret_uid})")
        try:
            return cls(record_hash=hash(model), **model.model_dump())
        except Exception as error:
            raise _loggy.CreationError(object_type=LLMConfigRecord, caused_by=error)

    def to_model(self, *_) -> LLMConfig:
        """Create model from record"""
        _loggy.debug(f"Creating LLMConfig from LLMConfigRecord ({self.record_hash})")

        try:
            # Create instance from record values
            model = LLMConfig(
                model_name=ModelName(self.model_name),
                model_provider=ModelProvider(self.model_provider),
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                streaming=self.streaming,
                reasoning=self.reasoning,
                thinking=self.thinking,
                max_retries=self.max_retries,
                timeout=self.timeout,
                seed=self.seed,
                count=self.count,
                context_size=self.context_size,
                gpu_count=self.gpu_count,
                thread_count=self.thread_count,
                keep_alive=self.keep_alive,
            )
        except Exception as error:
            raise _loggy.CreationError(object_type=LLMConfig, caused_by=error)

        # Hashes do not match -> not the same model that was saved
        model_hash = hash(model)
        if model_hash != self.record_hash:
            raise _loggy.RecordableIdentityError(
                record=model,
                other_record=self,
            )

        else:
            _loggy.debug(
                f"Hash check between LLMConfig ({model.secret_uid}) "
                f"and LLMConfigRecord ({self.secret_hash}) was successful"
            )
        # Return knowing it is the same model
        return model


class AgentConfigRecord(ImmutableRecord[AgentConfig], table=True):
    """Database record for AgentConfig with foreign key to LLMConfigRecord.

    Stores agent configuration with a normalized relationship to LLMConfig
    instead of nesting the entire configuration object.
    """

    name: str
    llm_config_hash: int = Field(
        sa_column=Column(Integer, ForeignKey("llmconfigrecord.record_hash"))
    )
    context_type_path: str | None = Field(default=None)
    system_prompt_template: str | None = Field(default=None)
    welcome_prompt_tag: str | None = Field(default=None)
    context_prompt_tag: str | None = Field(default=None)

    @classmethod
    def from_model(cls, model: AgentConfig) -> Self:
        """Create record from AgentConfig.

        Args:
            model (AgentConfig): The AgentConfig instance to convert

        Returns:
            AgentConfigRecord instance
        """
        _loggy.debug(
            f"Creating AgentConfigRecord from AgentConfig ({model.secret_uid})"
        )

        # Serialize context_type to module path string
        context_type_path = None
        if model.context_type is not None:
            context_type_path = get_class_import_path(model.context_type)

        # Convert RegisteredPromptTemplate enums to strings
        system_prompt = (
            model.system_prompt_template.value if model.system_prompt_template else None
        )
        welcome_prompt = (
            model.welcome_prompt_tag.value if model.welcome_prompt_tag else None
        )
        context_prompt = (
            model.context_prompt_tag.value if model.context_prompt_tag else None
        )

        # Get LLMConfig UID
        llm_config_hash = hash(model.llm_config)

        return cls(
            record_hash=hash(model),
            name=model.name,
            llm_config_hash=llm_config_hash,
            context_type_path=context_type_path,
            system_prompt_template=system_prompt,
            welcome_prompt_tag=welcome_prompt,
            context_prompt_tag=context_prompt,
        )

    def to_model(self, *dependencies: RecordableModel) -> "AgentConfig":
        """Create AgentConfig from record.

        Args:
            llm_config: The reconstructed LLMConfig instance

        Returns:
            AgentConfig instance
        """

        _loggy.debug(
            f"Creating AgentConfig from AgentConfigRecord ({self.secret_hash})"
        )

        # Input validation
        if len(dependencies) != 1 or not isinstance(dependencies[0], LLMConfig):
            raise _loggy.CreationError(
                object_type=AgentConfig,
                reason=(
                    "Expected a single LLMConfig dependency to reconstruct "
                    f"AgentConfig from AgentConfigRecord ({self.secret_hash})"
                ),
                record_hash=self.record_hash,
            )

        # Deserialize context_type from module path
        context_type = None
        if self.context_type_path:
            try:
                context_type = get_class_from_import_path(self.context_type_path)
            except Exception as error:
                raise _loggy.CreationError(
                    object_type=AgentConfig,
                    reason=(
                        f"Failed to import context_type '{self.context_type_path}' "
                        f"for AgentConfigRecord ({self.secret_hash})"
                    ),
                    caused_by=error,
                    record_hash=self.record_hash,
                )

        # Convert strings back to RegisteredPromptTemplate enums
        system_prompt = (
            RegisteredPromptTemplate(self.system_prompt_template)
            if self.system_prompt_template
            else None
        )
        welcome_prompt = (
            RegisteredPromptTemplate(self.welcome_prompt_tag)
            if self.welcome_prompt_tag
            else None
        )
        context_prompt = (
            RegisteredPromptTemplate(self.context_prompt_tag)
            if self.context_prompt_tag
            else None
        )

        # Get LLMConfig dependency
        llm_config = dependencies[0]
        if hash(llm_config) != self.llm_config_hash:
            raise _loggy.CreationError(
                object_type=AgentConfig,
                reason=(
                    f"LLMConfig dependency hash ({secretify(hash(llm_config))}) does not match "
                    f"llm_config_hash ({secretify(self.llm_config_hash)}) in "
                    f"AgentConfigRecord ({self.secret_hash})"
                ),
                self_hash=self.record_hash,
                record_hash=self.llm_config_hash,
                other_record_hash=hash(llm_config),
            )

        # Create instance from record values
        model = AgentConfig(
            name=self.name,
            llm_config=llm_config,
            context_type=context_type,
            system_prompt_template=system_prompt,
            welcome_prompt_tag=welcome_prompt,
            context_prompt_tag=context_prompt,
        )

        # Verify hash integrity{self.r
        model_hash = hash(model)
        if model_hash != self.record_hash:
            raise _loggy.RecordableIdentityError(record=model, other_record=self)
        else:
            _loggy.debug(
                f"Hash check between AgentConfig ({secretify(model.secret_uid)}) "
                f"and AgentConfigRecord ({secretify(self.secret_hash)}) was successful"
            )

        return model


class ChatRecord(ImmutableRecord[Chat], table=True):
    """Database record for Chat with foreign key to AgentConfigRecord.

    Stores chat conversation with normalized relationship to AgentConfig.
    Messages are serialized as JSON.
    """

    agent_config_hash: int = Field(
        sa_column=Column(Integer, ForeignKey("agentconfigrecord.record_hash"))
    )
    messages_json: str = Field(sa_column=Column(Text))

    @classmethod
    def from_model(cls, model: Chat) -> "ChatRecord":
        """Create record from Chat.

        Args:
            model: The Chat instance to convert
            agent_config_record_uid: The uid of the related AgentConfigRecord

        Returns:
            ChatRecord instance
        """

        _loggy.debug(f"Creating ChatRecord from Chat ({model.secret_uid})")

        # Serialize messages to JSON using LangChain utility
        try:
            messages_dict = messages_to_dict(model.messages)
            messages_json_str = json.dumps(messages_dict)
        except Exception as error:
            raise _loggy.CreationError(
                object_type=ChatRecord,
                reason="Failed to serialize messages",
                caused_by=error,
            )

        # Get AgentConfig UID
        agent_config_hash = hash(model.agent_config)

        return cls(
            record_hash=hash(model),
            agent_config_hash=agent_config_hash,
            messages_json=messages_json_str,
        )

    def to_model(self, *dependencies: RecordableModel) -> "Chat":
        """Create Chat from record.

        Args:
            agent_config: The reconstructed AgentConfig instance

        Returns:
            Chat instance
        """

        _loggy.debug(f"Creating Chat from ChatRecord ({self.record_hash})")

        # Input validation
        if len(dependencies) != 1 or not isinstance(dependencies[0], AgentConfig):
            raise _loggy.CreationError(
                object_type=Chat,
                reason=(
                    "Expected a single AgentConfig dependency to reconstruct "
                    "Chat from ChatRecord"
                ),
                record_hash=self.record_hash,
            )

        # Deserialize messages from JSON
        try:
            messages_dict = json.loads(self.messages_json)
            messages = messages_from_dict(messages_dict)
        except Exception as error:
            raise _loggy.CreationError(
                object_type=Chat,
                reason=(
                    f"Failed to deserialize messages JSON for ChatRecord ({self.record_hash})"
                ),
                caused_by=error,
                record_hash=self.record_hash,
                messages_json=self.messages_json,
            )

        # Get AgentConfig dependency
        agent_config = dependencies[0]
        if hash(agent_config) != self.agent_config_hash:
            raise _loggy.CreationError(
                object_type=AgentConfig,
                reason=(
                    f"AgentConfig dependency hash ({secretify(hash(agent_config))}) does not match "
                    f"agent_config_hash ({secretify(self.agent_config_hash)}) in "
                    f"AgentConfigRecord ({self.secret_hash})"
                ),
                self_hash=self.record_hash,
                record_hash=self.agent_config_hash,
                other_record_hash=hash(agent_config),
            )

        # Create instance from record values
        model = Chat(
            agent_config=agent_config,
            messages=messages,
        )

        # Verify hash integrity
        model_hash = hash(model)
        if model_hash != self.record_hash:
            raise _loggy.RecordableIdentityError(record=model, other_record=self)
        else:
            _loggy.debug(
                f"Hash check between Chat ({secretify(model_hash)}) "
                f"and ChatRecord ({secretify(self.record_hash)}) was successful"
            )

        return model
