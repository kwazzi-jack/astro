"""
astro/llms/base.py

Core type definitions and base agent abstractions.

Author(s):
    - Brian Welman
Date: 2025-08-11
License: MIT

Description:
    This module provides core abstractions for creating and configuring chat models from different LLM providers.
    It supports OpenAI, Anthropic, and Ollama models with a unified interface, allowing easy switching between
    providers while maintaining consistent parameter handling. The module includes factory functions for both
    standard chat models and structured output models, with comprehensive parameter support for temperature,
    token limits, retries, and provider-specific settings.

Dependencies:
    - langchain-anthropic
    - langchain-core
    - langchain-ollama
    - langchain-openai
    - pydantic
"""

from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias, overload

from langchain_anthropic.chat_models import ChatAnthropic
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    computed_field,
)

from astro.llms.contexts import ChatContext
from astro.llms.prompts import get_chat_system_prompt, get_chat_welcome_prompt
from astro.typings import TraceableModel
from astro.utilities.security import get_secret_key
from astro.utilities.timing import get_datetime_now
from astro.utilities.uids import named_uid_factory

ChatModel: TypeAlias = ChatAnthropic | ChatOllama | ChatOpenAI


class ModelName(StrEnum):
    # OpenAI
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"

    # Anthropic
    CLAUDE_SONNET_4 = "claude-sonnet-4"

    # Ollama
    GEMMA3_4B = "gemma3:4b"
    LLAMA3_8B = "llama3:8b"
    DEEPSEEK_R1_8B = "deepseek-r1:8b"
    DEEPSEEK_R1_14B = "deepseek-r1:14b"
    MISTRAL_7B = "mistral:7b"
    CODELLAMA_7B = "codellama:7b"
    CODELLAMA_13B = "codellama:13b"
    CODELLAMA_34B = "codellama:34b"
    GPT_OSS_20B = "gpt-oss:20b"

    @property
    def provider(self) -> "ModelProvider":
        match self:
            # OpenAI
            case ModelName.GPT_4O | ModelName.GPT_4O_MINI:
                return ModelProvider.OPENAI
            # Anthropic
            case ModelName.CLAUDE_SONNET_4:
                return ModelProvider.ANTHROPIC
            # Ollama
            case (
                ModelName.GEMMA3_4B
                | ModelName.LLAMA3_8B
                | ModelName.DEEPSEEK_R1_8B
                | ModelName.DEEPSEEK_R1_14B
                | ModelName.MISTRAL_7B
                | ModelName.CODELLAMA_7B
                | ModelName.CODELLAMA_13B
                | ModelName.CODELLAMA_34B
                | ModelName.GPT_OSS_20B
            ):
                return ModelProvider.OLLAMA
            case _:
                # B - Annoying that we have to do this for type checker but oh well
                raise ValueError(f"Unsupported `ModelName`: `{self!r}`")

    @classmethod
    def available(cls, *exclusions: str) -> str:
        return ", ".join(f"`{m}`" for m in cls if m.value not in exclusions)

    @classmethod
    def supports(cls, model_name: str) -> bool:
        normalized_value = (
            model_name.strip().upper().replace(":", "_").replace("-", "_")
        )
        return normalized_value in cls.__members__


class ModelProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

    @property
    def default(self) -> "ModelProvider":
        return ModelProvider.OLLAMA

    @classmethod
    def associated_with(cls, model_name: str) -> "ModelProvider":
        if not ModelName.supports(model_name):
            raise ValueError(f"The model `{model_name}` is not supported by Astro")

        model_name_enum = ModelName(model_name)
        return model_name_enum.provider

    @property
    def models(self) -> list[ModelName]:
        match self:
            # OpenAI
            case ModelProvider.OPENAI:
                return [ModelName.GPT_4O, ModelName.GPT_4O_MINI]

            # Anthropic
            case ModelProvider.ANTHROPIC:
                return [ModelName.CLAUDE_SONNET_4]

            # Ollama
            case ModelProvider.OLLAMA:
                return [
                    ModelName.GEMMA3_4B,
                    ModelName.LLAMA3_8B,
                    ModelName.DEEPSEEK_R1_8B,  # NOTE B - might be issue if we use langchain-deepseek
                    ModelName.DEEPSEEK_R1_14B,  # NOTE B - might be issue if we use langchain-deepseek
                    ModelName.MISTRAL_7B,
                    ModelName.CODELLAMA_7B,
                    ModelName.CODELLAMA_13B,
                    ModelName.CODELLAMA_34B,
                    ModelName.GPT_OSS_20B,  # NOTE B - OpenAI does not support this via API
                ]

            # Fail safe
            case default:
                raise ValueError(
                    f"Probably internal mistake that {default} is not assigned yet"
                )

    @classmethod
    def available(cls, *exclusions: str) -> str:
        return ", ".join(f"`{m}`" for m in cls if m.value not in exclusions)

    @classmethod
    def supports(cls, model_name: str) -> bool:
        normalized_value = model_name.strip().upper()
        return normalized_value in cls.__members__


class LLMConfig(TraceableModel):
    model_name: str
    provider: ModelProvider
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int | None = None
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    streaming: bool = False
    reasoning: bool = False
    thinking: bool = False
    max_retries: int = 3
    timeout: float = 60.0
    seed: int | None = None
    count: int | None = None
    context_size: int = 2048
    gpu_count: int | None = None
    thread_count: int | None = None
    keep_alive: bool = False
    api_key: str | None = None

    @classmethod
    def _handle_identifier_only(
        cls,
        identifier: str | ModelName | ModelProvider,
        recommendations: dict[str, ModelName] | None,
    ) -> ModelName:
        """Parse the identifier and determine model name directly or via provider through recommendations"""

        # Handle direct ModelName
        if isinstance(identifier, ModelName):
            return identifier

        # Handle string identifiers
        if isinstance(identifier, str):
            # Check if it's a valid model name first
            if ModelName.supports(identifier):
                return ModelName(identifier)

            # Check if it's in recommendations
            if recommendations and identifier in recommendations:
                return recommendations[identifier]

            # Check if it's a valid provider
            if ModelProvider.supports(identifier):
                if recommendations is None:
                    raise ValueError(
                        f"No recommendations for any providers. "
                        f"Please specify a model name directly: {ModelName.available()}"
                    )
                elif identifier not in recommendations:
                    error_msg = f"No recommendation for provider `{identifier}`.\n"
                    error_msg += f"Available providers with recommendations: {ModelProvider.available(*recommendations.keys())}\n"
                    error_msg += f"Available models: {ModelName.available()}"
                    raise ValueError(error_msg)

                return recommendations[identifier]

            # Invalid string
            raise ValueError(f"'{identifier}' is not a valid model or provider")

        # Handle ModelProvider
        if isinstance(identifier, ModelProvider):
            if recommendations is None:
                raise ValueError(
                    f"No recommendations for any providers. "
                    f"Please specify a model name directly: {ModelName.available()}"
                )
            elif identifier not in recommendations:
                error_msg = f"No recommendation for provider `{identifier}`.\n"
                error_msg += f"Available providers with recommendations: {ModelProvider.available(*recommendations.keys())}\n"
                error_msg += f"Available models: {ModelName.available()}"
                raise ValueError(error_msg)
            return recommendations[identifier.value]

        raise ValueError(f"Unsupported identifier type: {type(identifier)}")

    @classmethod
    def _handle_identifier_provider(
        cls, identifier: str | ModelName, provider: str | ModelProvider
    ) -> ModelName:
        """
        Parse the identifier and provider and determine corresponding model name.

        Args:
            identifier: Model identifier (string model name or ModelName enum)
            provider: Provider identifier (string provider name or ModelProvider enum)

        Returns:
            Validated ModelName that matches the specified provider

        Raises:
            ValueError: If validation fails or model/provider don't match
        """
        # Step 1: Validate and parse the model identifier
        if isinstance(identifier, str):
            # Check if string is actually a provider name (common mistake)
            if ModelProvider.supports(identifier):
                raise ValueError(
                    f"Expected a model name but received provider name: `{identifier}`. "
                    f"Please specify a model name: {ModelName.available()}"
                )

            # Check if the model name is supported
            if not ModelName.supports(identifier):
                raise ValueError(
                    f"Model `{identifier}` is not supported by Astro. "
                    f"Please specify a model name: {ModelName.available()}"
                )

            model_name = ModelName(identifier)

        elif isinstance(identifier, ModelName):
            model_name = identifier

        else:
            raise ValueError(
                f"Invalid identifier type: `{type(identifier).__name__}`. "
                f"Expected `str` or `ModelName`"
            )

        # Step 2: Validate and parse the provider
        if isinstance(provider, str):
            # Check if string is actually a model name (common mistake)
            if ModelName.supports(provider):
                raise ValueError(
                    f"Expected a provider name but received model name: `{provider}`. "
                    f"Please specify a provider name or omit for automatic inference.\n"
                    f"Supported providers: {ModelProvider.available()}"
                )

            # Check if the provider name is supported
            if not ModelProvider.supports(provider):
                raise ValueError(
                    f"Provider `{provider}` is not supported by Astro. "
                    f"Supported providers: {ModelProvider.available()}"
                )

            model_provider = ModelProvider(provider)

        elif isinstance(provider, ModelProvider):
            model_provider = provider

        else:
            raise ValueError(
                f"Invalid provider type: `{type(provider).__name__}`. "
                f"Expected `str` or `ModelProvider`"
            )

        # Step 3: Validate that the model belongs to the specified provider
        if model_name.provider != model_provider:
            error_msg = (
                f"Model `{model_name}` belongs to provider `{model_name.provider}`, "
                f"not `{model_provider}`. Please use a model from the correct provider:\n"
            )
            for provider in ModelProvider:
                error_msg += (
                    f" - `{provider}`: {', '.join(f'`{m}`' for m in provider.models)}\n"
                )

            raise ValueError(error_msg)

        return model_name

    @classmethod
    def _parse_identifier_provider(
        cls,
        identifier: str | ModelName | ModelProvider,
        provider: str | ModelProvider | None = None,
        recommendations: dict[str, ModelName] | None = None,
    ) -> ModelName:
        """General parsing of identifier and parser to get model name"""

        if not isinstance(identifier, (str, ModelName, ModelProvider)):
            raise ValueError(
                f"Invalid identifier type: `{type(identifier).__name__}`. "
                "Expected `str`, `ModelName` or `ModelProvider`"
            )

        if provider is not None and not isinstance(provider, (str, ModelProvider)):
            raise ValueError(
                f"Invalid provider type: `{type(provider).__name__}`. "
                "Expected `str` or `ModelProvider`"
            )

        if recommendations is not None and not isinstance(recommendations, dict):
            raise ValueError(
                "recommendations must be a dict mapping provider names to ModelName instances"
            )

        if provider is None:
            return cls._handle_identifier_only(identifier, recommendations)
        else:
            if isinstance(identifier, ModelProvider):
                raise ValueError(
                    "Cannot specify both ModelProvider as identifier and separate provider parameter"
                )
            return cls._handle_identifier_provider(identifier, provider)

    @overload
    @classmethod
    def for_conversational(
        cls, identifier: str | ModelName | ModelProvider
    ) -> "LLMConfig": ...

    @overload
    @classmethod
    def for_conversational(
        cls, identifier: str | ModelName, provider: str | ModelProvider
    ) -> "LLMConfig": ...

    @overload
    @classmethod
    def for_conversational(
        cls, identifier: str | ModelName | ModelProvider, provider: None = None
    ) -> "LLMConfig": ...

    @classmethod
    def for_conversational(
        cls,
        identifier: str | ModelName | ModelProvider,
        provider: str | ModelProvider | None = None,
        **overrides,
    ) -> "LLMConfig":
        # Recommended conversational models per provider
        recommendations: dict[str, ModelName] = {
            "openai": ModelName.GPT_4O_MINI,
            "anthropic": ModelName.CLAUDE_SONNET_4,
            "ollama": ModelName.LLAMA3_8B,
        }

        # Parse and validate the identifier/provider combination
        if provider is None:
            model_name = cls._handle_identifier_only(identifier, recommendations)
        else:
            if isinstance(identifier, ModelProvider):
                raise ValueError(
                    "Cannot specify both ModelProvider as identifier and separate provider parameter. "
                    "Either pass ModelProvider as identifier only, or pass model name with provider."
                )
            model_name = cls._handle_identifier_provider(identifier, provider)

        # Base defaults tuned for conversational usage (can be overridden)
        base_defaults: dict[str, object] = {
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 1024,
            "streaming": True,
            "reasoning": False,
            "thinking": False,
            "max_retries": 3,
            "timeout": 60.0,
            "seed": None,
            "count": 1,
            "context_size": 8192,
            "gpu_count": None,
            "thread_count": None,
            "keep_alive": False,
            "api_key": None,
        }

        # Merge overrides onto defaults (overrides wins)
        merged = {**base_defaults, **overrides}

        # Create LLM config with validated model and provider including defaults + overrides
        return cls(model_name=model_name.value, provider=model_name.provider, **merged)


def create_chat_model(llm_config: LLMConfig) -> ChatModel:
    """Create a chat model instance based on the specified provider and configuration.

    This factory function creates and configures chat model instances from different
    providers (OpenAI, Anthropic, or Ollama) with unified parameter handling.

    Args:
        llm_config (LLMConfig): LLM configuration model

    Returns:
        ChatModel: Configured chat model instance for the specified provider

    Raises:
        ValueError: If the provider extracted from name is not supported

    Note:
        Not all parameters are supported by all providers. Provider-specific
        parameters are ignored for other providers (with TODO warnings planned).

        **Parameter Support by Provider:**

        **Anthropic:** name, temperature, top_p, top_k, max_tokens, streaming, max_retries, timeout

        **Ollama:** name, temperature, top_p, top_k, max_tokens, seed, context_size, gpu_count, thread_count, keep_alive

        **OpenAI:** name, temperature, top_p, max_tokens, frequency_penalty, presence_penalty, streaming, max_retries, timeout, seed, count

        Provider-specific parameters are ignored for other providers.

    Examples:
        >>> # OpenAI model with custom temperature
        >>> model = create_chat_model("openai::gpt-4o-mini", temperature=0.3)

        >>> # Anthropic model with max tokens
        >>> model = create_chat_model("anthropic::claude-sonnet-4", max_tokens=1000)

        >>> # Ollama model with custom context size and GPU settings
        >>> model = create_chat_model("ollama::llama3:latest", context_size=4096, gpu_count=1)

        >>> # Streaming enabled for any provider
        >>> model = create_chat_model("openai::gpt-4o-mini", streaming=True)
    """
    # Validate input
    if not isinstance(llm_config, LLMConfig):
        raise ValueError(
            f"Expected config model of type `LLMConfig`. Got `{llm_config.__class__.__name__}`"
        )

    # Create langchain chat model based on provider
    match llm_config.provider:
        case ModelProvider.OPENAI:
            # TODO Add warnings for settings given that do not effect openai
            reasoning_config = (
                {"effort": "medium", "summary": "auto"}
                if llm_config.reasoning
                else None
            )
            api_key = (
                get_secret_key("OPENAI_API_KEY")
                if llm_config.api_key is None
                else get_secret_key(llm_config.api_key)
            )
            return ChatOpenAI(
                model=llm_config.model_name,
                temperature=llm_config.temperature,
                top_p=llm_config.top_p,
                max_completion_tokens=llm_config.max_tokens,
                frequency_penalty=llm_config.frequency_penalty,
                presence_penalty=llm_config.presence_penalty,
                streaming=llm_config.streaming,
                reasoning=reasoning_config,
                max_retries=llm_config.max_retries,
                timeout=llm_config.timeout,
                seed=llm_config.seed,
                n=llm_config.count,
                api_key=api_key,
            )

        case ModelProvider.ANTHROPIC:
            # TODO Add warnings for settings given that do not effect anthropic
            thinking_config = (
                {"type": "enabled", "budget_tokens": 2000}
                if llm_config.reasoning
                else None
            )
            api_key = (
                get_secret_key("ANTHROPIC_API_KEY")
                if llm_config.api_key is None
                else get_secret_key(llm_config.api_key)
            )
            return ChatAnthropic(
                model=llm_config.model_name,  # pyright: ignore[reportCallIssue]
                temperature=llm_config.temperature,
                top_p=llm_config.top_p,
                top_k=llm_config.top_k,
                max_tokens=llm_config.max_tokens or 2048,  # pyright: ignore[reportCallIssue]
                streaming=llm_config.streaming,
                thinking=thinking_config,
                max_retries=llm_config.max_retries,
                timeout=llm_config.timeout,
                api_key=api_key,
            )  # pyright: ignore[reportCallIssue]

        case ModelProvider.OLLAMA:
            # TODO Add warnings for settings given that do not effect ollama
            return ChatOllama(
                model=llm_config.model_name,
                temperature=llm_config.temperature,
                top_p=llm_config.top_p,
                top_k=llm_config.top_k,
                num_predict=llm_config.max_tokens,
                reasoning=llm_config.reasoning,
                seed=llm_config.seed,
                num_ctx=llm_config.context_size,
                num_gpu=llm_config.gpu_count,
                num_thread=llm_config.thread_count,
                keep_alive=llm_config.keep_alive,
            )
        case _:
            raise ValueError(f"Unsupported chat model provider `{llm_config.provider}`")


def create_structured_model(
    llm_config: LLMConfig,
    output_schema: type[BaseModel],
    include_raw: bool = False,
) -> Runnable:
    """Create a structured language model that outputs data conforming to a specified schema.

    This function creates a chat model with the given parameters and configures it to
    return structured output that matches the provided Pydantic schema.

    Args:
        name (str): Model name in format "provider::model"
        output_schema (Type[BaseModel]): Pydantic schema class for structured output
        include_raw (bool): Whether to include raw model output alongside structured output. Defaults to False.

        # ... Other Arguments ...

        **This function uses `astro.llms.create_chat_model()` internally. See
        it for reference for other argument details.

    Returns:
        Runnable: A configured language model that outputs structured data

    Raises:
        ValueError: If invalid parameters are provided
        RuntimeError: If model creation or configuration fails

    Note:
        For detailed information about additional support by provider, see the
        documentation for `astro.llms.create_chat_model()`.

        Not all parameters are supported by all providers. Provider-specific
        parameters are ignored for other providers.

    Examples:
        >>> from pydantic import BaseModel
        >>>
        >>> class PersonInfo(BaseModel):
        ...     name: str
        ...     age: int
        ...     occupation: str
        >>>
        >>> # Create structured model for person extraction
        >>> model = create_structured_model(
        ...     name="openai::gpt-4o-mini",
        ...     output_schema=PersonInfo,
        ...     temperature=0.3
        ... )
        >>>
        >>> # Use with include_raw for debugging
        >>> debug_model = create_structured_model(
        ...     name="anthropic::claude-sonnet-4",
        ...     output_schema=PersonInfo,
        ...     include_raw=True,
        ...     max_tokens=500
        ... )
        >>>
        >>> # Ollama model with custom settings
        >>> local_model = create_structured_model(
        ...     name="ollama::llama3:latest",
        ...     output_schema=PersonInfo,
        ...     context_size=4096,
        ...     gpu_count=1
        ... )
    """
    llm = create_chat_model(llm_config)

    # Return chat model bound to provided output schema
    return llm.with_structured_output(schema=output_schema, include_raw=include_raw)


def create_conversational_model(
    identifier: str | ModelName | ModelProvider,
    provider: str | ModelProvider | None = None,
    **overrides: Any,
) -> tuple[ChatModel, LLMConfig]:
    """Create a conversational chat model with optimized defaults.

    Args:
        identifier: Model name to use or provider name to use recommended model

    Returns:
        ChatModel: Configured chat model instance optimized for conversation
    """
    config = LLMConfig.for_conversational(identifier, provider=provider, **overrides)
    return create_chat_model(config), config


if __name__ == "__main__":
    ...
