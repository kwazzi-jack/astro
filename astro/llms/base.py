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

from typing import Any, TypeAlias, overload

from langchain_anthropic.chat_models import ChatAnthropic
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from pydantic import (
    BaseModel,
)

from astro.loggings import get_loggy
from astro.typings import (
    ModelName,
    ModelProvider,
    RecordableModel,
    _available_provider_model_options,
    _count_pydantic_fields,
    _model_name_options,
    _model_provider_options,
    _type_name,
)
from astro.utilities.security import get_secret_key

# --- GLOBALS ---
loggy = get_loggy(__file__)
ChatModel: TypeAlias = ChatAnthropic | ChatOllama | ChatOpenAI


def available_model_providers() -> list[str]:
    """Get a list of all supported model providers."""
    return ModelProvider.available()


def available_model_names() -> list[str]:
    """Get a list of all supported model names."""
    return ModelName.available()


def _available_models_message(
    recommendations: dict[str, ModelName] | None = None,
) -> str:
    message = ""
    if recommendations is not None:
        message += "Available providers with recommendations: "
        message += f"{_model_provider_options(recommendations)}\n"
    message += f"Available models: {_model_name_options()}"
    return message


class LLMConfig(RecordableModel, frozen=True):
    model_name: ModelName
    model_provider: ModelProvider
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

    @classmethod
    def _handle_identifier_only(
        cls,
        identifier: str | ModelName | ModelProvider,
        recommendations: dict[str, ModelName] | None,
    ) -> ModelName:
        """Parse the identifier and determine model name directly or via provider through recommendations"""
        loggy.debug(f"Handling identifier-only: {identifier}")

        # Handle direct ModelName
        if isinstance(identifier, ModelName):
            loggy.debug(f"Identifier is a ModelName: {identifier}")
            return identifier

        # Handle string identifiers
        if isinstance(identifier, str):
            loggy.debug(f"Identifier is a string: '{identifier}'")
            # Check if it's a valid model name first
            if ModelName.supports(identifier):
                loggy.debug("Identifier is a supported model name.")
                return ModelName(identifier)

            # Check if it's in recommendations
            if recommendations and identifier in recommendations:
                loggy.debug("Identifier found in recommendations for provider.")
                return recommendations[identifier]

            # Check if it's a valid provider
            if ModelProvider.supports(identifier):
                loggy.debug("Identifier is a supported provider.")
                if recommendations is None:
                    raise loggy.ValueError(
                        "No recommendations for any providers. "
                        f"Please specify a model name directly: {_model_name_options()}",
                        identifier=identifier,
                    )
                elif identifier not in recommendations:
                    error_msg = f"No recommendation for provider {identifier}.\n"
                    error_msg += _available_models_message(recommendations)
                    raise loggy.ValueError(
                        error_msg,
                        identifier=identifier,
                        recommendations=recommendations,
                    )
                return recommendations[identifier]

            # Invalid string
            raise loggy.SupportError(
                feature=identifier,
                reason="Not a valid model or provider",
                identifier=identifier,
            )

        # Handle ModelProvider
        if isinstance(identifier, ModelProvider):
            loggy.debug(f"Identifier is a ModelProvider: {identifier}")
            if recommendations is None:
                raise loggy.ValueError(
                    "No recommendations for any providers. "
                    f"Please specify a model name directly: {_model_name_options()}",
                    identifier=identifier,
                )
            elif identifier not in recommendations:
                error_msg = f"No recommendation for provider {identifier}.\n"
                raise loggy.ValueError(
                    error_msg,
                    identifier=identifier,
                    recommendations=recommendations,
                )
            return recommendations[identifier.value]

        raise loggy.SupportError(
            feature=identifier, reason=f"Unsupported identifier type {type(identifier)}"
        )

    @classmethod
    def _handle_identifier_provider(
        cls, identifier: str | ModelName, provider: str | ModelProvider
    ) -> ModelName:
        """
        Parse the identifier and provider and determine corresponding model name.

        Args:
            identifier (str | ModelName): Model identifier (string model name or ModelName enum)
            provider (str | ModelProvider): Provider identifier (string provider name or ModelProvider enum)

        Returns:
            ModelName: Validated ModelName that matches the specified provider

        Raises:
            ValueError: If validation fails or model/provider don't match
        """
        loggy.debug(f"Handling identifier '{identifier}' with provider '{provider}'")
        # Step 1: Validate and parse the model identifier
        if isinstance(identifier, str):
            loggy.debug("Identifier is a string, validating...")
            # Check if string is actually a provider name (common mistake)
            if ModelProvider.supports(identifier):
                got_provider = ModelProvider(identifier)
                got_models_str = _available_provider_model_options(got_provider)
                raise loggy.ValueError(
                    "Parsing model name with provider failed since "
                    f"identifier is model provider: '{identifier}'. ",
                    warning=(
                        "When using both identifier and provider arguments, "
                        "that identifier expects a string or ModelName. "
                        "Either omit the provider argument for automatic "
                        "selection, or choose identifier as a correct "
                        f"model name for '{got_provider.value}': {got_models_str}"
                    ),
                )

            # Check if the model name is supported
            if not ModelName.supports(identifier):
                raise loggy.ValueError(
                    f"Model '{identifier}' is not supported by Astro. "
                    f"Please specify a model name: {_model_name_options()}"
                )

            model_name = ModelName(identifier)
            loggy.debug(f"Identifier string parsed to ModelName: {model_name}")

        elif isinstance(identifier, ModelName):
            model_name = identifier
            loggy.debug(f"Identifier is already a ModelName: {model_name}")

        else:
            raise loggy.ExpectedVariableType(
                var_name="identifier", got=type(identifier), expected=(str, ModelName)
            )

        # Step 2: Validate and parse the provider
        if isinstance(provider, str):
            loggy.debug("Provider is a string, validating...")
            # Check if string is actually a model name (common mistake)
            if ModelName.supports(provider):
                raise loggy.ExpectedTypeError(
                    var_name="provider",
                    got=type(provider),
                    expected=(str, ModelProvider),
                    warning="Please specify a provider name or omit for automatic inference.",
                )

            # Check if the provider name is supported
            if not ModelProvider.supports(provider):
                raise loggy.ValueError(
                    f"Provider {provider} is not supported by Astro. "
                    f"Supported providers: {ModelProvider.available()}"
                )

            model_provider = ModelProvider(provider)
            loggy.debug(f"Provider string parsed to ModelProvider: {model_provider}")

        elif isinstance(provider, ModelProvider):
            model_provider = provider
            loggy.debug(f"Provider is already a ModelProvider: {model_provider}")

        else:
            raise loggy.ExpectedVariableType(
                var_name="provider",
                got=type(provider),
                expected=(str, ModelProvider),
            )

        # Step 3: Validate that the model belongs to the specified provider
        loggy.debug(
            f"Validating that model {model_name} belongs to provider {model_provider}"
        )
        if model_name.provider != model_provider:
            raise loggy.LLMError(
                f"Model {model_name.value!r} belongs to provider {model_name.provider.value!r}, "
                f"not {model_provider.value!r}. Please use a model from the correct provider.",
                model_identifier=model_name,
                model_provider=model_provider,
                model_name_provider=model_name.provider,
            )
        return model_name

    @classmethod
    def _parse_identifier_provider(
        cls,
        identifier: str | ModelName | ModelProvider,
        provider: str | ModelProvider | None = None,
        recommendations: dict[str, ModelName] | None = None,
    ) -> ModelName:
        """General parsing of identifier and parser to get model name"""
        loggy.debug(
            f"Parsing identifier {identifier!r} and provider {provider!r} with recommendations: {recommendations is not None}"
        )

        if not isinstance(identifier, (str, ModelName, ModelProvider)):
            raise loggy.ExpectedVariableType(
                var_name="identifier",
                got=type(identifier),
                expected=(str, ModelName, ModelProvider),
            )

        if provider is not None and not isinstance(provider, (str, ModelProvider)):
            raise loggy.ExpectedVariableType(
                var_name="provider", got=type(provider), expected=(str, ModelProvider)
            )

        if recommendations is not None and not isinstance(recommendations, dict):
            raise loggy.ExpectedVariableType(
                var_name="recommendations", got=type(recommendations), expected=dict
            )

        if provider is None:
            loggy.debug("Provider is None, handling identifier only.")
            return cls._handle_identifier_only(identifier, recommendations)
        else:
            loggy.debug("Provider is specified, handling identifier and provider.")
            if isinstance(identifier, ModelProvider):
                outer_error = loggy.ExpectedVariableType(
                    var_name="identifier",
                    got=type(identifier),
                    expected=(str, ModelName),
                )
                raise loggy.ValueError(
                    "Cannot specify both identifier to be of type ModelProvider if provider is not None",
                    caused_by=outer_error,
                )
            return cls._handle_identifier_provider(identifier, provider)

    @overload
    @classmethod
    def for_conversational(
        cls,
        identifier: str | ModelName | ModelProvider,
        **overrides: Any,
    ) -> "LLMConfig": ...

    @overload
    @classmethod
    def for_conversational(
        cls,
        identifier: str | ModelName,
        provider: str | ModelProvider,
        **overrides: Any,
    ) -> "LLMConfig": ...

    @overload
    @classmethod
    def for_conversational(
        cls,
        identifier: str | ModelName | ModelProvider,
        provider: None = None,
        **overrides: Any,
    ) -> "LLMConfig": ...

    @classmethod
    def for_conversational(
        cls,
        identifier: str | ModelName | ModelProvider,
        provider: str | ModelProvider | None = None,
        **overrides: Any,
    ) -> "LLMConfig":
        loggy.debug(f"Creating conversational LLMConfig for identifier: {identifier!r}")
        # Recommended conversational models per provider
        recommendations: dict[str, ModelName] = {
            "openai": ModelName.GPT_4O_MINI,
            "anthropic": ModelName.CLAUDE_SONNET_4,
            "ollama": ModelName.LLAMA3_8B,
        }
        debug_str = ", ".join(
            f"{provider_str}/{model.value}"
            for provider_str, model in recommendations.items()
        )
        loggy.debug(
            f"Recommendations for conversational: {debug_str}",
            recommendations=recommendations,
        )

        # Parse and validate the identifier/provider combination
        if provider is None:
            model_name = cls._handle_identifier_only(identifier, recommendations)
        else:
            model_name = cls._handle_identifier_provider(identifier, provider)
        loggy.debug(
            f"Successfully validated model {model_name.value!r} for provider {model_name.provider.value!r}",
            model_name=model_name,
        )

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
        }

        # Merge overrides onto defaults (overrides wins)
        debug_str = ", "
        loggy.debug(
            f"Overriden fields equals {len(overrides)}", overriden_fields=overrides
        )
        merged = {**base_defaults, **overrides}

        # Create LLM config with validated model and provider including defaults + overrides
        loggy.debug("Attempting to create and return LLMConfig", merged_fields=merged)
        return cls(model_name=model_name, model_provider=model_name.provider, **merged)


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
    loggy.debug("Creating new chat model")
    # Validate input
    if not isinstance(llm_config, LLMConfig):
        raise loggy.ExpectedVariableType(
            var_name="llm_config", got=type(llm_config), expected=LLMConfig
        )
    loggy.debug(
        f"Using LLMConfig {llm_config.secret_uid} "
        f"for {llm_config.model_name.value!r} "
        f"from {llm_config.model_provider.value!r} "
        f"(langchain-{llm_config.model_provider.value})"
    )

    # Create langchain chat model based on provider
    match llm_config.model_provider:
        case ModelProvider.OPENAI:
            # TODO Add warnings for settings given that do not effect openai
            reasoning_config = (
                {"effort": "medium", "summary": "auto"}
                if llm_config.reasoning
                else None
            )
            loggy.debug("Selected OpenAI and returning with model")
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
                api_key=get_secret_key("OPENAI_API_KEY"),
            )

        case ModelProvider.ANTHROPIC:
            # TODO Add warnings for settings given that do not effect anthropic
            thinking_config = (
                {"type": "enabled", "budget_tokens": 2048}
                if llm_config.reasoning
                else None
            )
            loggy.debug("Selected Anthropic and returning with model")
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
                api_key=get_secret_key("ANTHROPIC_API_KEY"),
            )  # pyright: ignore[reportCallIssue]

        case ModelProvider.OLLAMA:
            # TODO Add warnings for settings given that do not effect ollama
            loggy.debug("Selected Ollama and returning with model")
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
            raise loggy.SupportError(
                feature=llm_config.model_provider.value, llm_config=llm_config
            )


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
    loggy.info("Creating structured LLM model")

    # Additional Input validation
    if not isinstance(output_schema, type) or not issubclass(output_schema, BaseModel):
        raise loggy.ExpectedVariableType(
            var_name="output_schema",
            got=type(output_schema),
            expected=type[BaseModel],
        )
    if not isinstance(include_raw, bool):
        raise loggy.ExpectedVariableType(
            var_name="include_raw", got=type(include_raw), expected=bool
        )

    # No fields present
    if _count_pydantic_fields(output_schema) == 0:
        loggy.warning(
            f"Output schema {output_schema.__name__} has "
            "no fields to be used in structured output"
        )

    loggy.debug(f"Creating model from LLMConfig {llm_config.secret_uid!r}")
    llm = create_chat_model(llm_config)

    # Return chat model bound to provided output schema
    loggy.debug(
        f"Model created and returning with output schema {type(output_schema).__name__} ({include_raw=})"
    )
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
    loggy.info(f"Creating conversation model with {identifier!r}")
    config = LLMConfig.for_conversational(identifier, provider=provider, **overrides)

    loggy.debug("Config created. Creating model from config")
    return create_chat_model(config), config


if __name__ == "__main__":
    model_1 = LLMConfig.for_conversational("ollama")
    model_2 = LLMConfig.for_conversational("openai")
    model_3 = model_1.model_copy(deep=True)
    model_4 = LLMConfig.for_conversational("ollama")
    model_5 = LLMConfig.for_conversational("openai")

    test_pairs = [
        (None, None),  # No input
        (None, "ollama"),  # Provider only input
        ("gemini", None),  # Gemini only
        ("gemini", "gemini"),  # Double gemini
        ("ollama", "ollama"),  # Double ollama
        ("not_a_model", None),  # Invalid model name
        ("gpt-4o-mini", "anthropic"),  # Mismatched provider
        (ModelName.GPT_4O_MINI, "anthropic"),  # Mismatched provider (enum)
        ("openai", "gpt-4o-mini"),  # Swapped arguments
        (
            ModelProvider.OLLAMA,
            "ollama",
        ),  # Identifier as provider when provider is also specified
        (123, None),  # Invalid identifier type
        ("gpt-4o-mini", 123),  # Invalid provider type
    ]

    for identifier, provider in test_pairs:
        try:
            LLMConfig.for_conversational(identifier=identifier, provider=provider)
        except Exception:
            pass

    print(f"{model_1}, {hash(model_1)=}, {model_1.secret_uid=}")
    print(f"{model_2}, {hash(model_2)=}, {model_2.secret_uid=}")
    print(f"{model_3}, {hash(model_3)=}, {model_3.secret_uid=}")
    print(f"{model_4}, {hash(model_4)=}, {model_4.secret_uid=}")
    print(f"{model_5}, {hash(model_5)=}, {model_5.secret_uid=}")
    print()
    print(f"Model 1 is Model 2? {model_1 == model_2} or {model_1 is model_2}")
    print(f"Model 1 is Model 3? {model_1 == model_3} or {model_1 is model_3}")
    print(f"Model 2 is Model 3? {model_2 == model_3} or {model_2 is model_3}")
    print(f"Model 1 is Model 4? {model_1 == model_4} or {model_1 is model_4}")
    print(f"Model 2 is Model 5? {model_2 == model_5} or {model_2 is model_5}")
    print(f"Model 4 is Model 5? {model_4 == model_5} or {model_4 is model_5}")
