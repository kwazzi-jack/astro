"""
astro/llms/base.py

# TODO

Author(s):
    - Brian Welman
Date: 2025-10-08
License: MIT

Description:
    # TODO

Dependencies:
    - # TODO
"""

# --- Internal Imports ---
from typing import Annotated, ClassVar, Literal, TypeAlias

# --- External Imports ---
import ollama
from pydantic import BaseModel, StringConstraints
from pydantic_ai.models import KnownModelName as _KnownModelName
from pydantic_ai.models import Model
from pydantic_ai.models import infer_model as _infer_model
from pydantic_ai.providers.ollama import OllamaProvider

# --- Local Imports ---
from astro.loggings.base import get_loggy
from astro.typings import literal_to_list, options_to_str

# --- GLOBALS ---
loggy = get_loggy(__file__)


def _is_ollama_identifier(identifier: str) -> bool:
    return identifier.startswith("ollama:")


def _available_local_models() -> list[str]:
    return [f"ollama:{entry.model}" for entry in ollama.list().models]


StrName: TypeAlias = Annotated[
    str, StringConstraints(strip_whitespace=True, to_lower=True, min_length=1)
]


class ModelDetails(BaseModel):
    name: str
    provider: str
    source: Literal["pydantic_ai", "ollama"]
    variant: str | None = None
    internal: str

    @classmethod
    def from_identifier(cls, identifier: str) -> "ModelDetails":
        # Empty identifier
        if len(identifier) == 0:
            raise loggy.CreationError(
                object_type=ModelDetails, reason="Empty identifier"
            )

        # Ollama model
        if _is_ollama_identifier(identifier):
            provider, name, variant = identifier.split(":")
            return ModelDetails(
                name=name,
                provider=provider,
                source="ollama",
                variant=variant,
                internal=f"{name}:{variant}",
            )

        # Pydantic AI model
        else:
            provider, name = identifier.split(":", maxsplit=1)
            return ModelDetails(
                name=name, provider=provider, source="pydantic_ai", internal=identifier
            )

    def to_identifier(self) -> str:
        if self.variant is None:
            return f"{self.provider}:{self.name}"
        else:
            return f"{self.provider}:{self.name}:{self.variant}"

    def is_pydantic_identifier(self) -> bool:
        return self.variant is None

    def is_local_identifier(self) -> bool:
        return self.variant is not None


class KnownModels:
    _initialized: ClassVar[bool] = False
    _name_to_provider: ClassVar[dict[str, set[str]]] = {}
    _provider_to_name: ClassVar[dict[str, set[str]]] = {}
    _name_to_variant: ClassVar[dict[str, set[str]]] = {}
    _identifier_to_details: ClassVar[dict[str, ModelDetails]] = {}

    @classmethod
    def _init_class(cls, all_models: list[str]) -> None:
        # Only one initialization per import
        if cls._initialized:
            return

        # Get model details list
        models = cls._parse_model_details(all_models)

        # No models registered -> Panic
        if len(models) == 0:
            raise loggy.RuntimeError(
                "No models registered on import", all_models=all_models
            )

        # Create mappings
        cls._name_to_provider = cls._create_name_to_provider_map(models)
        cls._provider_to_name = cls._create_provider_to_name_map(models)
        cls._name_to_variant = cls._create_name_to_variant_map(models)
        cls._identifier_to_details = cls._create_identifier_to_details_map(models)

        # Set as initialized
        cls._initialized = True

    @classmethod
    def _parse_model_details(cls, all_models: list[str]) -> list[ModelDetails]:
        result = []
        for identifier in all_models:
            # TODO: Implement huggingface models
            # Turned off to avoid conflicting identifier names
            if "huggingface" in identifier:
                continue

            try:
                result.append(ModelDetails.from_identifier(identifier))
            except Exception as error:
                loggy.exception(error)
                loggy.warning(f"Failed to register model: {identifier!r}")
        return result

    @classmethod
    def _create_name_to_provider_map(
        cls, model_list: list[ModelDetails]
    ) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for model in model_list:
            if model.name in result:
                result[model.name].add(model.provider)
            else:
                result[model.name] = {model.provider}
        return result

    @classmethod
    def _create_provider_to_name_map(
        cls, model_list: list[ModelDetails]
    ) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for model in model_list:
            if model.provider in result:
                result[model.provider].add(model.name)
            else:
                result[model.provider] = {model.name}
        return result

    @classmethod
    def _create_name_to_variant_map(
        cls, model_list: list[ModelDetails]
    ) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for model in model_list:
            if model.variant is None:
                continue
            elif model.name in result:
                result[model.name].add(model.variant)
            else:
                result[model.name] = {model.variant}
        return result

    @classmethod
    def _create_identifier_to_details_map(
        cls, model_list: list[ModelDetails]
    ) -> dict[str, ModelDetails]:
        return {model.to_identifier(): model for model in model_list}

    @classmethod
    def is_valid_provider(cls, value: str) -> bool:
        return value in cls._provider_to_name

    @classmethod
    def is_valid_identifier(cls, value: str) -> bool:
        return value in cls._identifier_to_details

    @classmethod
    def parse(cls, identifier: str) -> ModelDetails:
        if not cls.is_valid_identifier(identifier):
            raise loggy.ValueError(
                f"Invalid model identifier {identifier!r}. "
                "See KnownModels.identifiers() for available valid identifiers",
                identifier=identifier,
            )
        return cls._identifier_to_details[identifier]

    @classmethod
    def identifiers(cls) -> list[str]:
        return sorted(list(cls._identifier_to_details.keys()))

    @classmethod
    def providers(cls) -> list[str]:
        return sorted(list(cls._provider_to_name.keys()))


# --- MORE GLOBALS ---

_PYDANTIC_AI_MODELS: list[str] = literal_to_list(_KnownModelName.__value__)[
    :-2
]  # Skips "test"  # NOTE: Based on Pydanti-AI
_LOCAL_MODELS: list[str] = _available_local_models()  # NOTE: Based on Ollama for now
_MODELS = _PYDANTIC_AI_MODELS + _LOCAL_MODELS  # NOTE: Combined list

# Initialize KnownModels (once)
KnownModels._init_class(_MODELS)

# Restricted Providers
_WHITE_LIST_PROVIDERS = {
    "openai",
    "anthropic",
    "ollama",
    "google-vertex",
    "google-gla",
    "deepseek",
}


# NOTE: Based on and uses `pydantic_ai.models.infer_model`
def infer_model(model_details: ModelDetails) -> Model:
    # NOTE: Restricting for now
    if model_details.provider not in _WHITE_LIST_PROVIDERS:
        error_msg = options_to_str(list(_WHITE_LIST_PROVIDERS), with_repr=True)
        raise loggy.NotImplementedError(
            f"Astro does not support {model_details.provider!r}. Try: {error_msg}"
        )
    # Ollama model
    if model_details.source == "ollama":  # Only Ollama for now
        from pydantic_ai.models.openai import OpenAIChatModel

        return OpenAIChatModel(
            model_name=model_details.internal,
            provider=OllamaProvider(base_url="http://localhost:11434/v1"),
        )

    # Any other model
    return _infer_model(model_details.internal)


def create_llm_model(identifier: str) -> Model:
    # Input validation
    if not isinstance(identifier, str):
        raise loggy.ExpectedVariableType(
            var_name="identifier",
            expected=str,
            got=type(identifier),
            with_value=identifier,
        )

    # Parse identifier
    model_details = KnownModels.parse(identifier)

    # Return the inferred model
    return infer_model(model_details)


if __name__ == "__main__":
    from dotenv import load_dotenv
    from rich.console import Console

    load_dotenv()

    console = Console()
    identifiers = KnownModels.identifiers()
    count = 0
    total = len(identifiers)

    for i, identifier in enumerate(identifiers, start=1):
        try:
            model = create_llm_model(identifier)
            details = KnownModels.parse(identifier)
            count += 1
            style = "green"
        except Exception:
            details = None
            model = None
            style = "red"

        model_str = (
            f"{type(model).__name__} (name={details.name!r}, provider={details.provider!r}, variant={details.variant!r}, source={details.source!r})"
            if model and details
            else "Nope"
        )
        console.print(f"({i:02}/{total}) {identifier}: {model_str}", style=style)

    console.print(f"Total correct: {count}/{total} ({count / total:.2g}%)")
