from datetime import datetime

from sqlmodel import Field, SQLModel

from astro.errors import RecordableIdentityError
from astro.llms.base import LLMConfig
from astro.loggings import get_loggy
from astro.typings import ImmutableRecord, ModelName, ModelProvider, RecordConverter
from astro.utilities.timing import get_datetime_now

# Global variables
_logger = get_loggy(__file__)


# Brian - Not recommended to store API-key -> Will be loaded when llm-model is created
class LLMConfigRecord(ImmutableRecord, RecordConverter, table=True):
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
        _logger.debug(f"Creating `LLMConfigRecord` from `LLMConfig` ({model.to_hex()})")
        return cls(record_hash=hash(model), **model.model_dump())

    def to_model(self) -> LLMConfig:
        """Create model from record"""
        _logger.debug(
            f"Creating `LLMConfig` from `LLMConfigRecord` (record={self.record_hash})"
        )

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

        # Hashes do not match -> not the same model that was saved
        model_hash = hash(model)
        if model_hash != self.record_hash:
            error = RecordableIdentityError(
                model=LLMConfig,
                model_hash=model_hash,
                record=LLMConfigRecord,
                record_hash=self.record_hash,
            )
            _logger.error(**error.to_log())
            raise error
        else:
            _logger.debug(
                f"Hash check between `LLMConfig` ({model.to_hex()}) "
                "and `LLMConfigRecord` was successful"
            )
        # Return knowing it is the same model
        return model
