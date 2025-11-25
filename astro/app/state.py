# --- Internal Imports ---
import json
from typing import Any, Literal, overload

# --- External Imports ---
from pydantic import BaseModel, Field, field_validator

# --- Local Imports ---
from astro.llms.base import KnownModels, ModelDetails
from astro.logger import get_loggy
from astro.paths import APPSTATE_PATH
from astro.utilities.system import get_users_name

# --- Globals ---
_loggy = get_loggy(__file__)


# --- Helper Function ---
def _check_if_initialised() -> Literal["valid", "corrupted", "missing"]:
    return "valid"


# --- Application State ---
class _AppState(BaseModel):
    username: str = Field(default_factory=get_users_name)
    current_model: ModelDetails = Field(
        default=KnownModels.parse("ollama:llama3.1:latest")
    )
    prompt_enabled: bool = Field(default=False)
    initialised: bool = Field(default=False)

    @field_validator("initialised")
    @classmethod
    def validate_initialised(cls, value: Any) -> bool:
        return value

    def save(self):
        if APPSTATE_PATH is None:
            raise _loggy.ValueError("Appstate path has not been set")

        json_dict = self.model_dump_json()
        with open(APPSTATE_PATH, "w") as file:
            json.dump(json_dict, file)

    @classmethod
    def touch(cls, overwrite: bool = False) -> "_AppState":
        if APPSTATE_PATH is None:
            raise _loggy.ValueError("Appstate path has not been set")
        if APPSTATE_PATH.exists() and not overwrite:
            raise _loggy.FileExistsError(
                "Appstate file exists. Set overwrite to True to overwrite"
            )
        if APPSTATE_PATH.exists():
            APPSTATE_PATH.unlink()
        state = cls()
        state.save()
        return state

    @classmethod
    def load(cls) -> "_AppState":
        if APPSTATE_PATH is None:
            raise _loggy.ValueError("Appstate path has not been set")
        if not APPSTATE_PATH.exists():
            raise _loggy.FileNotFoundError("Appstate file does not exist")

        with open(APPSTATE_PATH) as file:
            json_dict = json.load(file)

        return cls.model_validate_json(json_dict)

    @overload
    def switch_model_to(self, identifier: str) -> None:
        pass

    @overload
    def switch_model_to(self, identifier: ModelDetails) -> None:
        pass

    def switch_model_to(self, identifier: str | ModelDetails) -> None:
        if isinstance(identifier, str):
            self.current_model = KnownModels.parse(identifier)
        else:
            self.current_model = identifier

    def switch_username_to(self, value: str) -> None:
        if len(value) == 0:
            raise _loggy.ValueError("Expected non-empty name")
        self.username = value
