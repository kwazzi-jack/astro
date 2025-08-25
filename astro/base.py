
from pydantic import BaseModel, Field, computed_field

from astro.utilities.timing import get_timestamp
from astro.utilities.uids import create_uid


class TraceableModel(BaseModel):
    """Base class for data models used in agents and modules.

    This class provides a common structure for data models, including metadata fields
    such as name, UID, and creation timestamp. It can be extended to create specific
    data models for inputs, outputs, and states in agents or modules.
    """

    uid: str = Field(
        default_factory=create_uid, description="Unique identifier for this instance."
    )
    created_at: str = Field(
        default_factory=get_timestamp,
        description="Creation timestamp of this instance.",
    )

    @computed_field
    @property
    def name_uid(self) -> str:
        """Returns a unique name identifier for this instance."""
        return f"{self.__class__.__name__}#{self.uid}"

