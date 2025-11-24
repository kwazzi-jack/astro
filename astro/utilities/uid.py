# --- Internal Imports ---
from uuid import uuid4

# --- Globals ---
_CHOSEN_UID_LENGTH = 8
_CHOSEN_DOUBLE_UID_LENGTH = 5


# --- UID Functions ---
def generate_uid(name: str | None = None) -> str:
    if name is None:
        return "#" + uuid4().hex[:_CHOSEN_UID_LENGTH]
    return f"{name}#{uuid4().hex[:_CHOSEN_UID_LENGTH]}"


def generate_double_uid(first_name: str, second_name: str) -> str:
    return f"{first_name}::{second_name}#{uuid4().hex[:_CHOSEN_DOUBLE_UID_LENGTH]}"


if __name__ == "__main__":
    uid = generate_uid("agent")

    for i in range(10):
        print(f"{generate_double_uid('astro-chat', 'tool_call')}")
