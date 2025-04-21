import uuid

SAFE_UUID4_LENGTH = 16


def create_uid(length: int = SAFE_UUID4_LENGTH) -> str:
    """Generates a unique string identifier
    using UUID4.

    Args:
        length (int): Length of UID.
    """
    if length > 32:
        raise ValueError(f"UID character length limit is 32. Got {length} > 32")

    return str(uuid.uuid4())[:length].upper()


def create_named_uid(name: str) -> str:
    return f"{name.lower()}-{create_uid(SAFE_UUID4_LENGTH)}"


def create_conv_uid() -> str:
    return create_named_uid("conv")


def create_model_uid(name: str = "model") -> str:
    return create_named_uid(name)


def create_agent_uid(name: str = "agent") -> str:
    return create_named_uid(name)


if __name__ == "__main__":
    print("Testing create_uid with default length:")
    print(f"{create_uid()=}")

    print("\nTesting create_uid with custom length 4:")
    print(f"{create_uid(4)=}")

    print("\nTesting create_uid with length 32:")
    print(f"{create_uid(32)=}")

    print("\nTesting create_uid with invalid length (>32):")
    try:
        print(create_uid(40))
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")

    print("\nTesting create_uid with negative length:")
    print(f"{create_uid(-1)=}  # Still works, slices 0 to -1")

    print("\nTesting create_conv_uid:")
    conv_uid = create_conv_uid()
    print(f"{conv_uid=}, starts with 'conv': {conv_uid.startswith('conv')}")

    print("\nTesting create_model_uid:")
    model_uid = create_model_uid("gpt")
    print(f"{model_uid=}, starts with 'gpt-': {model_uid.startswith('gpt-')}")

    print("\nTesting create_agent_uid with default name:")
    agent_uid_default = create_agent_uid()
    print(
        f"{agent_uid_default=}, starts with 'agent-': {agent_uid_default.startswith('agent-')}"
    )

    print("\nTesting create_agent_uid with custom name:")
    agent_uid_custom = create_agent_uid("assistant")
    print(
        f"{agent_uid_custom=}, starts with 'assistant-': {agent_uid_custom.startswith('assistant-')}"
    )
