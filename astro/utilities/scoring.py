import numpy as np
from numpy.typing import NDArray

from astro.configs.base import Recognition, Speed, Memory
from astro.utilities.display import rprint
from astro.utilities.functions import literal_to_tuple

# TODO: Add better scoring function.
# Currently tries to find closest match, but it does not account for ordinal categories.
# Also, moving all selection processes to here would be best, e.g. pricing, etc.

def get_norm_ordinals(size: int) -> NDArray[np.float32]:
    return np.linspace(0.0, 1.0, size)


_recog_values = literal_to_tuple(Recognition)
_speed_values = literal_to_tuple(Speed)
_mem_values = literal_to_tuple(Memory)

SCORABLE_PROPERTIES: dict[str, tuple[tuple[str, ...], NDArray[np.float32]]] = {
    "recognition": (_recog_values, get_norm_ordinals(len(_recog_values))),
    "speed": (_speed_values, get_norm_ordinals(len(_speed_values))),
    "memory": (_mem_values, get_norm_ordinals(len(_mem_values))),
}


def is_scorable_props(props) -> bool:
    """
    Check if the keys in the props dictionary match exactly the keys in SCORABLE_PROPERTIES.

    Args:
        props: Dictionary containing properties to check

    Returns:
        True if the sets of keys are identical, False otherwise
    """

    for key, (values, _) in SCORABLE_PROPERTIES.items():
        if key not in props or props[key] not in values:
            return False

    return True


def calculate_matching_score(
    flex_props: dict[str, str], fixed_props: dict[str, str]
) -> float:
    if not is_scorable_props(fixed_props):
        raise ValueError(
            "Fixed property dictionary does not match scoring property requirements"
        )
    score = 0.0
    count = 0
    for key, value1 in flex_props.items():

        if key not in SCORABLE_PROPERTIES:
            continue

        values, norm_ordinals = SCORABLE_PROPERTIES[key]

        if value1 not in values:
            fmt_values = ", ".join(f"`{value}`" for value in values)
            raise ValueError(
                f"Unknown value for property `{key.capitalize()}` in first properties. Expected {fmt_values}. Got `{value1}`"
            )

        value2 = fixed_props.get(key)
        idx1 = values.index(value1)
        idx2 = values.index(value2)
        score += np.abs(norm_ordinals[idx1] - norm_ordinals[idx2])
        count += 1

    return 1 - float(score / count)


if __name__ == "__main__":
    recog_ord_arr = get_norm_ordinals(len(literal_to_tuple(Recognition)))
    rprint(f"{Recognition=}\n{recog_ord_arr=}\n")

    speed_ord_arr = get_norm_ordinals(len(literal_to_tuple(Speed)))
    rprint(f"{Speed=}\n{speed_ord_arr=}\n")

    mem_ord_arr = get_norm_ordinals(len(literal_to_tuple(Memory)))
    rprint(f"{Memory=}\n{mem_ord_arr=}\n")

    gpt4o_props = {
        "speed": "medium",
        "recognition": "very high",
        "memory": "large",
    }

    gpt4o_mini_props = {
        "speed": "fast",
        "recognition": "high",
        "memory": "large",
    }

    test_1_props = {
        "speed": "fast",
        "recognition": "high",
        "memory": "large",
    }

    test_2_props = {
        "speed": "fast",
        "recognition": "high",
        "memory": "large",
    }

    test_3_props = {
        "speed": "very fast",
        "recognition": "very low",
        "memory": "small",
    }

    test_4_props = {
        "speed": "very fast",
    }

    test_5_props = {
        "speed": "very fast",
        "recognition": "very high",
    }

    rprint(f"{calculate_matching_score(gpt4o_mini_props, gpt4o_mini_props)=}")
    rprint(f"{calculate_matching_score(gpt4o_mini_props, gpt4o_props)=}")
    rprint(f"{calculate_matching_score(gpt4o_props, gpt4o_mini_props)=}")
    rprint(f"{calculate_matching_score(gpt4o_props, gpt4o_props)=}")
    rprint(f"{calculate_matching_score(test_1_props, gpt4o_mini_props)=}")
    rprint(f"{calculate_matching_score(test_1_props, gpt4o_props)=}")
    rprint(f"{calculate_matching_score(test_2_props, gpt4o_mini_props)=}")
    rprint(f"{calculate_matching_score(test_2_props, gpt4o_props)=}")
    rprint(f"{calculate_matching_score(test_3_props, gpt4o_mini_props)=}")
    rprint(f"{calculate_matching_score(test_3_props, gpt4o_props)=}")
    rprint(f"{calculate_matching_score(test_4_props, gpt4o_mini_props)=}")
    rprint(f"{calculate_matching_score(test_4_props, gpt4o_props)=}")
    rprint(f"{calculate_matching_score(test_5_props, gpt4o_mini_props)=}")
    rprint(f"{calculate_matching_score(test_5_props, gpt4o_props)=}")
