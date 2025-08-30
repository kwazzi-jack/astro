from collections.abc import Sequence
from typing import Any

from astro.typings import _options_to_str


def _expected_got_value_error(got: type, expected: Sequence[type] | type) -> ValueError:
    if not isinstance(expected, Sequence):
        expected = [expected]
    expected_str = _options_to_str(
        [expected_type.__name__ for expected_type in expected]
    )
    return ValueError(f"Expected type {expected_str}. Got `{got.__name__}` instead")


def _expected_got_var_value_error(
    var_name: str, got: type, expected: Sequence[type] | type
) -> ValueError:
    if not isinstance(expected, Sequence):
        expected = [expected]

    expected_str = _options_to_str(
        [expected_type.__name__ for expected_type in expected]
    )
    return ValueError(
        f"Expected `{var_name}` to be {expected_str}. Got `{got.__name__}` instead"
    )


def _expected_key_type_value_error(
    got: type, expected: Sequence[type] | type
) -> ValueError:
    return _expected_got_var_value_error(var_name="key", got=got, expected=expected)


def _expected_key_str_value_error(got: type) -> ValueError:
    return _expected_key_type_value_error(got=got, expected=str)


def _expected_value_type_value_error(
    got: type, expected: Sequence[type] | type
) -> ValueError:
    return _expected_got_var_value_error(var_name="value", got=got, expected=expected)


def _no_entry_key_error(key_value: Any) -> KeyError:
    return KeyError(f"No entry for key `{key_value}`")
