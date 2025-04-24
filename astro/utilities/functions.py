from datetime import datetime, timezone
import inspect
from inspect import Parameter
import math
from pathlib import Path
import re
from typing import (
    Any,
    Literal,
    Optional,
    Self,
    Type,
    Union,
    get_args,
    get_type_hints,
    get_origin,
)

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, create_model, model_validator
from requests import request, RequestException
import yaml
import docstring_parser
from docstring_parser import Docstring

ASTRO_DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
LiteralType = Any


def get_datetime_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def get_timestamp(
    datet: Optional[datetime] = None, pattern: str = ASTRO_DATETIME_FORMAT
) -> str:
    if datet is None:
        return get_datetime_now().strftime(pattern)
    else:
        return datet.strftime(pattern)


def from_timestamp(timestamp: str, pattern: str = ASTRO_DATETIME_FORMAT) -> datetime:
    try:
        dt = datetime.strptime(timestamp, pattern)
        return dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format. Got: {timestamp}") from e


def datetime_to_local(datet: datetime) -> datetime:
    return datet.astimezone()


def timestamp_to_local(timestamp: str, pattern: str = ASTRO_DATETIME_FORMAT) -> str:
    return get_timestamp(datetime_to_local(from_timestamp(timestamp, pattern)), pattern)


def is_api_alive(
    url: str,
    method: Literal["GET", "POST"] = "GET",
    headers: Optional[dict[str, str]] = None,
    timeout: Optional[float] = None,
    success_codes: Optional[tuple[int]] = None,
) -> bool:

    if url is None or len(url) == 0:
        raise ValueError("Empty or null URL for API check.")

    if headers is None or len(headers) == 0:
        headers = {}

    if timeout is None:
        timeout = 5
    elif timeout <= 0.0:
        raise ValueError(f"Only positive values allowed for timeout. Got {timeout=} ")

    if success_codes is None:
        success_codes = [200]
    elif len(success_codes) == 0:
        raise ValueError("No success codes provided for API check.")

    try:
        response = request(
            method=method,
            url=url,
            headers=headers,
            timeout=timeout,
        )
        return response.status_code in success_codes
    except (RequestException, ConnectionError, TimeoutError):
        return False


def round_up(number: float, ndigits: int = 0) -> int:
    offset = math.pow(10, ndigits)
    return round(math.ceil(number * offset) / offset)


def round_down(number: float, ndigits: int = 0) -> int:
    offset = math.pow(10, ndigits)
    return round(math.floor(number * offset) / offset)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(Path(path), "r") as file:
        return yaml.safe_load(file)


def strtime_to_seconds(time_str: str) -> float:
    """
    Convert a time string with unit to seconds.

    Parameters
    ----------
    time_str : str
        String representing time with unit (e.g., '2s', '10min', '1.5h')

    Returns
    -------
    float
        Time value converted to seconds

    Raises
    ------
    ValueError
        If the input string format is invalid or missing units

    Examples
    --------
    >>> strtime_to_seconds('2s')
    2.0
    >>> strtime_to_seconds('1.5h')
    5400.0
    >>> strtime_to_seconds('10min')
    600.0
    """
    if not isinstance(time_str, str):
        raise ValueError("Input must be a string")

    # Strip whitespace and convert to lowercase
    time_str = time_str.strip().lower()

    # Regular expression to match a number followed by a unit
    pattern = r"^(\d+\.?\d*)([a-z]+)$"
    match = re.match(pattern, time_str)

    if not match:
        raise ValueError(
            f"Invalid time format. Expected format: <number><unit> (e.g., '2s', '10min'). Got {time_str}"
        )

    value, unit = match.groups()

    # Convert value to float
    try:
        value = float(value)
    except ValueError:
        raise ValueError(f"Invalid numeric value: {value}")

    # Define unit conversion factors (to seconds)
    unit_map = {
        # Seconds
        "s": 1,
        "sec": 1,
        "second": 1,
        "seconds": 1,
        # Minutes
        "m": 60,
        "min": 60,
        "minute": 60,
        "minutes": 60,
        # Hours
        "h": 3600,
        "hr": 3600,
        "hour": 3600,
        "hours": 3600,
        # Days
        "d": 86400,
        "day": 86400,
        "days": 86400,
        # Weeks
        "w": 604800,
        "wk": 604800,
        "week": 604800,
        "weeks": 604800,
    }

    # Check if the unit is valid
    if unit not in unit_map:
        raise ValueError(
            f"Unknown time unit: '{unit}'. Supported units: {', '.join(unit_map.keys())}"
        )

    # Convert to seconds
    return value * unit_map[unit]


def floor_binary_power(value: int | float) -> int:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("Input must be a positive integer or float")

    if isinstance(value, float):
        value = int(value)

    return 1 << (value.bit_length() - 1)


def euclid_distance(
    x: int | float | np.number | NDArray[np.number],
    y: int | float | np.number | NDArray[np.number],
) -> float:
    return np.pow(x - y, 2)


def abs_distance(
    x: int | float | np.number | NDArray[np.number],
    y: int | float | np.number | NDArray[np.number],
) -> float:
    return np.abs(x - y)


def literal_to_tuple(literal: LiteralType) -> tuple[str, ...]:
    return get_args(literal)


def classify_str(text: str) -> str:
    return "".join(
        word.capitalize() for word in text.replace("-", " ").replace("_", " ").split()
    )


class Argument(BaseModel):
    name: str
    type: Any
    default: Any | None = None
    description: str | None = None
    optional: bool = False

    @model_validator(mode="after")
    def validate_default(cls, values: Self) -> Self:
        if values.default is not None:
            try:
                expected_type = values.type
                origin = get_origin(expected_type)
                if origin:
                    if not isinstance(values.default, origin):
                        raise TypeError
                elif not isinstance(values.default, expected_type):
                    raise TypeError
            except TypeError:
                raise TypeError(
                    f"Default value for '{values.name}' must be of type {expected_type}, "
                    f"got {type(values.default)} instead."
                )
        return values

    @staticmethod
    def _is_optional(annotation: Any) -> bool:
        origin = get_origin(annotation)
        args = get_args(annotation)
        return annotation is Optional or (origin is Union and type(None) in args)

    @staticmethod
    def from_parameter(parameter: Parameter) -> Self:
        param_type = (
            parameter.annotation if parameter.annotation is not Parameter.empty else Any
        )
        default = None if parameter.default is Parameter.empty else parameter.default

        is_optional = (
            Argument._is_optional(param_type)
            or parameter.default is not Parameter.empty
        )

        return Argument(
            name=parameter.name, type=param_type, default=default, optional=is_optional
        )


def function_arguments_schema(function: callable) -> list[Argument]:
    signature = inspect.signature(function)

    arguments = []
    for parameter in signature.parameters.values():
        arguments.append(Argument.from_parameter(parameter))

    return arguments


def build_function_input_schema(
    function: callable,
    name: str | None = None,
) -> Type[BaseModel]:
    # 1) pull in the docstring and parse it
    raw_doc = inspect.getdoc(function) or ""
    parsed = docstring_parser.parse(raw_doc)
    # build a map name -> description
    param_descriptions = {p.arg_name: p.description for p in parsed.params}

    # 2) gather your existing arguments
    arguments = function_arguments_schema(function)

    # 3) build the fields dict, injecting description where available
    fields: dict[str, tuple[Any, Any]] = {}
    for arg in arguments:
        desc = param_descriptions.get(arg.name, "")  # blank if not found
        if arg.optional:
            fields[arg.name] = (
                arg.type,
                Field(arg.default, description=desc),
            )
        else:
            fields[arg.name] = (
                arg.type,
                Field(..., description=desc),
            )

    # 4) decide on a model name if none given
    if name is None:
        name = classify_str(function.__name__) + "Args"

    return create_model(name, **fields)


# def build_function_arguments_model(
#     function: callable, name: str | None = None
# ) -> Type[BaseModel]:
#     fields = {}
#     arguments = function_arguments_schema(function)
#     print(arguments)
#     for argument in arguments:
#         if argument.optional:
#             fields[argument.name] = (argument.type, argument.default)
#         else:
#             fields[argument.name] = (argument.type, ...)

#     if name is None:
#         name = classify_str(function.__name__) + "Args"
#     return create_model(name, **fields)


def format_annotation(annotation):
    """Format a type annotation into a readable string representation."""
    # Handle None case
    if annotation is None:
        return "None"

    # Handle Any case
    if annotation is Any:
        return "Any"

    # Handle Union types (including Optional which is Union[T, None])
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        # Special case for Optional[T] which is Union[T, None]
        if len(args) == 2 and type(None) in args:
            other_type = args[0] if args[1] is type(None) else args[1]
            return f"Optional[{format_annotation(other_type)}]"
        else:
            return " | ".join(format_annotation(arg) for arg in args)

    # Handle generic types like List[str], Dict[str, int], etc.
    if origin is not None:
        args = get_args(annotation)
        if not args:
            return str(origin.__name__)

        origin_name = origin.__name__ if hasattr(origin, "__name__") else str(origin)

        # Format the type arguments
        args_str = ", ".join(format_annotation(arg) for arg in args)
        return f"{origin_name}[{args_str}]"

    # Handle simple types with __name__
    if hasattr(annotation, "__name__"):
        return annotation.__name__

    # Fallback for other types
    return str(annotation)


def format_function(function: callable, include_docstring: bool = True) -> str:
    """
    Format a function signature as a string with type annotations and optionally its docstring.

    Args:
        function: The function to format
        include_docstring: Whether to include the docstring in the output

    Returns:
        A formatted string representation of the function
    """
    try:
        # Try to get type hints, which resolves forward references
        type_hints = get_type_hints(function)
    except (NameError, TypeError):
        # Fall back if there are unresolved forward references
        type_hints = getattr(function, "__annotations__", {})

    signature = inspect.signature(function)
    return_type = type_hints.get("return", signature.return_annotation)
    if return_type is signature.empty:
        return_type = None

    parameters = []
    for name, param in signature.parameters.items():
        annotation = type_hints.get(name, param.annotation)
        if annotation is param.empty:
            annotation = Any

        # Format the annotation
        annotation_str = format_annotation(annotation)

        # Handle default values
        if param.default is not param.empty:
            if param.default is None:
                default = " = None"
            elif isinstance(param.default, str):
                default = f' = "{param.default}"'
            else:
                default = f" = {param.default}"
        else:
            default = ""

        parameters.append(f"{name}: {annotation_str}{default}")

    parameters_str = ", ".join(parameters)
    return_str = format_annotation(return_type)
    function_signature = f"{function.__name__}({parameters_str}) -> {return_str}"

    # Add docstring if requested and available
    if include_docstring:
        docstring = inspect.getdoc(function)
        if docstring:
            # Format the docstring with proper indentation
            formatted_docstring = "\n    ".join(docstring.split("\n"))
            return f'{function_signature}:\n    """{formatted_docstring}"""'

    return function_signature


if __name__ == "__main__":
    from typing import List, Optional, Union

    def tester(
        arg1: bool,
        arg2: List[str],
        arg3: Optional[int] = None,
        arg4: Union[float, None] = None,
        arg5: float = 1.0,
    ):
        """Testing function for testing.

        This function is a test function for our function inspecting
        functions. Lets see how it does when parsed!

        Parameters
        ----------
        arg1 : bool
            Argument that is a bool
        arg2 : List[str]
            A list of names
        arg3 : Optional[int], optional
            A number if you like, by default None
        arg4 : Union[float, None], optional
            Maybe give a float, by default None
        arg5 : float, optional
            This is another float, by default 1.0
        """
        pass

    print(format_function(tester))
    print("\n" + "-" * 60 + "\n")

    # Test without docstring
    print(format_function(tester, include_docstring=False))
    print("\n" + "-" * 60 + "\n")

    # Test function without a docstring
    def no_doc_function(a: int, b: str = "test") -> bool:
        return True

    def foo(a: int, b: str = "hello") -> str:
        """
        foo-boo your shoe

        Want to foo-boo your shoe. Then use this function.

        Parameters
        ----------
        a : int
            Look at this argument.
        b : str, optional
            This is another argument, by default "hello"

        Returns
        -------
        str
            Some sort of result.
        """
        # foo-boo your shoe

        # Want to foo-boo your shoe. Then use this function.

        # :param a: Look at this argument
        #     :type a: int
        # :param b: This is another argument
        #     :type b: str, optional
        #     :default b: "hello"
        # :return: Some sort of result
        # :rtype: str
        #
        return b * a

    print(format_function(no_doc_function))

    print("\n" + "-" * 60 + "\n")
    print(function_arguments_schema(tester))

    print("\n" + "-" * 60 + "\n")
    print(f"{classify_str(no_doc_function.__name__)}")
    NoDocFunctionArgs = build_function_arguments_model(no_doc_function)
    print(NoDocFunctionArgs)
    print(f"{NoDocFunctionArgs(a=2, b="hello")=}")

    print("\n" + "-" * 60 + "\n")
    FooArgs = build_function_arguments_model(foo)
    print(f"{FooArgs(a=10, b="wow").model_json_schema()}")

    print("\n" + "-" * 60 + "\n")
