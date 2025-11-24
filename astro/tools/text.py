# --- Internal Imports ---
import re
import unicodedata

# --- External Imports ---
from pylatexenc.latex2text import (
    EnvironmentTextSpec,
    LatexNodes2Text,
    MacroTextSpec,
    fmt_matrix_environment_node,
    get_default_latex_context_db,
)

# --- Local Imports ---
from astro.logger import get_loggy
from astro.utilities.display import get_terminal_width

# --- Globals ---
_loggy = get_loggy(__file__)

_ARG_STRIPPING_MACROS = ("mathrm", "mathit", "operatorname")
_NOOP_MACROS = ("limits", "left", "right")


def _needs_parentheses(segment: str) -> bool:
    """Determine whether a fraction component should be wrapped in parentheses.

    Args:
        segment (str): Fraction numerator or denominator candidate.

    Returns:
        bool: True if parentheses improve readability, False otherwise.
    """

    stripped = segment.strip()
    if len(stripped) <= 1:
        return False
    if (stripped.startswith("(") and stripped.endswith(")")) or (
        stripped.startswith("-(") and stripped.endswith(")")
    ):
        return False

    if stripped.startswith("-") and stripped[1:].isalnum():
        return False

    if any(op in stripped for op in ("+", "*", "·", "/", "=")):
        return True
    if "^" in stripped or " " in stripped:
        return True

    has_alpha = any(char.isalpha() for char in stripped)
    has_digit = any(char.isdigit() for char in stripped)
    return has_alpha and has_digit


def _extract_braced_segment(text: str, start: int) -> tuple[str | None, int]:
    """Extract a braced segment starting from the given index.

    Args:
        text (str): Source text containing the braced segment.
        start (int): Index pointing to the opening brace.

    Returns:
        tuple[str | None, int]: Extracted content without outer braces and the
        index just past the closing brace. Content is None if extraction fails.
    """

    if start >= len(text) or text[start] != "{":
        return None, start

    depth = 0
    content_chars: list[str] = []
    index = start

    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
            if depth > 1:
                content_chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(content_chars), index + 1
            content_chars.append(char)
        else:
            content_chars.append(char)
        index += 1

    return None, start


def _format_supersubs(text: str, *, depth: int = 0) -> str:
    """Format superscript and subscript segments for readability.

    Args:
        text (str): Plain text produced by LatexNodes2Text.
        depth (int): Current recursion depth for nested processing.

    Returns:
        str: Text with superscripts and subscripts adjusted.
    """

    if depth >= 2:
        return text

    result: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if char in ("^", "_") and index + 1 < length and text[index + 1] == "{":
            segment, next_index = _extract_braced_segment(text, index + 1)
            if segment is None:
                result.append(char)
                index += 1
                continue

            formatted_segment = _format_supersubs(segment, depth=depth + 1)
            result.append(char)
            if _needs_parentheses(formatted_segment):
                result.append(f"({formatted_segment})")
            else:
                result.append(formatted_segment)
            index = next_index
            continue

        result.append(char)
        index += 1

    return "".join(result)


def _format_fraction_component(component: str) -> str:
    """Format a fraction component with optional parentheses.

    Args:
        component (str): Component string extracted from LaTeX nodes.

    Returns:
        str: Component string with parentheses if required.
    """

    cleaned = component.strip()
    return f"({cleaned})" if _needs_parentheses(cleaned) else cleaned


def _format_fraction_macro(macronode, l2tobj, **_) -> str:
    """Render a \frac macro with readability-preserving parentheses.

    Args:
        macronode: Macro node representing the fraction.
        l2tobj (LatexNodes2Text): Converter performing the conversion.

    Returns:
        str: Formatted fraction string combining numerator and denominator.
    """

    numerator_nodes = macronode.nodeargd.argnlist[0].nodelist
    denominator_nodes = macronode.nodeargd.argnlist[1].nodelist

    numerator = l2tobj.nodelist_to_text(numerator_nodes)
    denominator = l2tobj.nodelist_to_text(denominator_nodes)

    formatted_numerator = _format_fraction_component(numerator)
    formatted_denominator = _format_fraction_component(denominator)
    return f"{formatted_numerator}/{formatted_denominator}"


def _render_first_argument(macronode, l2tobj, **_) -> str:
    """Render only the first argument of a macro.

    Args:
        macronode: Macro node with potential arguments.
        l2tobj (LatexNodes2Text): Converter used for rendering.

    Returns:
        str: Rendered text for the first argument, or an empty string when missing.
    """

    if macronode.nodeargd is None or not macronode.nodeargd.argnlist:
        return ""

    arg_nodes = macronode.nodeargd.argnlist[0].nodelist
    return l2tobj.nodelist_to_text(arg_nodes)


def _build_arg_stripping_macro_specs() -> tuple[MacroTextSpec, ...]:
    """Create macro specs that render only their first argument.

    Returns:
        tuple[MacroTextSpec, ...]: Macro specifications that drop extra
        arguments while preserving the first one.
    """

    return tuple(
        MacroTextSpec(name, simplify_repl=_render_first_argument, discard=False)
        for name in _ARG_STRIPPING_MACROS
    )


def _build_noop_macro_specs() -> tuple[MacroTextSpec, ...]:
    """Create macro specs that collapse to an empty string.

    Returns:
        tuple[MacroTextSpec, ...]: Macro specifications that remove the macro
        from the rendered output.
    """

    return tuple(
        MacroTextSpec(name, simplify_repl="", discard=False) for name in _NOOP_MACROS
    )


def _format_matrix_with_delimiters(left: str, right: str):
    """Wrap matrix-like environment output with custom delimiters.

    Args:
        left (str): Left delimiter to wrap the matrix output.
        right (str): Right delimiter to wrap the matrix output.

    Returns:
        Callable: Formatter callable compatible with LatexNodes2Text
        environment hooks.
    """

    def formatter(node, l2tobj, *, _left: str = left, _right: str = right) -> str:
        formatted = fmt_matrix_environment_node(node, l2tobj)
        if formatted.startswith("[ ") and formatted.endswith(" ]"):
            inner = formatted[2:-2]
            return f"{_left} {inner} {_right}"
        return f"{_left} {formatted} {_right}"

    return formatter


def _build_matrix_environment_specs() -> tuple[EnvironmentTextSpec, ...]:
    """Create environment specs for matrix variants missing in defaults.

    Returns:
        tuple[EnvironmentTextSpec, ...]: Environment specifications that apply
        Astro-specific formatting to matrix-like structures.
    """

    return (
        EnvironmentTextSpec(
            "matrix",
            simplify_repl=_format_matrix_with_delimiters("[", "]"),
            discard=False,
        ),
        EnvironmentTextSpec(
            "Bmatrix",
            simplify_repl=_format_matrix_with_delimiters("{", "}"),
            discard=False,
        ),
        EnvironmentTextSpec(
            "vmatrix",
            simplify_repl=_format_matrix_with_delimiters("|", "|"),
            discard=False,
        ),
        EnvironmentTextSpec(
            "Vmatrix",
            simplify_repl=_format_matrix_with_delimiters("||", "||"),
            discard=False,
        ),
    )


def _create_latex_converter() -> LatexNodes2Text:
    """Create a configured LatexNodes2Text converter with Astro overrides.

    Returns:
        LatexNodes2Text: Converter instance configured with Astro formatting
        helpers.
    """

    latex_context = get_default_latex_context_db()
    latex_context.add_context_category(
        "astro-overrides",
        macros=[
            MacroTextSpec("frac", simplify_repl=_format_fraction_macro, discard=False),
            MacroTextSpec("det", simplify_repl="det", discard=False),
            *_build_arg_stripping_macro_specs(),
            *_build_noop_macro_specs(),
        ],
        environments=_build_matrix_environment_specs(),
        prepend=True,
    )
    return LatexNodes2Text(
        latex_context=latex_context,
        keep_braced_groups=True,
        keep_braced_groups_minlen=2,
    )


_LATEX_CONVERTER = _create_latex_converter()

_SUBSCRIPT_FIRST_BASES = {
    "\\int",
    "\\iint",
    "\\iiint",
    "\\iiiint",
    "\\oint",
    "\\oiint",
    "\\oiiint",
    "\\intop",
}


def _get_script_base(latex_text: str, script_index: int) -> str | None:
    """Identify the base token associated with a script indicator.

    Args:
        latex_text (str): Source LaTeX text.
        script_index (int): Index pointing at the script marker.

    Returns:
        str | None: Token that serves as the script base, or None when the
        base cannot be determined.
    """

    index = script_index - 1
    while index >= 0 and latex_text[index].isspace():
        index -= 1

    if index < 0:
        return None

    end = index
    while index >= 0 and latex_text[index].isalpha():
        index -= 1
    start = index + 1

    if index >= 0 and latex_text[index] == "\\":
        return latex_text[index : end + 1]

    if start <= end:
        return latex_text[start : end + 1]

    return latex_text[end]


def _extract_script_token(text: str, start: int) -> tuple[str, int]:
    """Extract a superscript/subscript token starting at the given index.

    Args:
        text (str): Source LaTeX text.
        start (int): Index pointing to the '^' or '_' character.

    Returns:
        tuple[str, int]: Token substring and index just past the token.
    """

    length = len(text)
    index = start + 1

    while index < length and text[index].isspace():
        index += 1

    if index >= length:
        return text[start:index], index

    if text[index] == "{":
        segment, next_index = _extract_braced_segment(text, index)
        if segment is None:
            return text[start:index], index
        return text[start:next_index], next_index

    # Single character token (e.g., _i or ^2)
    return text[start : index + 1], min(index + 1, length)


def _normalize_supersub_order(latex_text: str) -> str:
    """Normalize LaTeX script tokens so superscripts precede subscripts.

    Args:
        latex_text (str): Source LaTeX text containing script markers.

    Returns:
        str: Adjusted LaTeX string with consistent script ordering.
    """

    result: list[str] = []
    index = 0
    length = len(latex_text)

    while index < length:
        char = latex_text[index]

        if char == "_":
            sub_token, after_sub = _extract_script_token(latex_text, index)
            lookahead = after_sub
            while lookahead < length and latex_text[lookahead].isspace():
                lookahead += 1

            base_token = _get_script_base(latex_text, index)

            if (
                lookahead < length
                and latex_text[lookahead] == "^"
                and base_token not in _SUBSCRIPT_FIRST_BASES
            ):
                sup_token, after_sup = _extract_script_token(latex_text, lookahead)
                result.append(sup_token)
                result.append(sub_token)
                index = after_sup
                continue

            result.append(sub_token)
            index = after_sub
            continue

        if char == "^":
            sup_token, after_sup = _extract_script_token(latex_text, index)
            lookahead = after_sup
            while lookahead < length and latex_text[lookahead].isspace():
                lookahead += 1

            base_token = _get_script_base(latex_text, index)

            if (
                lookahead < length
                and latex_text[lookahead] == "_"
                and base_token not in _SUBSCRIPT_FIRST_BASES
            ):
                sub_token, after_sub = _extract_script_token(latex_text, lookahead)
                result.append(sup_token)
                result.append(sub_token)
                index = after_sub
                continue

            result.append(sup_token)
            index = after_sup
            continue

        result.append(char)
        index += 1

    return "".join(result)


def _normalize_latex_input(latex_text: str) -> str:
    """Normalize user-provided LaTeX before conversion.

    Args:
        latex_text (str): Raw LaTeX string provided by the caller.

    Returns:
        str: Normalized LaTeX string.
    """

    stripped = latex_text.strip()
    return _normalize_supersub_order(stripped)


def _tidy_spacing(text: str) -> str:
    """Normalize spacing around common operators in plain text output.

    Args:
        text (str): Plain text produced by the LatexNodes2Text converter.

    Returns:
        str: Spacing-normalized plain text string.
    """

    adjusted = text
    adjusted = re.sub(r"(?<!\s)=(?=\s)", " =", adjusted)
    adjusted = re.sub(r"(?<=\s)=(?!\s)", "= ", adjusted)
    adjusted = re.sub(r"(?<!\s)\+(?=\s)", " +", adjusted)
    adjusted = re.sub(r"(?<=\s)\+(?!\s)", "+ ", adjusted)
    adjusted = re.sub(r"(?<=\S)([·×⋅])", r" \1", adjusted)
    adjusted = re.sub(r"([·×⋅])(?!\s)", r"\1 ", adjusted)
    return re.sub(r"(?<=\S) {2,}(?=\S)", " ", adjusted)


_FUNCTION_NAMES = ("sin", "cos", "tan", "log", "ln", "exp", "sinh", "cosh", "det")

_IDENTIFIER_ALLOWLIST = {
    *(_FUNCTION_NAMES),
    "arcsin",
    "arccos",
    "arctan",
    "arccot",
    "arccsc",
    "arcsec",
    "sec",
    "csc",
    "cot",
    "det",
    "diag",
    "rank",
    "ker",
    "tr",
    "re",
    "im",
    "max",
    "min",
    "sup",
    "inf",
    "lim",
    "var",
    "cov",
    "erf",
    "erfc",
    "sgn",
    "deg",
    "dim",
    "mod",
    "gcd",
    "lcm",
}

_ACCENT_SUFFIXES = {
    "\u0302": "Hat",
    "\u0307": "Dot",
    "\u0308": "DDot",
    "\u0303": "Tilde",
    "\u0304": "Bar",
    "\u0305": "Overline",
    "\u0306": "Breve",
    "\u20d7": "Vec",
}


def _insert_function_spacing(text: str) -> str:
    """Insert spacing between math function names and their arguments.

    Args:
        text (str): Converter output prior to modifier fallbacks.

    Returns:
        str: Text with lightweight spacing inserted between adjacent factors to
        improve readability.
    """

    adjusted = text
    for name in _FUNCTION_NAMES:
        pattern = rf"\b{name}(?=[^\s(])"
        adjusted = re.sub(pattern, f"{name} ", adjusted)
    return adjusted


def _insert_factor_spacing(text: str) -> str:
    """Add spacing between implicit multiplicative factors.

    Args:
        text (str): Converter output prior to modifier fallbacks.

    Returns:
        str: Text with spacing between recognized function names and their
        arguments.
    """

    operator_chars = {"+", "-", "=", "*", "×", "·", "/", "^", "_", ",", ";", ":"}

    def segment_word_token(token: str) -> list[str]:
        normalized = token.lower()
        if "_" in token or normalized in _IDENTIFIER_ALLOWLIST:
            return [token]

        if not any(unicodedata.combining(char) for char in token):
            return [token]

        segments: list[str] = []
        index = 0
        while index < len(token):
            char = token[index]
            segment_chars = [char]
            index += 1
            while index < len(token) and unicodedata.combining(token[index]):
                segment_chars.append(token[index])
                index += 1
            segments.append("".join(segment_chars))

        return segments if len(segments) > 1 else [token]

    def extract_token(source: str, start: int) -> tuple[str, int, str]:
        char = source[start]
        if char.isspace():
            return char, start + 1, "space"
        if char in "([{":
            return char, start + 1, "open_paren"
        if char in ")]}":
            return char, start + 1, "close_paren"
        if char in operator_chars:
            return char, start + 1, "operator"
        if char.isdigit():
            end = start + 1
            while end < len(source) and source[end].isdigit():
                end += 1
            return source[start:end], end, "number"

        end = start + 1
        allow_continuation = char.isalnum() or char == "_"
        while (
            end < len(source)
            and allow_continuation
            and (
                source[end].isalnum()
                or source[end] == "_"
                or unicodedata.combining(source[end])
            )
        ):
            end += 1
        return source[start:end], end, "word"

    result: list[str] = []
    index = 0
    previous_type: str | None = None

    while index < len(text):
        token, next_index, token_type = extract_token(text, index)

        if token_type == "word":
            emit_tokens = segment_word_token(token)
        else:
            emit_tokens = [token]

        for part in emit_tokens:
            current_type = "word" if token_type == "word" else token_type

            if current_type in {"word", "number"}:
                if previous_type in {"word", "number", "close_paren"}:
                    if not result or not result[-1].endswith(" "):
                        result.append(" ")
            elif current_type == "open_paren":
                if previous_type in {"close_paren"}:
                    if not result or not result[-1].endswith(" "):
                        result.append(" ")
            elif current_type == "space":
                if result and result[-1].endswith(" "):
                    continue
                result.append(" ")
                previous_type = None
                continue

            result.append(part)

            if current_type == "close_paren":
                previous_type = "close_paren"
            elif current_type == "open_paren":
                previous_type = "open_paren"
            else:
                previous_type = current_type

        index = next_index

    return "".join(result)


def _apply_modifier_fallbacks(text: str) -> str:
    """Replace combining accent output with ASCII-friendly modifiers.

    Args:
        text (str): Converter output containing potential combining accents.

    Returns:
        str: String where known combining accents are transformed into suffix
        modifiers (e.g., xdot, Fvec) to improve portability.
    """

    result: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        decomposed = unicodedata.normalize("NFD", char)
        base_char = decomposed[0] if decomposed else char
        combining_chars = list(decomposed[1:])

        lookahead = index + 1
        while lookahead < length and unicodedata.combining(text[lookahead]):
            combining_chars.append(text[lookahead])
            lookahead += 1

        if combining_chars:
            suffixes: list[str] = []
            unknown_combining = False
            for combo in combining_chars:
                suffix = _ACCENT_SUFFIXES.get(combo)
                if suffix is None:
                    unknown_combining = True
                    break
                suffixes.append(suffix)

            if not unknown_combining:
                result.append(f"{base_char}{''.join(suffixes)}")
                index = lookahead
                continue

        result.append(char)
        index += 1

    return "".join(result)


def _post_process_plain_text(text: str) -> str:
    """Apply readability adjustments to plain text LaTeX output.

    Args:
        text (str): Plain text produced by the LaTeX converter.

    Returns:
        str: Readability-enhanced plain text string.
    """

    supersubs_adjusted = _format_supersubs(text)
    function_adjusted = _insert_function_spacing(supersubs_adjusted)
    factor_adjusted = _insert_factor_spacing(function_adjusted)
    modifier_adjusted = _apply_modifier_fallbacks(factor_adjusted)
    adjusted = modifier_adjusted.replace("{", "(").replace("}", ")")
    return _tidy_spacing(adjusted)


# --- Math/LaTeX Functions ---
def latex_to_text(latex_text: str) -> str:
    """Convert a LaTeX mathematical expression to readable plain text.

    Uses a pre-configured LatexNodes2Text converter with Astro-specific
    overrides to render mathematical expressions while preserving key visual
    cues such as grouped fractions.

    Args:
        latex_text (str): LaTeX mathematical expression string to convert.

    Returns:
        str: Plain text representation of the mathematical expression.

    Raises:
        ExpectedVariableType: If latex_text is not a string.
        ValueError: If latex_text is empty after normalization.
        ParseError: If the LaTeX expression cannot be converted to plain text.

    Examples:
        >>> latex_to_text(r"\\frac{1}{2}")
        '1/2'
        >>> latex_to_text(r"\\frac{\\pi^2}{6}")
        '(π^2)/6'
    """

    if not isinstance(latex_text, str):
        raise _loggy.ExpectedVariableType(
            var_name="latex_text",
            expected=str,
            got=type(latex_text),
            with_value=latex_text,
        )

    normalized_input = _normalize_latex_input(latex_text)
    if len(normalized_input) == 0:
        raise _loggy.ValueError("Input latex expression is empty")

    try:
        plain_text = _LATEX_CONVERTER.latex_to_text(normalized_input)
    except Exception as error:  # noqa: BLE001 - propagate converter errors
        raise _loggy.ParseError(
            type_to_parse=str,
            value_to_parse=normalized_input,
            expected_type=str,
            caused_by=error,
            reason="Failed to convert LaTeX to plain text",
        ) from error

    return _post_process_plain_text(plain_text)


if __name__ == "__main__":
    # Test with various mathematical expressions
    test_expressions = [
        r"x + y^2 + \exp(\frac{1}{2})",
        r"\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}",
        r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}",
        r"\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t}",
        r"\mathcal{L} = \frac{1}{2}m\dot{x}^2 - V(x)",
        r"\left|\psi\right\rangle = \alpha\left|0\right\rangle + \beta\left|1\right\rangle",
        r"\hat{H}\psi = E\psi \quad \text{where} \quad \hat{H} = -\frac{\hbar^2}{2m}\nabla^2 + V",
        r"\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u",
        r"e^{x^2} + e^{\sin{x}} + e^{\frac{1}{2}}",
        r"V^2_{p + q} + V^2_{pq} + V^2_{p_{i + j}}",
        r"V_{p + q}^2 + V_{pq}^2 + V_{p_{i + j}}^2",
        r"V(u, v) = \iint I(l, m) e^{-2\pi i (ul + vm)} \, dl \, dm",
        r"E_{\mathrm{obs}}(t) = \int B(\nu) E_{\mathrm{emit}}(t - \tau(\nu)) \, d\nu",
        r"\phi_{ij}(\nu) = 2\pi \nu \, \tau_{ij} + \theta_{ij}(\nu)",
        r"\mathcal{L} = \sum_{i=1}^{N} \left| V_{i}^{\mathrm{meas}} - V_{i}^{\mathrm{model}^\eta} \right|^2 / \sigma_i^2",
        r"w_{pq}(\mathbf{b}) = e^{- (\mathbf{b} \cdot \hat{s}_{pq})^2 / (2 \sigma^2)}",
        r"f'(x) := \lim\limits_{h \to \infty} \frac{f(x + h) - f(x)}{h}",
        # Nested fractions and complex superscripts
        r"\frac{1}{1 + \frac{1}{1 + \frac{1}{x}}}",
        r"\frac{a^{b^c}}{d_{e_f}}",
        r"e^{-\frac{(x - \mu)^2}{2\sigma^2}}",
        # Multiple integral expressions
        r"\iiint_V \nabla \cdot \vec{F} \, dV = \iint_S \vec{F} \cdot \hat{n} \, dS",
        r"\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} e^{-(x^2 + y^2)} \, dx \, dy = \pi",
        # Matrix and tensor notation
        r"g_{\mu\nu} = \eta_{\mu\nu} + h_{\mu\nu}",
        r"R^\mu_{\nu\lambda\sigma} = \partial_\lambda \Gamma^\mu_{\nu\sigma} - \partial_\sigma \Gamma^\mu_{\nu\lambda}",
        r"T^{\mu\nu} = \frac{2}{\sqrt{-g}} \frac{\delta S}{\delta g_{\mu\nu}}",
        # Complex quantum mechanics expressions
        r"\left[\hat{x}, \hat{p}\right] = i\hbar",
        r"\langle \psi | \hat{A} | \phi \rangle = \int \psi^*(x) \hat{A} \phi(x) \, dx",
        r"|\psi(t)\rangle = e^{-i\hat{H}t/\hbar} |\psi(0)\rangle",
        # Statistical mechanics and thermodynamics
        r"Z = \sum_n e^{-\beta E_n} = \mathrm{Tr}\left(e^{-\beta \hat{H}}\right)",
        r"S = -k_B \sum_i p_i \ln p_i",
        r"F = E - TS = -k_B T \ln Z",
        # Field theory expressions
        r"\mathcal{L} = \frac{1}{2}(\partial_\mu \phi)^2 - \frac{1}{2}m^2\phi^2 - \frac{\lambda}{4!}\phi^4",
        r"D_\mu = \partial_\mu - ieA_\mu",
        r"\langle 0 | T\{\phi(x)\phi(y)\} | 0 \rangle = \int \frac{d^4p}{(2\pi)^4} \frac{ie^{-ip(x-y)}}{p^2 - m^2 + i\epsilon}",
        # General relativity
        r"R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4}T_{\mu\nu}",
        r"ds^2 = -c^2dt^2 + a(t)^2\left[\frac{dr^2}{1-kr^2} + r^2(d\theta^2 + \sin^2\theta \, d\phi^2)\right]",
        # Complex sums and products
        r"\prod_{n=1}^{\infty} \left(1 - \frac{x^2}{n^2\pi^2}\right) = \frac{\sin x}{x}",
        r"\sum_{k=0}^{n} \binom{n}{k} x^k y^{n-k} = (x + y)^n",
        r"\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s} = \prod_p \frac{1}{1 - p^{-s}}",
        # Special functions
        r"\Gamma(z) = \int_0^\infty t^{z-1} e^{-t} \, dt",
        r"J_\nu(x) = \sum_{m=0}^{\infty} \frac{(-1)^m}{m!\Gamma(m+\nu+1)} \left(\frac{x}{2}\right)^{2m+\nu}",
        r"P_n(x) = \frac{1}{2^n n!} \frac{d^n}{dx^n}(x^2 - 1)^n",
        # Astrophysics specific
        r"L = 4\pi R^2 \sigma T_{\mathrm{eff}}^4",
        r"\frac{dN}{dE} = \frac{N_0}{E_0} \left(\frac{E}{E_0}\right)^{-\Gamma} \exp\left(-\frac{E}{E_{\mathrm{cut}}}\right)",
        r"\tau_\nu = \int_0^s \kappa_\nu(s') \rho(s') \, ds'",
        r"I_\nu(s) = I_\nu(0) e^{-\tau_\nu(s)} + \int_0^s S_\nu(s') e^{-\tau_\nu(s,s')} \, d\tau_\nu",
        # Vectors and matrices
        r"\vec{v} = \begin{pmatrix} x \\ y \\ z \end{pmatrix}",
        r"\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}",
        r"\det(\mathbf{A}) = \begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad - bc",
        r"\mathbf{R} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}",
        r"\nabla = \begin{pmatrix} \frac{\partial}{\partial x} \\ \frac{\partial}{\partial y} \\ \frac{\partial}{\partial z} \end{pmatrix}",
        r"\mathbf{J} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}",
        r"\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}",
        r"\vec{F} \cdot \vec{r} = \begin{pmatrix} F_x \\ F_y \\ F_z \end{pmatrix} \cdot \begin{pmatrix} x \\ y \\ z \end{pmatrix} = F_x x + F_y y + F_z z",
        # Cases environments
        r"|x| = \begin{cases} x & \text{if } x \geq 0 \\ -x & \text{if } x < 0 \end{cases}",
        r"f(x) = \begin{cases} 0 & x < 0 \\ x^2 & 0 \leq x < 1 \\ 2x - 1 & x \geq 1 \end{cases}",
        r"\delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}",
        r"H(x) = \begin{cases} 0 & x < 0 \\ \frac{1}{2} & x = 0 \\ 1 & x > 0 \end{cases}",
        r"\rho(\vec{r}) = \begin{cases} \rho_0 & r \leq R \\ 0 & r > R \end{cases}",
        r"V(r) = \begin{cases} -\frac{GM}{r} & r \geq R \\ -\frac{GM}{2R^3}(3R^2 - r^2) & r < R \end{cases}",
        r"\chi^2 = \begin{cases} \sum_{i=1}^{n} \frac{(O_i - E_i)^2}{E_i} & \text{Pearson} \\ 2\sum_{i=1}^{n} O_i \ln\left(\frac{O_i}{E_i}\right) & \text{Likelihood ratio} \end{cases}",
    ]

    import statistics
    from timeit import timeit

    from astro.utilities.timing import create_time_converter, create_timer

    output_unit = "msec"
    converter = create_time_converter("sec", output_unit)
    start, stop = create_timer("sec")
    n = 5
    all_times = []
    start()
    for num, latex_text in enumerate(test_expressions, start=1):
        print(f"{num}. {latex_text}")

        try:
            unicode_latex = latex_to_text(latex_text)

            print(f"Output:\n{unicode_latex}")

            times = [
                timeit(lambda: latex_to_text(latex_text), number=1) for _ in range(n)
            ]
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if n > 1 else 0.0
            print(
                f"Time: {converter(avg_time):.4f}±{converter(std_time):.4f}{output_unit}"
            )
            all_times.extend(times)
        except Exception as error:
            print(f"Error: {error}")
        print("-" * get_terminal_width())
    total_elapsed = stop()
    print(
        f"Total time taken to process {n * len(test_expressions)} latex expressions: {total_elapsed:.2f}sec"
    )
    avg_time = statistics.mean(all_times)
    std_time = statistics.stdev(all_times)
    print(
        f"Total average time: {converter(avg_time):.4f}±{converter(std_time):.4f}{output_unit}"
    )
