"""Regression tests for latex_to_text KaTeX-style conversions.

This suite exercises a broad set of math expressions that large language
models commonly emit in Markdown/KaTeX blocks. The goal is to guard against
regressions in spacing, superscript ordering, and modifier fallbacks handled by
astro.tools.text.latex_to_text.
"""

from __future__ import annotations

import pytest

from astro.errors import ExpectedVariableType
from astro.tools.text import latex_to_text

LLM_LATEX_CASES: list[tuple[str, str]] = [
    (r"x + y^2", "x + y^2"),
    (r"\frac{1}{2}", "1/2"),
    (r"\frac{\pi^2}{6}", "(π^2)/6"),
    (r"\sum_{n=1}^{\infty} \frac{1}{n^2}", "∑^∞_(n=1) 1/(n^2)"),
    (r"\int_{0}^{\infty} e^{-x^2} dx", "∫_0^∞ e^(-x^2) dx"),
    (r"\vec{E} \cdot \vec{B}", "EVec · BVec"),
    (r"\dot{x}", "xDot"),
    (r"\mathcal{L} = \frac{1}{2}m\dot{x}^2 - V(x)", "ℒ = 1/2 mxDot^2 - V(x)"),
    (
        r"\left|\psi\right\rangle = \alpha\left|0\right\rangle + \beta\left|1\right\rangle",
        "| ψ ⟩ = α | 0 ⟩ + β | 1 ⟩",
    ),
    (
        r"\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}",
        "𝐀 = [ a b; c d ]",
    ),
    (
        r"\det(\mathbf{A}) = \begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad - bc",
        "det(𝐀) = | a b; c d | = ad - bc",
    ),
    (r"\frac{1}{1 + \frac{1}{x}}", "1/(1 + 1/x)"),
    (r"e^{-\frac{(x - \mu)^2}{2\sigma^2}}", "e^-((x - μ)^2)/(2 σ^2)"),
    (r"\sum_{k=0}^{n} \binom{n}{k} x^k y^{n-k}", "∑^n_(k=0) nk x^k y^n-k"),
    (
        r"\prod_{n=1}^{\infty} \left(1 - \frac{x^2}{n^2\pi^2}\right)",
        "∏^∞_(n=1) (1 - (x^2)/(n^2 π^2))",
    ),
    (r"\vec{F} = m\vec{a}", "FVec = m aVec"),
    (
        r"\frac{dN}{dE} = \frac{N_0}{E_0} \left(\frac{E}{E_0}\right)^{-\Gamma}",
        "dN/dE = (N_0)/(E_0) (E/(E_0))^-Γ",
    ),
    (
        r"\nabla = \begin{pmatrix} \frac{\partial}{\partial x} \\ \frac{\partial}{\partial y} \\ \frac{\partial}{\partial z} \end{pmatrix}",
        "∇ = [ ∂/∂ x; ∂/∂ y; ∂/∂ z ]",
    ),
    (r"T_{\mathrm{eff}}^4", "T^4_eff"),
    (r"\frac{a^{b^c}}{d_{e_f}}", "(a^(b^c))/d_e_f"),
    (r"\int_{-\infty}^{\infty} e^{-x^2} \, dx", "∫_-∞^∞ e^(-x^2) dx"),
    (r"\frac{d^2 y}{dx^2}", "(d^2 y)/(dx^2)"),
    (r"\sqrt{1 - e^{-\tau}}", "√(1 - e^-τ)"),
    (r"P(\text{signal}) = 1 - e^{-\lambda t}", "P(signal) = 1 - e^-λt"),
    (
        r"f(x) = \begin{cases} x^2 & x < 0 \\ x & x \geq 0 \end{cases}",
        "f(x) = x^2 x < 0 x x ≥ 0 ",
    ),
    (r"x_{i}^{2}", "x^2_i"),
    (
        r"\Gamma(z) = \int_0^\infty t^{z-1} e^{-t} \, dt",
        "Γ(z) = ∫_0^∞ t^(z-1) e^-t dt",
    ),
    (
        r"\vec{F} \cdot \vec{r} = \begin{pmatrix} F_x \\ F_y \\ F_z \end{pmatrix} \cdot \begin{pmatrix} x \\ y \\ z \end{pmatrix} = F_x x + F_y y + F_z z",
        "FVec · rVec = [ F_x; F_y; F_z ] · [ x; y; z ] = F_x x + F_y y + F_z z",
    ),
]


@pytest.mark.parametrize(("expression", "expected"), LLM_LATEX_CASES)
def test_latex_to_text_examples(expression: str, expected: str) -> None:
    """Ensure latex_to_text matches curated KaTeX-style conversions."""

    assert latex_to_text(expression) == expected


def test_latex_to_text_rejects_non_string() -> None:
    """Ensure type validation errors bubble up for non-string inputs."""

    with pytest.raises(ExpectedVariableType):
        latex_to_text(123)  # type: ignore[arg-type]


def test_latex_to_text_rejects_blank_input() -> None:
    """Ensure blank input is rejected to prevent silent whitespace results."""

    with pytest.raises(ValueError):
        latex_to_text("   ")
