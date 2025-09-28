"""Trigonometry helpers for plotting and identity verification."""

from __future__ import annotations

from math import pi
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
from sympy import Eq, Symbol, solveset, sympify

from .utils import ComputationResult, PlotElement


_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "cot": lambda x: 1 / np.tan(x),
    "sec": lambda x: 1 / np.cos(x),
    "csc": lambda x: 1 / np.sin(x),
}


def plot_basic_trig(func: str, domain: Tuple[float, float] = (-2 * pi, 2 * pi), resolution: int = 800) -> ComputationResult:
    if func not in _FUNCTIONS:
        return ComputationResult(
            status="error",
            op="plot_trig",
            payload={"message": f"Unknown function {func}"},
        )

    xs = np.linspace(domain[0], domain[1], resolution)
    ys = _FUNCTIONS[func](xs)

    plot = [
        PlotElement(
            type="curve",
            data={"points": list(zip(xs.tolist(), ys.tolist())), "label": func},
            style={"color": "#1f77b4"},
        )
    ]

    steps = [f"Plot y = {func}(x) over domain {domain}"]

    return ComputationResult(
        status="ok",
        op="plot_trig",
        payload={"function": func, "domain": domain},
        steps=steps,
        plot_elements=plot,
    )


def transform_trig(
    amplitude: float,
    frequency: float,
    phase: float,
    vertical: float,
    base: str = "sin",
    domain: Tuple[float, float] = (-2 * pi, 2 * pi),
    resolution: int = 800,
) -> ComputationResult:
    if base not in _FUNCTIONS:
        return ComputationResult(
            status="error",
            op="transform_trig",
            payload={"message": f"Unknown base function {base}"},
        )

    xs = np.linspace(domain[0], domain[1], resolution)
    ys = amplitude * _FUNCTIONS[base](frequency * xs + phase) + vertical

    plot = [
        PlotElement(
            type="curve",
            data={"points": list(zip(xs.tolist(), ys.tolist())), "label": "Transformed"},
            style={"color": "#ff7f0e"},
        ),
    ]

    steps = [
        rf"Apply transformation y = {amplitude:.3g}{base}({frequency:.3g}x + {phase:.3g}) + {vertical:.3g}",
    ]

    payload = {
        "amplitude": amplitude,
        "frequency": frequency,
        "phase": phase,
        "vertical_shift": vertical,
    }

    return ComputationResult(
        status="ok",
        op="transform_trig",
        payload=payload,
        steps=steps,
        plot_elements=plot,
    )


def verify_identity(lhs: str, rhs: str, samples: Iterable[float] | None = None) -> ComputationResult:
    lhs_expr = sympify(lhs)
    rhs_expr = sympify(rhs)
    eq = Eq(lhs_expr, rhs_expr)

    points = samples or np.linspace(-2 * pi, 2 * pi, 20)
    differences = []
    for value in points:
        subs = {Symbol("x"): value}
        differences.append(float((lhs_expr - rhs_expr).subs(subs).evalf()))

    max_error = max(abs(d) for d in differences)
    steps = [rf"Max numerical error over samples: {max_error:.3g}"]

    plot = []

    return ComputationResult(
        status="ok" if max_error < 1e-6 else "warning",
        op="verify_trig_identity",
        payload={"max_error": max_error},
        steps=steps,
        plot_elements=plot,
        warnings=["numerical_tolerance"] if max_error >= 1e-6 else [],
    )


def solve_trig_equation(expression: str, domain: Tuple[float, float]) -> ComputationResult:
    expr = sympify(expression)
    solutions = solveset(expr, Symbol("x"), domain=sympy_interval(domain))
    numeric = [float(sol.evalf()) for sol in solutions]
    steps = [f"Solve {expression} = 0 over domain {domain}"]

    plot = [
        PlotElement(
            type="points",
            data={"coords": [(x_val, 0) for x_val in numeric]},
            style={"color": "#2ca02c"},
        )
    ]

    return ComputationResult(
        status="ok",
        op="solve_trig_equation",
        payload={"solutions": numeric},
        steps=steps,
        plot_elements=plot,
    )


def sympy_interval(domain: Tuple[float, float]):
    from sympy import Interval

    return Interval(domain[0], domain[1])
