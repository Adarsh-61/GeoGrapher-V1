"""Conic section helpers covering NCERT-standard forms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sympy import Matrix, Rational, Symbol, symbols
from sympy.abc import x, y
from sympy.solvers import solve

from .lines import Line
from .points import Point
from .utils import ComputationResult, PlotElement


def parabola_y2_4ax(a: float) -> ComputationResult:
    if a == 0:
        return ComputationResult(
            status="error",
            op="parabola_y2_4ax",
            payload={"message": "Parameter 'a' must be non-zero"},
            warnings=["degenerate"]
        )

    focus = Point(a, 0, label="F")
    directrix = Line.from_general(1, 0, -a, label="directrix")
    steps = [
        rf"Focus: (a, 0) = ({a:.6g}, 0)",
        rf"Directrix: x = -a = {-a:.6g}",
        "Axis: x-axis",
        rf"Latus rectum length: |4a| = {abs(4*a):.6g}",
    ]

    t_vals = np.linspace(-5, 5, 200)
    points = [( (t**2)/(4*a), t ) for t in t_vals]

    plot = [
        PlotElement(
            type="curve",
            data={"points": points, "label": "Parabola"},
            style={"color": "#1f77b4"},
        ),
        PlotElement(
            type="point",
            data={"coords": focus.as_tuple(), "label": "F"},
            style={"color": "#d62728"},
        ),
        PlotElement(
            type="line",
            data={"from": (-a, -10), "to": (-a, 10), "label": "Directrix"},
            style={"color": "#2ca02c", "dash": "dash"},
        ),
    ]

    payload = {
        "vertex": (0.0, 0.0),
        "focus": focus.as_tuple(),
        "directrix": {"A": 1.0, "B": 0.0, "C": -a},
        "axis": "x-axis",
        "latus_rectum": abs(4 * a),
        "parametric": {"x": "at^2", "y": "2at"},
    }

    return ComputationResult(
        status="ok",
        op="parabola_y2_4ax",
        payload=payload,
        steps=steps,
        plot_elements=plot,
    )


def parabola_x2_4ay(a: float) -> ComputationResult:
    if a == 0:
        return ComputationResult(
            status="error",
            op="parabola_x2_4ay",
            payload={"message": "Parameter 'a' must be non-zero"},
            warnings=["degenerate"],
        )

    focus = Point(0, a, label="F")
    directrix = Line.from_general(0, 1, -a, label="directrix")
    steps = [
        rf"Focus: (0, a) = (0, {a:.6g})",
        rf"Directrix: y = -a = {-a:.6g}",
        "Axis: y-axis",
        rf"Latus rectum length: |4a| = {abs(4*a):.6g}",
    ]

    t_vals = np.linspace(-5, 5, 200)
    points = [(t, (t**2)/(4*a)) for t in t_vals]

    plot = [
        PlotElement(
            type="curve",
            data={"points": points, "label": "Parabola"},
            style={"color": "#1f77b4"},
        ),
        PlotElement(
            type="point",
            data={"coords": focus.as_tuple(), "label": "F"},
            style={"color": "#d62728"},
        ),
        PlotElement(
            type="line",
            data={"from": (-10, -a), "to": (10, -a), "label": "Directrix"},
            style={"color": "#2ca02c", "dash": "dash"},
        ),
    ]

    payload = {
        "vertex": (0.0, 0.0),
        "focus": focus.as_tuple(),
        "directrix": {"A": 0.0, "B": 1.0, "C": -a},
        "axis": "y-axis",
        "latus_rectum": abs(4 * a),
        "parametric": {"x": "2at", "y": "at^2"},
    }

    return ComputationResult(
        status="ok",
        op="parabola_x2_4ay",
        payload=payload,
        steps=steps,
        plot_elements=plot,
    )


def ellipse_standard(a: float, b: float) -> ComputationResult:
    if a <= 0 or b <= 0:
        return ComputationResult(
            status="error",
            op="ellipse_standard",
            payload={"message": "Semi-axes must be positive"},
            warnings=["degenerate"],
        )

    c = float(np.sqrt(abs(a**2 - b**2)))
    eccentricity = c / max(a, b)
    focus1 = (c, 0)
    focus2 = (-c, 0)

    t_vals = np.linspace(0, 2 * np.pi, 400)
    points = [(a * np.cos(t), b * np.sin(t)) for t in t_vals]

    steps = [
        rf"Major axis = {2*max(a,b):.6g}",
        rf"Minor axis = {2*min(a,b):.6g}",
        rf"Foci at (±c, 0) with c = \sqrt{{|a^2 - b^2|}} = {c:.6g}",
        rf"Eccentricity e = c/a = {eccentricity:.6g}",
    ]

    plot = [
        PlotElement(
            type="curve",
            data={"points": points, "label": "Ellipse"},
            style={"color": "#9467bd"},
        ),
        PlotElement(
            type="point",
            data={"coords": focus1, "label": "F1"},
            style={"color": "#d62728"},
        ),
        PlotElement(
            type="point",
            data={"coords": focus2, "label": "F2"},
            style={"color": "#2ca02c"},
        ),
    ]

    payload = {
        "semi_major": max(a, b),
        "semi_minor": min(a, b),
        "foci": [focus1, focus2],
        "eccentricity": eccentricity,
        "parametric": {"x": "a cos t", "y": "b sin t"},
    }

    return ComputationResult(
        status="ok",
        op="ellipse_standard",
        payload=payload,
        steps=steps,
        plot_elements=plot,
    )


def hyperbola_standard(a: float, b: float) -> ComputationResult:
    if a <= 0 or b <= 0:
        return ComputationResult(
            status="error",
            op="hyperbola_standard",
            payload={"message": "Semi-axes must be positive"},
            warnings=["degenerate"],
        )

    c = float(np.sqrt(a**2 + b**2))
    eccentricity = c / a
    asymptote_slope = b / a

    t_vals = np.linspace(-3, 3, 400)
    right_branch = [(a * np.cosh(t), b * np.sinh(t)) for t in t_vals]
    left_branch = [(-x, y) for x, y in right_branch]

    steps = [
        rf"c = \sqrt{{a^2 + b^2}} = {c:.6g}",
        rf"Eccentricity e = c/a = {eccentricity:.6g}",
        rf"Asymptotes: y = ±({b:.3g}/{a:.3g}) x",
    ]

    plot = [
        PlotElement(
            type="curve",
            data={"points": right_branch, "label": "Hyperbola+"},
            style={"color": "#1f77b4"},
        ),
        PlotElement(
            type="curve",
            data={"points": left_branch, "label": "Hyperbola-"},
            style={"color": "#1f77b4"},
        ),
        PlotElement(
            type="line",
            data={"from": (-10, -asymptote_slope * 10), "to": (10, asymptote_slope * 10)},
            style={"color": "#ff7f0e", "dash": "dot"},
        ),
        PlotElement(
            type="line",
            data={"from": (-10, asymptote_slope * 10), "to": (10, -asymptote_slope * 10)},
            style={"color": "#ff7f0e", "dash": "dot"},
        ),
    ]

    payload = {
        "foci": [(c, 0), (-c, 0)],
        "eccentricity": eccentricity,
        "asymptotes": [
            {"A": asymptote_slope, "B": -1, "C": 0},
            {"A": -asymptote_slope, "B": -1, "C": 0},
        ],
    }

    return ComputationResult(
        status="ok",
        op="hyperbola_standard",
        payload=payload,
        steps=steps,
        plot_elements=plot,
    )


def classify_general_quadratic(A: float, B: float, C: float, D: float, E: float, F: float) -> ComputationResult:
    discriminant = B**2 - 4 * A * C
    steps = [rf"Δ = B^2 - 4AC = {discriminant:.6g}"]

    if discriminant < 0:
        conic_type = "ellipse" if A == C else "imaginary ellipse"
    elif discriminant == 0:
        conic_type = "parabola"
    else:
        conic_type = "hyperbola"

    payload = {
        "discriminant": discriminant,
        "type": conic_type,
    }

    return ComputationResult(
        status="ok",
        op="classify_quadratic",
        payload=payload,
        steps=steps,
    )


def line_conic_intersection(line: Line, coeffs: Dict[str, float]) -> ComputationResult:
    A = coeffs.get("A", 0.0)
    B = coeffs.get("B", 0.0)
    C = coeffs.get("C", 0.0)
    D = coeffs.get("D", 0.0)
    E = coeffs.get("E", 0.0)
    F = coeffs.get("F", 0.0)

    equation = A * x**2 + B * x * y + C * y**2 + D * x + E * y + F
    line_eq = line.A * x + line.B * y + line.C

    substitution = solve(line_eq, y if abs(line.B) > 1e-9 else x)
    if not substitution:
        return ComputationResult(
            status="error",
            op="line_conic_intersection",
            payload={"message": "Unable to parametrize line"},
        )

    if abs(line.B) > 1e-9:
        y_expr = substitution[0]
        resulting = equation.subs(y, y_expr)
        roots = solve(resulting, x)
        points = [(float(root), float(y_expr.subs(x, root))) for root in roots]
    else:
        x_expr = substitution[0]
        resulting = equation.subs(x, x_expr)
        roots = solve(resulting, y)
        points = [(float(x_expr.subs(y, root)), float(root)) for root in roots]

    plot = [
        PlotElement(
            type="points",
            data={"coords": points},
        )
    ]

    steps = ["Solved quadratic obtained by substituting line equation into conic"]

    return ComputationResult(
        status="ok",
        op="line_conic_intersection",
        payload={"points": points},
        steps=steps,
        plot_elements=plot,
    )
