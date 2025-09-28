"""Circle algebra, intersections, and tangents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sympy import Circle as SymCircle, Point2D, Symbol
from sympy.geometry import Line as SymLine

from .lines import Line
from .points import Point
from .utils import ComputationResult, PlotElement


@dataclass
class Circle:
    center: Point
    radius: float
    label: str | None = None

    def to_sympy(self) -> SymCircle:
        return SymCircle(self.center.to_sympy(), self.radius)

    @classmethod
    def from_center_radius(cls, center: Point, radius: float, label: str | None = None) -> "Circle":
        if radius <= 0:
            raise ValueError("Radius must be positive")
        return cls(center=center, radius=float(radius), label=label)

    @classmethod
    def from_three_points(cls, p1: Point, p2: Point, p3: Point, label: str | None = None) -> "Circle":
        sym_circle = SymCircle(p1.to_sympy(), p2.to_sympy(), p3.to_sympy())
        center = Point(float(sym_circle.center.x), float(sym_circle.center.y), label="O")
        return cls(center=center, radius=float(sym_circle.radius), label=label or "∘")

    @classmethod
    def from_general_form(cls, A: float, B: float, C: float, D: float, E: float, label: str | None = None) -> "Circle":
        # General second degree: Ax^2 + Ay^2 + Dx + Ey + F = 0 assuming A = B != 0
        if abs(A) < 1e-9 or abs(A - B) > 1e-9:
            raise ValueError("Invalid general circle equation: coefficients must satisfy A=B≠0")
        h = -D / (2 * A)
        k = -E / (2 * A)
        r_sq = (D**2 + E**2) / (4 * A**2) - C / A
        if r_sq <= 0:
            raise ValueError("Non-positive radius squared")
        center = Point(h, k, label="O")
        return cls(center=center, radius=float(np.sqrt(r_sq)), label=label or "∘")

    def display(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "center": self.center.as_tuple(),
            "radius": self.radius,
        }
        if self.label:
            payload["label"] = self.label
        return payload

    def summary(self) -> ComputationResult:
        steps = [
            rf"Center = ({self.center.x:.6g}, {self.center.y:.6g})",
            rf"Radius = {self.radius:.6g}",
        ]
        plot = [
            PlotElement(
                type="circle",
                data={"center": self.center.as_tuple(), "radius": self.radius, "label": self.label or "∘"},
                style={"line": {"color": "#1f77b4"}},
            ),
            PlotElement(
                type="point",
                data={"coords": self.center.as_tuple(), "label": self.center.label or "O"},
                style={"color": "#d62728"},
            ),
        ]
        return ComputationResult(
            status="ok",
            op="circle_summary",
            payload=self.display(),
            steps=steps,
            plot_elements=plot,
        )


def line_circle_intersection(circle: Circle, line: Line) -> ComputationResult:
    sym_circle = circle.to_sympy()
    sym_line = line.as_sympy()
    points = sym_circle.intersection(sym_line)
    if not points:
        return ComputationResult(
            status="warning",
            op="line_circle_intersection",
            payload={"message": "No real intersection"},
            steps=["Discriminant < 0"],
            warnings=["no_real_intersection"],
        )

    coords = [(float(pt.x), float(pt.y)) for pt in points]
    steps = [
        rf"Solve {sym_circle.equation()} and {sym_line.equation()} simultaneously",
    ]

    plot = [
        PlotElement(
            type="circle",
            data={"center": circle.center.as_tuple(), "radius": circle.radius},
            style={"line": {"color": "#1f77b4"}},
        ),
        PlotElement(
            type="line",
            data={"from": line.sample_points()[0].as_tuple(), "to": line.sample_points()[1].as_tuple()},
            style={"color": "#ff7f0e"},
        ),
    ]
    for idx, coord in enumerate(coords):
        plot.append(
            PlotElement(
                type="point",
                data={"coords": coord, "label": f"P{idx+1}"},
                style={"color": "#2ca02c"},
            )
        )

    return ComputationResult(
        status="ok",
        op="line_circle_intersection",
        payload={"points": coords},
        steps=steps,
        plot_elements=plot,
    )


def circle_circle_intersection(circle1: Circle, circle2: Circle) -> ComputationResult:
    sym1 = circle1.to_sympy()
    sym2 = circle2.to_sympy()
    points = sym1.intersection(sym2)
    if not points:
        return ComputationResult(
            status="warning",
            op="circle_circle_intersection",
            payload={"message": "No real intersection"},
            steps=["Circles are separate or one contains the other"],
            warnings=["no_real_intersection"],
        )

    coords = [(float(pt.x), float(pt.y)) for pt in points]
    steps = ["Solve circle equations pairwise"]
    plot = [
        PlotElement(
            type="circle",
            data={"center": circle1.center.as_tuple(), "radius": circle1.radius},
            style={"line": {"color": "#1f77b4"}},
        ),
        PlotElement(
            type="circle",
            data={"center": circle2.center.as_tuple(), "radius": circle2.radius},
            style={"line": {"color": "#ff7f0e"}},
        ),
    ]
    for idx, coord in enumerate(coords):
        plot.append(
            PlotElement(
                type="point",
                data={"coords": coord, "label": f"P{idx+1}"},
                style={"color": "#2ca02c"},
            )
        )
    return ComputationResult(
        status="ok",
        op="circle_circle_intersection",
        payload={"points": coords},
        steps=steps,
        plot_elements=plot,
    )


def tangents_from_external_point(circle: Circle, point: Point) -> ComputationResult:
    sym_circle = circle.to_sympy()
    sym_point = point.to_sympy()
    tangent_lines = sym_circle.tangent_lines(sym_point)
    if not tangent_lines:
        return ComputationResult(
            status="warning",
            op="tangents_from_point",
            payload={"message": "Point lies inside the circle"},
            warnings=["no_tangent"],
        )

    lines_payload: List[Dict[str, float]] = []
    plot = [
        PlotElement(
            type="circle",
            data={"center": circle.center.as_tuple(), "radius": circle.radius},
            style={"line": {"color": "#1f77b4"}},
        ),
        PlotElement(
            type="point",
            data={"coords": point.as_tuple(), "label": point.label or "P"},
            style={"color": "#d62728"},
        ),
    ]

    for idx, tangent in enumerate(tangent_lines):
        coeffs = tangent.coefficients
        tangent_line = Line.from_general(coeffs[0], coeffs[1], coeffs[2], label=f"t{idx+1}")
        lines_payload.append(tangent_line.display())
        plot.append(
            PlotElement(
                type="line",
                data={"from": tangent_line.sample_points()[0].as_tuple(), "to": tangent_line.sample_points()[1].as_tuple(), "label": tangent_line.label},
                style={"color": "#ff7f0e", "dash": "dash"},
            )
        )

    return ComputationResult(
        status="ok",
        op="tangents_from_point",
        payload={"tangents": lines_payload},
        steps=["Use SymPy tangent_lines for external point"],
        plot_elements=plot,
    )


def radical_axis(circle1: Circle, circle2: Circle) -> ComputationResult:
    sym1 = circle1.to_sympy()
    sym2 = circle2.to_sympy()
    axis_line: SymLine = sym1.radical_axis(sym2)
    coeffs = axis_line.coefficients
    line = Line.from_general(coeffs[0], coeffs[1], coeffs[2], label="radical_axis")

    steps = ["Radical axis derived from difference of circle power equations."]
    plot = [
        PlotElement(
            type="line",
            data={"from": line.sample_points()[0].as_tuple(), "to": line.sample_points()[1].as_tuple(), "label": line.label},
            style={"color": "#17becf", "dash": "dash"},
        )
    ]

    return ComputationResult(
        status="ok",
        op="radical_axis",
        payload={"line": line.display()},
        steps=steps,
        plot_elements=plot,
    )
