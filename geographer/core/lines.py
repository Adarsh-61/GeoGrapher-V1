"""Line representations and operations."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from typing import Dict, List, Optional, Tuple

import numpy as np
from sympy import Eq, Line as SymLine, Point2D, Symbol

from .points import Point
from .utils import ComputationResult, PlotElement, EPS, ensure_numeric


@dataclass(frozen=True)
class Line:
    A: float
    B: float
    C: float
    label: str | None = None

    def normalized(self) -> "Line":
        norm = np.hypot(self.A, self.B)
        if norm <= EPS:
            raise ValueError("Degenerate line coefficients: both A and B near zero")
        return Line(self.A / norm, self.B / norm, self.C / norm, label=self.label)

    @classmethod
    def from_points(cls, p1: Point, p2: Point, label: str | None = None) -> "Line":
        if p1.as_tuple() == p2.as_tuple():
            raise ValueError("Cannot define a line with two identical points")
        x1, y1 = p1.as_tuple()
        x2, y2 = p2.as_tuple()
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        return cls(A, B, C, label)

    @classmethod
    def from_slope_intercept(cls, slope: float, intercept: float, label: str | None = None) -> "Line":
        if np.isfinite(slope):
            return cls(slope, -1.0, intercept, label)
        return cls(1.0, 0.0, -intercept, label)

    @classmethod
    def from_general(cls, A: float, B: float, C: float, label: str | None = None) -> "Line":
        if abs(A) <= EPS and abs(B) <= EPS:
            raise ValueError("Invalid general form: A and B cannot both be zero")
        return cls(float(A), float(B), float(C), label)

    def slope(self) -> Optional[float]:
        if abs(self.B) <= EPS:
            return None
        return -self.A / self.B

    def intercept(self) -> Optional[float]:
        if abs(self.B) <= EPS:
            return None
        return -self.C / self.B

    def x_intercept(self) -> Optional[float]:
        if abs(self.A) <= EPS:
            return None
        return -self.C / self.A

    def as_sympy(self) -> SymLine:
        if abs(self.B) <= EPS:
            # vertical line x = constant
            x_val = -self.C / self.A
            return SymLine(Point2D(x_val, 0), slope=None)
        y = Symbol("y")
        x = Symbol("x")
        expr = Eq(self.A * x + self.B * y + self.C, 0)
        return SymLine(expr)

    def sample_points(self) -> Tuple[Point, Point]:
        if abs(self.B) <= EPS:
            x = -self.C / self.A
            return (Point(x, 0), Point(x, 1))
        else:
            y1 = (-self.C - self.A * 0) / self.B
            y2 = (-self.C - self.A * 1) / self.B
            return (Point(0, y1), Point(1, y2))

    def display(self) -> Dict[str, float | str]:
        payload: Dict[str, float | str] = {"A": self.A, "B": self.B, "C": self.C}
        if self.label:
            payload["label"] = self.label
        slope_value = self.slope()
        if slope_value is not None:
            payload["slope"] = slope_value
            payload["y_intercept"] = self.intercept()
        else:
            payload["vertical"] = True
            payload["x_intercept"] = self.x_intercept()
        return payload


def line_from_points(p1: Point, p2: Point) -> ComputationResult:
    line = Line.from_points(p1, p2, label="ℓ")
    steps = [
        rf"A = y_1 - y_2 = {p1.y:.3g} - {p2.y:.3g} = {line.A:.6g}",
        rf"B = x_2 - x_1 = {p2.x:.3g} - {p1.x:.3g} = {line.B:.6g}",
        rf"C = x_1 y_2 - x_2 y_1 = {p1.x:.3g}\cdot {p2.y:.3g} - {p2.x:.3g}\cdot {p1.y:.3g} = {line.C:.6g}",
    ]

    p_start, p_end = line.sample_points()
    plot = [
        PlotElement(
            type="line",
            data={"from": p_start.as_tuple(), "to": p_end.as_tuple(), "label": line.label or "ℓ"},
            style={"color": "#1f77b4"},
        ),
        PlotElement(
            type="point",
            data={"coords": p1.as_tuple(), "label": p1.label or "A"},
            style={"color": "#d62728"},
        ),
        PlotElement(
            type="point",
            data={"coords": p2.as_tuple(), "label": p2.label or "B"},
            style={"color": "#2ca02c"},
        ),
    ]

    return ComputationResult(
        status="ok",
        op="line_from_points",
        payload={"line": line.display()},
        steps=steps,
        plot_elements=plot,
    )


def intersection(line1: Line, line2: Line) -> ComputationResult:
    sym_line1 = line1.as_sympy()
    sym_line2 = line2.as_sympy()
    point = sym_line1.intersection(sym_line2)
    if not point:
        return ComputationResult(
            status="warning",
            op="intersection",
            payload={"message": "Lines are parallel or coincident"},
            steps=["No unique intersection found."],
            plot_elements=[],
            warnings=["parallel_lines"],
        )

    intersection_point = Point(float(point[0].x), float(point[0].y), label="I")
    steps = [
        "Solve the system: {0} = 0, {1} = 0".format(
            sym_line1.equation(), sym_line2.equation()
        ),
        rf"I = ({intersection_point.x:.6g}, {intersection_point.y:.6g})",
    ]

    plot = [
        PlotElement(
            type="line",
            data={"from": line1.sample_points()[0].as_tuple(), "to": line1.sample_points()[1].as_tuple(), "label": line1.label or "ℓ₁"},
            style={"color": "#1f77b4"},
        ),
        PlotElement(
            type="line",
            data={"from": line2.sample_points()[0].as_tuple(), "to": line2.sample_points()[1].as_tuple(), "label": line2.label or "ℓ₂"},
            style={"color": "#ff7f0e"},
        ),
        PlotElement(
            type="point",
            data={"coords": intersection_point.as_tuple(), "label": "I"},
            style={"color": "#9467bd", "size": 12},
        ),
    ]

    return ComputationResult(
        status="ok",
        op="intersection",
        payload={"point": intersection_point.as_tuple()},
        steps=steps,
        plot_elements=plot,
    )


def angle_between_lines(line1: Line, line2: Line) -> ComputationResult:
    m1 = line1.slope()
    m2 = line2.slope()

    if m1 is None and m2 is None:
        angle = 0.0
    elif m1 is None or m2 is None:
        angle = 90.0
    else:
        tan_theta = abs((m2 - m1) / (1 + m1 * m2))
        angle = degrees(atan2(tan_theta, 1))

    steps = [
        rf"m_1 = {'∞' if m1 is None else f'{m1:.6g}'}",
        rf"m_2 = {'∞' if m2 is None else f'{m2:.6g}'}",
        rf"θ = \tan^{{-1}}\left|\frac{{m_2 - m_1}}{{1 + m_1 m_2}}\right| = {angle:.6g}°",
    ]

    plot = [
        PlotElement(
            type="angle",
            data={
                "line1": line1.sample_points()[0].as_tuple(),
                "line2": line2.sample_points()[0].as_tuple(),
                "vertex": intersection(line1, line2).payload.get("point")
                if angle not in (0.0, 90.0)
                else None,
                "value": angle,
            },
            style={"color": "#17becf"},
        )
    ]

    return ComputationResult(
        status="ok",
        op="angle_between_lines",
        payload={"angle_degrees": angle},
        steps=steps,
        plot_elements=plot,
    )


def distance_from_point(line: Line, point: Point) -> ComputationResult:
    numerator = abs(line.A * point.x + line.B * point.y + line.C)
    denominator = np.hypot(line.A, line.B)
    distance = numerator / denominator

    foot = foot_of_perpendicular(line, point)
    steps = [
        rf"d = \frac{{|{line.A:.3g}\cdot {point.x:.3g} + {line.B:.3g}\cdot {point.y:.3g} + {line.C:.3g}|}}{{\sqrt{{{line.A:.3g}^2 + {line.B:.3g}^2}}}} = {distance:.6g}",
    ]

    plot = [
        PlotElement(
            type="line",
            data={"from": line.sample_points()[0].as_tuple(), "to": line.sample_points()[1].as_tuple(), "label": line.label or "ℓ"},
            style={"color": "#1f77b4"},
        ),
        PlotElement(
            type="point",
            data={"coords": point.as_tuple(), "label": point.label or "P"},
            style={"color": "#d62728", "size": 12},
        ),
    ]

    if foot.status == "ok":
        foot_point = foot.payload["point"]
        plot.extend(
            [
                PlotElement(
                    type="point",
                    data={"coords": foot_point, "label": "F"},
                    style={"color": "#2ca02c", "size": 10},
                ),
                PlotElement(
                    type="segment",
                    data={"from": point.as_tuple(), "to": tuple(foot_point)},
                    style={"color": "#9467bd", "dash": "dot"},
                ),
            ]
        )

    return ComputationResult(
        status="ok",
        op="distance_point_line",
        payload={"distance": distance},
        steps=steps,
        plot_elements=plot,
    )


def foot_of_perpendicular(line: Line, point: Point) -> ComputationResult:
    normalized = line.normalized()
    A, B, C = normalized.A, normalized.B, normalized.C
    foot_x = (B * (B * point.x - A * point.y) - A * C) / (A**2 + B**2)
    foot_y = (A * (-B * point.x + A * point.y) - B * C) / (A**2 + B**2)
    foot_point = Point(float(foot_x), float(foot_y), label="F")

    steps = [
        "Project point onto line using normal projection formulas.",
        rf"F = ({foot_point.x:.6g}, {foot_point.y:.6g})",
    ]

    plot = [
        PlotElement(
            type="point",
            data={"coords": foot_point.as_tuple(), "label": "F"},
            style={"color": "#2ca02c"},
        )
    ]

    return ComputationResult(
        status="ok",
        op="foot_of_perpendicular",
        payload={"point": foot_point.as_tuple()},
        steps=steps,
        plot_elements=plot,
    )


def angle_bisectors(line1: Line, line2: Line) -> ComputationResult:
    l1 = line1.normalized()
    l2 = line2.normalized()

    bisector1 = Line(l1.A + l2.A, l1.B + l2.B, l1.C + l2.C, label="bisector1")
    bisector2 = Line(l1.A - l2.A, l1.B - l2.B, l1.C - l2.C, label="bisector2")

    steps = [
        "Normalize both lines to unit normals.",
        "Internal bisector: n₁ + n₂ = 0",
        "External bisector: n₁ - n₂ = 0",
    ]

    payload = {
        "internal": bisector1.display(),
        "external": bisector2.display(),
    }

    plot = [
        PlotElement(
            type="line",
            data={"from": bisector1.sample_points()[0].as_tuple(), "to": bisector1.sample_points()[1].as_tuple(), "label": "bisector₁"},
            style={"color": "#17becf", "dash": "dash"},
        ),
        PlotElement(
            type="line",
            data={"from": bisector2.sample_points()[0].as_tuple(), "to": bisector2.sample_points()[1].as_tuple(), "label": "bisector₂"},
            style={"color": "#bcbd22", "dash": "dot"},
        ),
    ]

    return ComputationResult(
        status="ok",
        op="angle_bisectors",
        payload=payload,
        steps=steps,
        plot_elements=plot,
    )
