"""Point primitives and segment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sympy import Point2D, symbols

from .utils import ComputationResult, PlotElement, ensure_numeric


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    label: str | None = None

    def as_tuple(self) -> Tuple[float, float]:
        return (float(self.x), float(self.y))

    def to_numpy(self) -> np.ndarray:
        return np.array(self.as_tuple(), dtype=float)

    def to_sympy(self) -> Point2D:
        return Point2D(self.x, self.y)

    @classmethod
    def from_iterable(cls, coords: Iterable[float], label: str | None = None) -> "Point":
        x, y = coords
        return cls(float(x), float(y), label=label)


def _point_payload(point: Point) -> Dict[str, float | str]:
    payload: Dict[str, float | str] = {"x": point.x, "y": point.y}
    if point.label:
        payload["label"] = point.label
    return payload


def _point_element(point: Point, color: str = "#1f77b4") -> PlotElement:
    data: Dict[str, object] = {"coords": point.as_tuple()}
    if point.label:
        data["label"] = point.label
    return PlotElement(
        type="point",
        data=data,
        style={"color": color, "size": 10},
    )


def distance(point_a: Point, point_b: Point) -> ComputationResult:
    ax, ay = point_a.as_tuple()
    bx, by = point_b.as_tuple()
    dx = bx - ax
    dy = by - ay
    squared = dx**2 + dy**2
    dist = float(np.sqrt(squared))

    steps: List[str] = [
        rf"d = \sqrt{{({bx:.3g}-{ax:.3g})^2 + ({by:.3g}-{ay:.3g})^2}} = {dist:.6g}",
    ]

    return ComputationResult(
        status="ok",
        op="distance",
        payload={"distance": dist},
        steps=steps,
        plot_elements=[
            _point_element(point_a, color="#d62728"),
            _point_element(point_b, color="#2ca02c"),
            PlotElement(
                type="segment",
                data={"from": point_a.as_tuple(), "to": point_b.as_tuple()},
                style={"color": "#9467bd", "width": 3},
            ),
        ],
    )


def midpoint(point_a: Point, point_b: Point) -> ComputationResult:
    mx = (point_a.x + point_b.x) / 2
    my = (point_a.y + point_b.y) / 2
    mid = Point(mx, my, label="M")

    steps = [
        rf"M_x = \frac{{{point_a.x:.3g} + {point_b.x:.3g}}}{{2}} = {mx:.6g}",
        rf"M_y = \frac{{{point_a.y:.3g} + {point_b.y:.3g}}}{{2}} = {my:.6g}",
    ]

    return ComputationResult(
        status="ok",
        op="midpoint",
        payload={"midpoint": _point_payload(mid)},
        steps=steps,
        plot_elements=[
            _point_element(point_a, color="#d62728"),
            _point_element(point_b, color="#2ca02c"),
            PlotElement(
                type="segment",
                data={"from": point_a.as_tuple(), "to": point_b.as_tuple()},
                style={"color": "#7f7f7f", "width": 1, "dash": "dash"},
            ),
            _point_element(mid, color="#1f77b4"),
        ],
    )


def section_point(point_a: Point, point_b: Point, ratio: Tuple[float, float], external: bool = False) -> ComputationResult:
    m, n = ratio
    if external:
        px = (m * point_a.x - n * point_b.x) / (m - n)
        py = (m * point_a.y - n * point_b.y) / (m - n)
        label = "P_ext"
    else:
        px = (m * point_a.x + n * point_b.x) / (m + n)
        py = (m * point_a.y + n * point_b.y) / (m + n)
        label = "P"

    section = Point(float(px), float(py), label=label)

    steps: List[str] = []
    if external:
        steps.append(rf"P_x = \frac{{{m}\cdot {point_a.x:.3g} - {n}\cdot {point_b.x:.3g}}}{{{m}-{n}}} = {px:.6g}")
        steps.append(rf"P_y = \frac{{{m}\cdot {point_a.y:.3g} - {n}\cdot {point_b.y:.3g}}}{{{m}-{n}}} = {py:.6g}")
    else:
        steps.append(rf"P_x = \frac{{{m}\cdot {point_a.x:.3g} + {n}\cdot {point_b.x:.3g}}}{{{m}+{n}}} = {px:.6g}")
        steps.append(rf"P_y = \frac{{{m}\cdot {point_a.y:.3g} + {n}\cdot {point_b.y:.3g}}}{{{m}+{n}}} = {py:.6g}")

    plot_elements = [
        _point_element(point_a, color="#d62728"),
        _point_element(point_b, color="#2ca02c"),
        _point_element(section, color="#1f77b4"),
        PlotElement(
            type="segment",
            data={"from": point_a.as_tuple(), "to": point_b.as_tuple()},
            style={"color": "#7f7f7f", "width": 1},
        ),
    ]

    return ComputationResult(
        status="ok",
        op="section_point_external" if external else "section_point",
        payload={"point": _point_payload(section)},
        steps=steps,
        plot_elements=plot_elements,
    )


def parametric_point(point_a: Point, point_b: Point, t: float) -> ComputationResult:
    ax, ay = point_a.as_tuple()
    bx, by = point_b.as_tuple()
    px = ax + t * (bx - ax)
    py = ay + t * (by - ay)
    param = Point(px, py, label=f"P({t:.2g})")

    steps = [
        rf"P_x = {ax:.3g} + {t:.3g}({bx:.3g} - {ax:.3g}) = {px:.6g}",
        rf"P_y = {ay:.3g} + {t:.3g}({by:.3g} - {ay:.3g}) = {py:.6g}",
    ]

    plot_elements = [
        _point_element(point_a, color="#d62728"),
        _point_element(point_b, color="#2ca02c"),
        _point_element(param, color="#1f77b4"),
        PlotElement(
            type="segment",
            data={"from": point_a.as_tuple(), "to": point_b.as_tuple()},
            style={"color": "#7f7f7f", "width": 1},
        ),
    ]

    return ComputationResult(
        status="ok",
        op="parametric_point",
        payload={"point": _point_payload(param)},
        steps=steps,
        plot_elements=plot_elements,
    )


def annotate_symbolic(point: Point) -> Dict[str, str]:
    x, y = symbols("x y")
    expr = point.to_sympy()
    return {
        "latex": rf"({expr.x}, {expr.y})",
        "text": f"({ensure_numeric(expr.x):.4g}, {ensure_numeric(expr.y):.4g})",
    }
