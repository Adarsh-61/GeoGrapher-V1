"""Coordinate transforms and affine utilities."""

from __future__ import annotations

from math import cos, radians, sin
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .lines import Line
from .points import Point
from .utils import ComputationResult, PlotElement


def apply_translation(points: Iterable[Point], dx: float, dy: float) -> ComputationResult:
    original = [p.as_tuple() for p in points]
    translated = [(x + dx, y + dy) for x, y in original]

    plot = [
        PlotElement(
            type="points",
            data={"coords": original, "label": "Original"},
            style={"color": "#7f7f7f", "size": 10},
        ),
        PlotElement(
            type="points",
            data={"coords": translated, "label": "Translated"},
            style={"color": "#1f77b4", "size": 10},
        ),
    ]

    steps = [rf"(x', y') = (x + {dx}, y + {dy})"]

    return ComputationResult(
        status="ok",
        op="translation",
        payload={"translated": translated},
        steps=steps,
        plot_elements=plot,
    )


def apply_rotation(points: Iterable[Point], angle_degrees: float, pivot: Point | None = None) -> ComputationResult:
    theta = radians(angle_degrees)
    pivot = pivot or Point(0, 0)

    def rotate(x: float, y: float) -> Tuple[float, float]:
        shifted_x = x - pivot.x
        shifted_y = y - pivot.y
        rx = shifted_x * cos(theta) - shifted_y * sin(theta)
        ry = shifted_x * sin(theta) + shifted_y * cos(theta)
        return (rx + pivot.x, ry + pivot.y)

    original = [p.as_tuple() for p in points]
    rotated = [rotate(x, y) for x, y in original]

    plot = [
        PlotElement(
            type="points",
            data={"coords": original, "label": "Original"},
            style={"color": "#7f7f7f"},
        ),
        PlotElement(
            type="points",
            data={"coords": rotated, "label": "Rotated"},
            style={"color": "#ff7f0e"},
        ),
        PlotElement(
            type="point",
            data={"coords": pivot.as_tuple(), "label": pivot.label or "Pivot"},
            style={"color": "#d62728", "size": 12},
        ),
    ]

    steps = [rf"Rotation by {angle_degrees}° about {pivot.as_tuple()}"]

    return ComputationResult(
        status="ok",
        op="rotation",
        payload={"rotated": rotated},
        steps=steps,
        plot_elements=plot,
    )


def apply_scaling(points: Iterable[Point], sx: float, sy: float, pivot: Point | None = None) -> ComputationResult:
    pivot = pivot or Point(0, 0)
    original = [p.as_tuple() for p in points]
    scaled = [
        (
            pivot.x + sx * (x - pivot.x),
            pivot.y + sy * (y - pivot.y),
        )
        for x, y in original
    ]

    plot = [
        PlotElement(
            type="points",
            data={"coords": original, "label": "Original"},
            style={"color": "#7f7f7f"},
        ),
        PlotElement(
            type="points",
            data={"coords": scaled, "label": "Scaled"},
            style={"color": "#2ca02c"},
        ),
    ]

    steps = [rf"Scaling about {pivot.as_tuple()} with factors ({sx}, {sy})"]

    return ComputationResult(
        status="ok",
        op="scaling",
        payload={"scaled": scaled},
        steps=steps,
        plot_elements=plot,
    )


def apply_reflection(points: Iterable[Point], line: Line) -> ComputationResult:
    sym_line = line.as_sympy()
    reflected = []
    for point in points:
        sym_point = point.to_sympy()
        mirrored = sym_point.reflect(sym_line)
        reflected.append((float(mirrored.x), float(mirrored.y)))

    plot = [
        PlotElement(
            type="line",
            data={"from": line.sample_points()[0].as_tuple(), "to": line.sample_points()[1].as_tuple(), "label": line.label or "Mirror"},
            style={"color": "#1f77b4", "dash": "dash"},
        )
    ]

    plot.append(
        PlotElement(
            type="points",
            data={"coords": [p.as_tuple() for p in points], "label": "Original"},
            style={"color": "#d62728"},
        )
    )
    plot.append(
        PlotElement(
            type="points",
            data={"coords": reflected, "label": "Reflected"},
            style={"color": "#2ca02c"},
        )
    )

    steps = ["Reflect each point across the mirror line"]

    return ComputationResult(
        status="ok",
        op="reflection",
        payload={"reflected": reflected},
        steps=steps,
        plot_elements=plot,
    )


def apply_affine(matrix: Sequence[Sequence[float]], points: Iterable[Point]) -> ComputationResult:
    mat = np.array(matrix, dtype=float)
    if mat.shape != (3, 3):
        raise ValueError("Affine transform requires 3x3 homogeneous matrix")

    original = [p.as_tuple() for p in points]
    homo_points = np.column_stack((np.array(original), np.ones(len(original))))
    transformed = (mat @ homo_points.T).T
    transformed_cartesian = [(row[0] / row[2], row[1] / row[2]) for row in transformed]

    plot = [
        PlotElement(
            type="points",
            data={"coords": original, "label": "Original"},
            style={"color": "#7f7f7f"},
        ),
        PlotElement(
            type="points",
            data={"coords": transformed_cartesian, "label": "Affine"},
            style={"color": "#9467bd"},
        ),
    ]

    steps = ["Apply homogeneous matrix to points"]

    return ComputationResult(
        status="ok",
        op="affine_transform",
        payload={"transformed": transformed_cartesian},
        steps=steps,
        plot_elements=plot,
    )
