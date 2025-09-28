"""Locus utilities for classical coordinate-geometry problems."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from .lines import Line
from .points import Point
from .utils import ComputationResult, PlotElement


def perpendicular_bisector(a: Point, b: Point) -> ComputationResult:
    midpoint_x = (a.x + b.x) / 2
    midpoint_y = (a.y + b.y) / 2
    direction = (- (b.y - a.y), b.x - a.x)
    line = Line.from_general(direction[0], direction[1], -(direction[0] * midpoint_x + direction[1] * midpoint_y), label="perp_bisector")

    steps = ["Find midpoint and slope perpendicular to segment"]

    plot = [
        PlotElement(
            type="segment",
            data={"from": a.as_tuple(), "to": b.as_tuple()},
            style={"color": "#7f7f7f"},
        ),
        PlotElement(
            type="line",
            data={"from": line.sample_points()[0].as_tuple(), "to": line.sample_points()[1].as_tuple(), "label": "bisector"},
            style={"color": "#1f77b4", "dash": "dash"},
        ),
    ]

    return ComputationResult(
        status="ok",
        op="perpendicular_bisector",
        payload={"line": line.display()},
        steps=steps,
        plot_elements=plot,
    )


def apollonius_circle(point_a: Point, point_b: Point, ratio: Tuple[float, float]) -> ComputationResult:
    m, n = ratio
    if m <= 0 or n <= 0:
        return ComputationResult(
            status="error",
            op="apollonius_circle",
            payload={"message": "Ratio parts must be positive"},
        )

    ax, ay = point_a.as_tuple()
    bx, by = point_b.as_tuple()

    center_x = (m * bx + n * ax) / (m + n)
    center_y = (m * by + n * ay) / (m + n)
    radius = abs(m * n) / (m + n) * np.linalg.norm(np.array(point_b.as_tuple()) - np.array(point_a.as_tuple())) / m

    circle_center = Point(center_x, center_y, label="C")

    plot = [
        PlotElement(
            type="circle",
            data={"center": circle_center.as_tuple(), "radius": radius},
            style={"line": {"color": "#ff7f0e"}},
        ),
        PlotElement(
            type="point",
            data={"coords": point_a.as_tuple(), "label": point_a.label or "A"},
            style={"color": "#d62728"},
        ),
        PlotElement(
            type="point",
            data={"coords": point_b.as_tuple(), "label": point_b.label or "B"},
            style={"color": "#2ca02c"},
        ),
    ]

    steps = ["Use Apollonius ratio formula for center and radius"]

    return ComputationResult(
        status="ok",
        op="apollonius_circle",
        payload={"center": circle_center.as_tuple(), "radius": radius},
        steps=steps,
        plot_elements=plot,
    )


def locus_midpoints(points: Iterable[Point]) -> ComputationResult:
    pts = list(points)
    segments = []
    midpoints = []
    for idx in range(0, len(pts), 2):
        if idx + 1 < len(pts):
            a, b = pts[idx], pts[idx + 1]
            midpoint = ((a.x + b.x) / 2, (a.y + b.y) / 2)
            midpoints.append(midpoint)
            segments.append((a.as_tuple(), b.as_tuple()))

    plot = [
        PlotElement(
            type="points",
            data={"coords": midpoints, "label": "Midpoints"},
            style={"color": "#9467bd"},
        )
    ]
    for seg in segments:
        plot.append(
            PlotElement(
                type="segment",
                data={"from": seg[0], "to": seg[1]},
                style={"color": "#7f7f7f", "dash": "dot"},
            )
        )

    steps = ["Compute midpoints for each given segment"]

    return ComputationResult(
        status="ok",
        op="locus_midpoints",
        payload={"midpoints": midpoints},
        steps=steps,
        plot_elements=plot,
    )
