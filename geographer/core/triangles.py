"""Triangle metrics and notable centers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sympy import Point2D, Polygon
from sympy.geometry import Triangle as SymTriangle

from .points import Point
from .utils import ComputationResult, PlotElement, EPS


@dataclass
class TriangleMetrics:
    a: Point
    b: Point
    c: Point

    def _sym_triangle(self) -> SymTriangle:
        return SymTriangle(self.a.to_sympy(), self.b.to_sympy(), self.c.to_sympy())

    def side_lengths(self) -> Tuple[float, float, float]:
        p_a, p_b, p_c = self.a.to_numpy(), self.b.to_numpy(), self.c.to_numpy()
        ab = float(np.linalg.norm(p_a - p_b))
        bc = float(np.linalg.norm(p_b - p_c))
        ca = float(np.linalg.norm(p_c - p_a))
        return ab, bc, ca

    def area(self) -> float:
        return abs(Polygon(self.a.to_sympy(), self.b.to_sympy(), self.c.to_sympy()).area)

    def centroid(self) -> Point:
        mx = (self.a.x + self.b.x + self.c.x) / 3
        my = (self.a.y + self.b.y + self.c.y) / 3
        return Point(mx, my, label="G")

    def circumcenter(self) -> Point:
        center = self._sym_triangle().circumcenter
        return Point(float(center.x), float(center.y), label="O")

    def incenter(self) -> Point:
        tri = self._sym_triangle()
        center = tri.incenter
        return Point(float(center.x), float(center.y), label="I")

    def orthocenter(self) -> Point:
        tri = self._sym_triangle()
        center = tri.orthocenter
        return Point(float(center.x), float(center.y), label="H")

    def circumradius(self) -> float:
        tri = self._sym_triangle()
        return float(tri.circumradius)

    def inradius(self) -> float:
        tri = self._sym_triangle()
        return float(tri.inradius)

    def classification(self) -> Dict[str, bool]:
        ab, bc, ca = self.side_lengths()
        sides = sorted([ab, bc, ca])
        result = {
            "equilateral": max(sides) - min(sides) <= EPS,
            "isosceles": abs(ab - bc) <= EPS or abs(bc - ca) <= EPS or abs(ca - ab) <= EPS,
            "right": abs(sides[2] ** 2 - (sides[0] ** 2 + sides[1] ** 2)) <= 1e-6,
        }
        result["scalene"] = not result["equilateral"] and not result["isosceles"]
        return result

    def summary(self) -> ComputationResult:
        area_value = float(self.area())
        centroid_point = self.centroid()
        circumcenter_point = self.circumcenter()
        incenter_point = self.incenter()
        orthocenter_point = self.orthocenter()
        circumradius_value = self.circumradius()
        inradius_value = self.inradius()
        classification = self.classification()
        sides = self.side_lengths()

        steps: List[str] = [
            rf"\text{{Area}} = \frac{{1}}{{2}}|x_1(y_2 - y_3) + x_2(y_3 - y_1) + x_3(y_1 - y_2)| = {area_value:.6g}",
            "Compute centroid as mean of vertices.",
            "Circumcenter is the intersection of perpendicular bisectors (SymPy).",
            "Incenter derived from angle bisectors (SymPy).",
            "Orthocenter is intersection of altitudes (SymPy).",
        ]

        plot_elements = [
            PlotElement(
                type="polygon",
                data={
                    "vertices": [self.a.as_tuple(), self.b.as_tuple(), self.c.as_tuple()],
                    "label": "ΔABC",
                },
                style={"fillcolor": "rgba(31,119,180,0.1)", "line": {"color": "#1f77b4"}},
            ),
            PlotElement(
                type="point",
                data={"coords": centroid_point.as_tuple(), "label": "G"},
                style={"color": "#ff7f0e"},
            ),
            PlotElement(
                type="point",
                data={"coords": circumcenter_point.as_tuple(), "label": "O"},
                style={"color": "#2ca02c"},
            ),
            PlotElement(
                type="circle",
                data={"center": circumcenter_point.as_tuple(), "radius": circumradius_value},
                style={"line": {"color": "#2ca02c", "dash": "dash"}},
            ),
            PlotElement(
                type="point",
                data={"coords": incenter_point.as_tuple(), "label": "I"},
                style={"color": "#d62728"},
            ),
            PlotElement(
                type="point",
                data={"coords": orthocenter_point.as_tuple(), "label": "H"},
                style={"color": "#9467bd"},
            ),
        ]

        payload = {
            "area": area_value,
            "centroid": centroid_point.as_tuple(),
            "circumcenter": circumcenter_point.as_tuple(),
            "incenter": incenter_point.as_tuple(),
            "orthocenter": orthocenter_point.as_tuple(),
            "circumradius": circumradius_value,
            "inradius": inradius_value,
            "classification": classification,
            "side_lengths": {
                "AB": sides[0],
                "BC": sides[1],
                "CA": sides[2],
            },
        }

        return ComputationResult(
            status="ok",
            op="triangle_summary",
            payload=payload,
            steps=steps,
            plot_elements=plot_elements,
        )
