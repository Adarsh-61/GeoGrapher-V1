"""Calculus utilities for visualization and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sympy import Eq, Integral, diff, integrate, lambdify, solveset, sympify
from sympy.abc import x

from .utils import ComputationResult, PlotElement


@dataclass
class FunctionAnalyzer:
    expression: str

    def __post_init__(self) -> None:
        self.sympy_expr = sympify(self.expression)
        self.numeric = lambdify(x, self.sympy_expr, "numpy")
        self.derivative_expr = diff(self.sympy_expr, x)
        self.derivative_numeric = lambdify(x, self.derivative_expr, "numpy")
        self.second_derivative_expr = diff(self.derivative_expr, x)
        self.second_derivative_numeric = lambdify(x, self.second_derivative_expr, "numpy")

    def plot(self, domain: Tuple[float, float], resolution: int = 800, show_derivative: bool = False) -> ComputationResult:
        xs = np.linspace(domain[0], domain[1], resolution)
        ys = self.numeric(xs)
        plot = [
            PlotElement(
                type="curve",
                data={"points": list(zip(xs.tolist(), ys.tolist())), "label": "f(x)"},
                style={"color": "#1f77b4"},
            )
        ]
        payload: Dict[str, object] = {"domain": domain}
        steps: List[str] = [f"Plot f(x) = {self.expression}"]

        if show_derivative:
            dy = self.derivative_numeric(xs)
            plot.append(
                PlotElement(
                    type="curve",
                    data={"points": list(zip(xs.tolist(), dy.tolist())), "label": "f'(x)"},
                    style={"color": "#ff7f0e", "dash": "dash"},
                )
            )
            payload["derivative"] = str(self.derivative_expr)
            steps.append("Overlay derivative curve")

        return ComputationResult(
            status="ok",
            op="plot_function",
            payload=payload,
            steps=steps,
            plot_elements=plot,
        )

    def derivative_at(self, x0: float) -> ComputationResult:
        slope = float(self.derivative_numeric(x0))
        y0 = float(self.numeric(x0))
        tangent_points = [(x0 - 1, y0 - slope), (x0 + 1, y0 + slope)]

        steps = [
            rf"f'({x0:.3g}) = {slope:.6g}",
            rf"Tangent line: y - {y0:.6g} = {slope:.6g}(x - {x0:.6g})",
        ]

        plot = [
            PlotElement(
                type="point",
                data={"coords": (x0, y0), "label": "P"},
                style={"color": "#d62728", "size": 12},
            ),
            PlotElement(
                type="line",
                data={"from": tangent_points[0], "to": tangent_points[1], "label": "Tangent"},
                style={"color": "#ff7f0e", "dash": "dash"},
            ),
        ]

        return ComputationResult(
            status="ok",
            op="tangent_line",
            payload={"slope": slope, "point": (x0, y0)},
            steps=steps,
            plot_elements=plot,
        )

    def definite_integral(self, start: float, end: float) -> ComputationResult:
        symbol_integral = Integral(self.sympy_expr, (x, start, end))
        value = float(symbol_integral.doit().evalf())

        xs = np.linspace(start, end, 200)
        ys = self.numeric(xs)
        area_points = list(zip(xs.tolist(), ys.tolist()))

        plot = [
            PlotElement(
                type="area",
                data={"points": area_points, "baseline": 0},
                style={"color": "rgba(31,119,180,0.3)"},
            )
        ]

        steps = [rf"∫_{{{start:.3g}}}^{{{end:.3g}}} f(x) dx = {value:.6g}"]

        return ComputationResult(
            status="ok",
            op="definite_integral",
            payload={"integral": value},
            steps=steps,
            plot_elements=plot,
        )

    def critical_points(self) -> ComputationResult:
        roots = solveset(self.derivative_expr, x)
        candidates = []
        for root in roots:
            if root.is_real:
                double_prime = float(self.second_derivative_expr.subs(x, root).evalf())
                classification = (
                    "min"
                    if double_prime > 0
                    else "max" if double_prime < 0 else "saddle"
                )
                candidates.append(
                    {
                        "x": float(root.evalf()),
                        "y": float(self.numeric(root.evalf())),
                        "classification": classification,
                    }
                )

        plot = [
            PlotElement(
                type="points",
                data={"coords": [(cand["x"], cand["y"]) for cand in candidates], "labels": [cand["classification"] for cand in candidates]},
                style={"color": "#2ca02c", "size": 12},
            )
        ]

        steps = ["Solve f'(x) = 0 and use second derivative test"]

        return ComputationResult(
            status="ok",
            op="critical_points",
            payload={"points": candidates},
            steps=steps,
            plot_elements=plot,
        )

    def taylor_polynomial(self, order: int, about: float = 0.0) -> ComputationResult:
        series = self.sympy_expr.series(x, about, order + 1).removeO()
        steps = [f"Taylor polynomial up to order {order} around {about}"]

        xs = np.linspace(about - 5, about + 5, 400)
        approx_fn = lambdify(x, series, "numpy")
        ys_original = self.numeric(xs)
        ys_series = approx_fn(xs)

        plot = [
            PlotElement(
                type="curve",
                data={"points": list(zip(xs.tolist(), ys_original.tolist())), "label": "f(x)"},
                style={"color": "#1f77b4"},
            ),
            PlotElement(
                type="curve",
                data={"points": list(zip(xs.tolist(), ys_series.tolist())), "label": f"Taylor order {order}"},
                style={"color": "#ff7f0e", "dash": "dot"},
            ),
        ]

        return ComputationResult(
            status="ok",
            op="taylor_polynomial",
            payload={"series": str(series)},
            steps=steps,
            plot_elements=plot,
        )
