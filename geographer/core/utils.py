"""Utility helpers shared across GeoGrapher core modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from sympy import Eq, latex, sympify
from sympy.core.expr import Expr

EPS = 1e-9
DEFAULT_PRECISION = 12


@dataclass
class PlotElement:
    """Declarative specification of a drawable object."""

    type: str
    data: Dict[str, Any]
    style: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {"type": self.type, **self.data}
        if self.style:
            payload["style"] = self.style
        return payload


@dataclass
class ComputationResult:
    """Standard response for downstream consumers."""

    status: str
    op: str
    payload: Dict[str, Any]
    steps: List[str] = field(default_factory=list)
    plot_elements: List[PlotElement] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "op": self.op,
            "payload": self.payload,
            "steps": self.steps,
            "plot": [element.to_dict() for element in self.plot_elements],
            "warnings": self.warnings,
        }


def is_close(a: float, b: float, eps: float = EPS) -> bool:
    return abs(a - b) <= eps


def vector_norm(vector: Iterable[float]) -> float:
    arr = np.array(list(vector), dtype=float)
    return float(np.linalg.norm(arr))


def format_equation(expr: Expr) -> str:
    return latex(Eq(expr.lhs, expr.rhs)) if isinstance(expr, Eq) else latex(expr)


def to_latex_strings(expressions: Iterable[Expr]) -> List[str]:
    return [latex(expr) for expr in expressions]


def sympify_expr(expr: Any, **kwargs: Any) -> Expr:
    return sympify(expr, evaluate=kwargs.get("evaluate", True))


def ensure_numeric(value: Any, precision: int = DEFAULT_PRECISION) -> float:
    if hasattr(value, "evalf"):
        return float(value.evalf(precision))
    return float(value)


def clamp_domain(value: float, domain: Iterable[float]) -> float:
    low, high = tuple(domain)
    return max(low, min(high, value))
