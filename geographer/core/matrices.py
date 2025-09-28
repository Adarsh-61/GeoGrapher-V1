"""Matrix operations relevant to GeoGrapher visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sympy import Matrix

from .utils import ComputationResult, PlotElement, EPS


def _ensure_matrix(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.array(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Matrix must be two-dimensional")
    return arr


def matrix_addition(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> ComputationResult:
    mat_a = _ensure_matrix(a)
    mat_b = _ensure_matrix(b)
    if mat_a.shape != mat_b.shape:
        return ComputationResult(
            status="error",
            op="matrix_addition",
            payload={"message": "Matrices must share the same shape"},
        )

    result = mat_a + mat_b
    steps = ["Element-wise addition"]
    return ComputationResult(
        status="ok",
        op="matrix_addition",
        payload={"result": result.tolist()},
        steps=steps,
    )


def matrix_multiplication(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> ComputationResult:
    mat_a = _ensure_matrix(a)
    mat_b = _ensure_matrix(b)
    if mat_a.shape[1] != mat_b.shape[0]:
        return ComputationResult(
            status="error",
            op="matrix_multiplication",
            payload={"message": "Inner dimensions must agree"},
        )

    result = mat_a @ mat_b
    steps = ["Row-column multiplication"]
    return ComputationResult(
        status="ok",
        op="matrix_multiplication",
        payload={"result": result.tolist()},
        steps=steps,
    )


def matrix_determinant(matrix: Sequence[Sequence[float]]) -> ComputationResult:
    mat = _ensure_matrix(matrix)
    if mat.shape[0] != mat.shape[1]:
        return ComputationResult(
            status="error",
            op="matrix_determinant",
            payload={"message": "Matrix must be square"},
        )

    det = float(np.linalg.det(mat))
    steps = [rf"det(A) = {det:.6g}"]

    plot: List[PlotElement] = []
    if mat.shape == (2, 2):
        circle_points = _unit_circle_samples(120)
        transformed = (mat @ circle_points.T).T
        plot = [
            PlotElement(
                type="curve",
                data={"points": circle_points.tolist(), "label": "Unit circle"},
                style={"color": "#7f7f7f", "dash": "dot"},
            ),
            PlotElement(
                type="curve",
                data={"points": transformed.tolist(), "label": "Transformed"},
                style={"color": "#1f77b4"},
            ),
        ]

    return ComputationResult(
        status="ok",
        op="matrix_determinant",
        payload={"determinant": det},
        steps=steps,
        plot_elements=plot,
    )


def matrix_inverse(matrix: Sequence[Sequence[float]]) -> ComputationResult:
    mat = _ensure_matrix(matrix)
    if mat.shape[0] != mat.shape[1]:
        return ComputationResult(
            status="error",
            op="matrix_inverse",
            payload={"message": "Matrix must be square"},
        )

    det = np.linalg.det(mat)
    if abs(det) < EPS:
        return ComputationResult(
            status="warning",
            op="matrix_inverse",
            payload={"message": "Matrix is nearly singular"},
            warnings=["singular_matrix"],
        )

    inverse = np.linalg.inv(mat)
    steps = ["Compute adjugate / determinant (NumPy)"]
    return ComputationResult(
        status="ok",
        op="matrix_inverse",
        payload={"inverse": inverse.tolist(), "determinant": float(det)},
        steps=steps,
    )


def eigen_analysis(matrix: Sequence[Sequence[float]]) -> ComputationResult:
    mat = Matrix(matrix)
    eigenpairs = mat.eigenvects()
    payload = []
    plot: List[PlotElement] = []

    for eigenvalue, multiplicity, eigenvectors in eigenpairs:
        vector = eigenvectors[0]
        payload.append({
            "value": float(eigenvalue.evalf()),
            "vector": [float(component.evalf()) for component in vector],
            "multiplicity": multiplicity,
        })
        vec = np.array([float(component.evalf()) for component in vector], dtype=float)
        if vec.size == 2:
            points = [[0, 0], (vec * 2).tolist(), (vec * -2).tolist()]
            plot.append(
                PlotElement(
                    type="vector",
                    data={"points": points, "label": f"λ={float(eigenvalue.evalf()):.3g}"},
                    style={"color": "#d62728"},
                )
            )

    steps = ["SymPy eigen decomposition"]

    return ComputationResult(
        status="ok",
        op="eigen_analysis",
        payload={"eigenpairs": payload},
        steps=steps,
        plot_elements=plot,
    )


def matrix_transform(matrix: Sequence[Sequence[float]], shape: str = "unit_circle") -> ComputationResult:
    mat = _ensure_matrix(matrix)
    if mat.shape[0] != mat.shape[1]:
        return ComputationResult(
            status="error",
            op="matrix_transform",
            payload={"message": "Matrix must be square"},
        )

    if mat.shape[0] not in (2, 3):
        return ComputationResult(
            status="error",
            op="matrix_transform",
            payload={"message": "Only 2x2 or 3x3 matrices supported"},
        )

    if shape == "unit_circle":
        points = _unit_circle_samples(200)
    elif shape == "grid":
        points = _grid_samples()
    else:
        raise ValueError("Unsupported shape")

    transformed = (mat @ points.T).T

    plot = [
        PlotElement(
            type="curve" if shape == "unit_circle" else "lines",
            data={"points": points.tolist(), "label": "Original"},
            style={"color": "#7f7f7f", "dash": "dot"},
        ),
        PlotElement(
            type="curve" if shape == "unit_circle" else "lines",
            data={"points": transformed.tolist(), "label": "Transformed"},
            style={"color": "#1f77b4"},
        ),
    ]

    det = float(np.linalg.det(mat)) if mat.shape[0] == 2 else None
    payload = {
        "determinant": det,
        "transformed": transformed.tolist(),
    }

    steps = ["Apply linear map to sampled geometry"]

    return ComputationResult(
        status="ok",
        op="matrix_transform",
        payload=payload,
        steps=steps,
        plot_elements=plot,
    )


def _unit_circle_samples(n: int) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n)
    return np.column_stack((np.cos(theta), np.sin(theta)))


def _grid_samples() -> np.ndarray:
    xs = np.linspace(-1, 1, 5)
    ys = np.linspace(-1, 1, 5)
    grid_lines: List[List[float]] = []
    for x_val in xs:
        for y_val in ys:
            grid_lines.append([x_val, y_val])
    return np.array(grid_lines, dtype=float)
