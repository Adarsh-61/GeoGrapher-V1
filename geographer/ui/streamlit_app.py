"""Streamlit front-end for GeoGrapher."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import streamlit as st

from geographer.core import (
    FunctionAnalyzer,
    Point,
    angle_between_lines,
    circle_circle_intersection,
    distance,
    distance_from_point,
    ellipse_standard,
    eigen_analysis,
    foot_of_perpendicular,
    hyperbola_standard,
    intersection,
    line_circle_intersection,
    line_conic_intersection,
    line_from_points,
    matrix_transform,
    midpoint,
    parabola_x2_4ay,
    parabola_y2_4ax,
    plot_basic_trig,
    radical_axis,
    section_point,
    tangents_from_external_point,
    transform_trig,
)
from geographer.core.lines import Line
from geographer.core.matrices import matrix_addition, matrix_determinant, matrix_inverse, matrix_multiplication
from geographer.core.triangles import TriangleMetrics
from geographer.viz import build_figure, export_json


@dataclass
class Operation:
    label: str
    handler: Callable[..., object]
    category: str
    description: str


_OPERATIONS: Dict[str, Operation] = {
    "distance": Operation(
        label="Distance between two points",
        handler=lambda args: distance(Point(*args["A"], label="A"), Point(*args["B"], label="B")),
        category="Coordinate Geometry",
        description="Compute distance between two points and show segment",
    ),
    "midpoint": Operation(
        label="Midpoint of segment",
        handler=lambda args: midpoint(Point(*args["A"], label="A"), Point(*args["B"], label="B")),
        category="Coordinate Geometry",
        description="Find midpoint of two points",
    ),
    "section_point": Operation(
        label="Section formula (internal)",
        handler=lambda args: section_point(
            Point(*args["A"], label="A"),
            Point(*args["B"], label="B"),
            (args["m"], args["n"]),
            external=args.get("external", False),
        ),
        category="Coordinate Geometry",
        description="Point dividing segment in ratio",
    ),
    "line_from_points": Operation(
        label="Line from two points",
        handler=lambda args: line_from_points(Point(*args["A"], label="A"), Point(*args["B"], label="B")),
        category="Coordinate Geometry",
        description="Line equation passing through two points",
    ),
    "line_intersection": Operation(
        label="Intersection of two lines",
        handler=lambda args: intersection(Line.from_general(*args["L1"]), Line.from_general(*args["L2"])),
        category="Coordinate Geometry",
        description="Intersection of two general-form lines",
    ),
    "triangle_summary": Operation(
        label="Triangle centers and area",
        handler=lambda args: TriangleMetrics(
            Point(*args["A"], label="A"),
            Point(*args["B"], label="B"),
            Point(*args["C"], label="C"),
        ).summary(),
        category="Coordinate Geometry",
        description="Area, centroid, circumcenter, incenter, orthocenter",
    ),
    "circle_line": Operation(
        label="Line and circle intersection",
        handler=lambda args: line_circle_intersection(
            args["circle"], args["line"]
        ),
        category="Coordinate Geometry",
        description="Intersection of a circle with a line",
    ),
    "circle_circle": Operation(
        label="Intersection of two circles",
        handler=lambda args: circle_circle_intersection(args["circle1"], args["circle2"]),
        category="Coordinate Geometry",
        description="Common points of two circles",
    ),
    "tangent_external": Operation(
        label="Tangents from external point",
        handler=lambda args: tangents_from_external_point(args["circle"], Point(*args["P"], label="P")),
        category="Coordinate Geometry",
        description="Tangents from a point to circle",
    ),
    "radical_axis": Operation(
        label="Radical axis",
        handler=lambda args: radical_axis(args["circle1"], args["circle2"]),
        category="Coordinate Geometry",
        description="Radical axis of two circles",
    ),
    "matrix_transform": Operation(
        label="Matrix transform of unit circle",
        handler=lambda args: matrix_transform(args["matrix"], shape=args.get("shape", "unit_circle")),
        category="Matrices",
        description="Apply matrix to unit circle or grid",
    ),
    "matrix_add": Operation(
        label="Matrix addition",
        handler=lambda args: matrix_addition(args["A"], args["B"]),
        category="Matrices",
        description="Add two matrices element-wise",
    ),
    "matrix_multiply": Operation(
        label="Matrix multiplication",
        handler=lambda args: matrix_multiplication(args["A"], args["B"]),
        category="Matrices",
        description="Multiply compatible matrices",
    ),
    "matrix_det": Operation(
        label="Determinant",
        handler=lambda args: matrix_determinant(args["A"]),
        category="Matrices",
        description="Determinant and area scaling",
    ),
    "matrix_inverse": Operation(
        label="Matrix inverse",
        handler=lambda args: matrix_inverse(args["A"]),
        category="Matrices",
        description="Compute inverse if exists",
    ),
    "eigen_analysis": Operation(
        label="Eigenvalues and eigenvectors",
        handler=lambda args: eigen_analysis(args["A"]),
        category="Matrices",
        description="Eigen decomposition for 2x2/3x3",
    ),
    "plot_trig": Operation(
        label="Plot basic trig function",
        handler=lambda args: plot_basic_trig(args["function"], tuple(args["domain"])),
        category="Trigonometry",
        description="Plot sin/cos/tan etc",
    ),
    "transform_trig": Operation(
        label="Transform trig function",
        handler=lambda args: transform_trig(
            args["amplitude"],
            args["frequency"],
            args["phase"],
            args["vertical"],
            base=args.get("function", "sin"),
            domain=tuple(args["domain"]),
        ),
        category="Trigonometry",
        description="Apply amplitude/frequency/phase shifts",
    ),
    "parabola_y": Operation(
        label="Parabola y^2 = 4ax",
        handler=lambda args: parabola_y2_4ax(args["a"]),
        category="Conics",
        description="Standard parabola opening right",
    ),
    "parabola_x": Operation(
        label="Parabola x^2 = 4ay",
        handler=lambda args: parabola_x2_4ay(args["a"]),
        category="Conics",
        description="Standard parabola opening up",
    ),
    "ellipse": Operation(
        label="Ellipse x^2/a^2 + y^2/b^2 = 1",
        handler=lambda args: ellipse_standard(args["a"], args["b"]),
        category="Conics",
        description="Ellipse properties and plot",
    ),
    "hyperbola": Operation(
        label="Hyperbola x^2/a^2 - y^2/b^2 = 1",
        handler=lambda args: hyperbola_standard(args["a"], args["b"]),
        category="Conics",
        description="Hyperbola properties and asymptotes",
    ),
    "function_plot": Operation(
        label="Plot function",
        handler=lambda args: FunctionAnalyzer(args["expr"]).plot(tuple(args["domain"]), show_derivative=args.get("show_derivative", False)),
        category="Calculus",
        description="Plot function with optional derivative",
    ),
    "tangent": Operation(
        label="Tangent at a point",
        handler=lambda args: FunctionAnalyzer(args["expr"]).derivative_at(args["x0"]),
        category="Calculus",
        description="Tangent slope at x0",
    ),
    "integral": Operation(
        label="Definite integral",
        handler=lambda args: FunctionAnalyzer(args["expr"]).definite_integral(args["a"], args["b"]),
        category="Calculus",
        description="Area under curve between bounds",
    ),
}


def _category_options() -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for op_key, op in _OPERATIONS.items():
        mapping.setdefault(op.category, []).append(op_key)
    return mapping


def _point_inputs(prefix: str, container) -> Tuple[float, float]:
    col1, col2 = container.columns(2)
    x_val = col1.number_input(f"{prefix} x", value=0.0, key=f"{prefix}_x")
    y_val = col2.number_input(f"{prefix} y", value=0.0, key=f"{prefix}_y")
    return (x_val, y_val)


def _matrix_input(label: str, container, size: Tuple[int, int] = (2, 2)) -> List[List[float]]:
    rows, cols = size
    container.markdown(f"**{label} ({rows}×{cols})**")
    matrix: List[List[float]] = []
    for r in range(rows):
        row_vals: List[float] = []
        cols_container = container.columns(cols)
        for c in range(cols):
            row_vals.append(cols_container[c].number_input(f"{label}[{r+1},{c+1}]", value=1.0 if r == c else 0.0, key=f"{label}_{r}_{c}") )
        matrix.append(row_vals)
    return matrix


def _circle_input(label: str, container):
    container.markdown(f"**{label}**")
    center = _point_inputs(f"{label}_center", container)
    radius = container.number_input(f"{label} radius", value=1.0, min_value=0.0, key=f"{label}_radius")
    from geographer.core.circles import Circle
    return Circle(Point(*center, label=f"{label[0]}"), radius)


def main() -> None:
    st.set_page_config(page_title="GeoGrapher", layout="wide")
    st.title("GeoGrapher — Interactive Coordinate Geometry & Math Visualizer")

    if "history" not in st.session_state:
        st.session_state.history = []

    categories = _category_options()
    sidebar = st.sidebar
    sidebar.header("Controls")
    category = sidebar.selectbox("Mode", list(categories.keys()))
    operation_key = sidebar.selectbox(
        "Operation", categories[category], format_func=lambda key: _OPERATIONS[key].label
    )
    operation = _OPERATIONS[operation_key]
    sidebar.write(operation.description)

    input_container = sidebar.form(key="inputs")
    submitted = False
    with input_container:
        args: Dict[str, object] = {}

        if operation_key in {"distance", "midpoint", "section_point"}:
            args["A"] = _point_inputs("A", input_container)
            args["B"] = _point_inputs("B", input_container)
            if operation_key == "section_point":
                args["m"] = input_container.number_input("m", value=1.0)
                args["n"] = input_container.number_input("n", value=1.0)
                args["external"] = input_container.checkbox("External division")
        elif operation_key == "line_from_points":
            args["A"] = _point_inputs("A", input_container)
            args["B"] = _point_inputs("B", input_container)
        elif operation_key == "line_intersection":
            input_container.markdown("**Line ℓ₁ (Ax + By + C = 0)**")
            args["L1"] = (
                input_container.number_input("A₁", value=1.0),
                input_container.number_input("B₁", value=-1.0),
                input_container.number_input("C₁", value=0.0),
            )
            input_container.markdown("**Line ℓ₂ (Ax + By + C = 0)**")
            args["L2"] = (
                input_container.number_input("A₂", value=0.0),
                input_container.number_input("B₂", value=1.0),
                input_container.number_input("C₂", value=-2.0),
            )
        elif operation_key == "triangle_summary":
            args["A"] = _point_inputs("A", input_container)
            args["B"] = _point_inputs("B", input_container)
            args["C"] = _point_inputs("C", input_container)
        elif operation_key in {"circle_line", "circle_circle", "tangent_external", "radical_axis"}:
            if operation_key == "circle_line":
                args["circle"] = _circle_input("Circle", input_container)
                input_container.markdown("**Line (Ax + By + C = 0)**")
                args["line"] = Line.from_general(
                    input_container.number_input("A", value=1.0),
                    input_container.number_input("B", value=-1.0),
                    input_container.number_input("C", value=0.0),
                    label="ℓ",
                )
            elif operation_key == "circle_circle":
                args["circle1"] = _circle_input("Circle 1", input_container)
                args["circle2"] = _circle_input("Circle 2", input_container)
            elif operation_key == "tangent_external":
                args["circle"] = _circle_input("Circle", input_container)
                args["P"] = _point_inputs("External point", input_container)
            else:
                args["circle1"] = _circle_input("Circle 1", input_container)
                args["circle2"] = _circle_input("Circle 2", input_container)
        elif operation_key.startswith("matrix_") or operation_key == "eigen_analysis":
            size = input_container.selectbox("Matrix size", options=["2x2", "3x3"], index=0)
            dim = 2 if size == "2x2" else 3
            args["A"] = _matrix_input("A", input_container, (dim, dim))
            if operation_key in {"matrix_add", "matrix_multiply"}:
                args["B"] = _matrix_input("B", input_container, (dim, dim))
            if operation_key == "matrix_transform":
                args["matrix"] = args["A"]
                args["shape"] = input_container.selectbox("Shape", options=["unit_circle", "grid"], index=0)
        elif operation_key in {"plot_trig", "transform_trig"}:
            func = input_container.selectbox("Function", list({"sin", "cos", "tan"}))
            args["function"] = func
            domain_start = input_container.number_input("Domain start", value=-6.28)
            domain_end = input_container.number_input("Domain end", value=6.28)
            args["domain"] = (domain_start, domain_end)
            if operation_key == "transform_trig":
                args["amplitude"] = input_container.number_input("Amplitude", value=1.0)
                args["frequency"] = input_container.number_input("Frequency", value=1.0)
                args["phase"] = input_container.number_input("Phase", value=0.0)
                args["vertical"] = input_container.number_input("Vertical shift", value=0.0)
        elif operation_key in {"parabola_y", "parabola_x", "ellipse", "hyperbola"}:
            if operation_key in {"parabola_y", "parabola_x"}:
                args["a"] = input_container.number_input("a", value=1.0)
            else:
                args["a"] = input_container.number_input("a", value=3.0)
                args["b"] = input_container.number_input("b", value=2.0)
        elif operation_key in {"function_plot", "tangent", "integral"}:
            args["expr"] = input_container.text_input("f(x)", value="sin(x)")
            if operation_key == "function_plot":
                args["domain"] = (
                    input_container.number_input("Domain start", value=-6.28),
                    input_container.number_input("Domain end", value=6.28),
                )
                args["show_derivative"] = input_container.checkbox("Show derivative", value=False)
            elif operation_key == "tangent":
                args["x0"] = input_container.number_input("x₀", value=0.0)
            elif operation_key == "integral":
                args["a"] = input_container.number_input("Lower limit a", value=0.0)
                args["b"] = input_container.number_input("Upper limit b", value=3.14)

    submitted = input_container.form_submit_button("Compute & Plot")

    placeholder_fig, placeholder_steps, placeholder_summary = st.columns([2.5, 1.2, 1.2])

    if submitted:
        try:
            result = operation.handler(args)
        except Exception as exc:  # pragma: no cover
            st.error(f"Operation failed: {exc}")
            return

        st.session_state.history.insert(0, {
            "operation": operation.label,
            "payload": result.payload,
        })

        figure = build_figure(result.plot_elements) if result.plot_elements else None
        with placeholder_fig:
            st.subheader("Interactive Plot")
            if figure:
                st.plotly_chart(figure, use_container_width=True)
            else:
                st.info("No plot available for this operation")

        with placeholder_steps:
            st.subheader("Steps")
            for step in result.steps:
                st.markdown(f"- ${step}$")

        with placeholder_summary:
            st.subheader("Result")
            st.json(result.payload)
            if result.warnings:
                st.warning("\n".join(result.warnings))

        export_col1, export_col2 = st.columns(2)
        with export_col1:
            if st.button("Export JSON", key="export_json"):
                path = export_json(result, "exports/result.json")
                st.success(f"Saved to {path}")
        with export_col2:
            st.download_button(
                "Copy LaTeX steps",
                data="\n".join(result.steps),
                file_name="steps.tex",
            )

    st.markdown("---")
    st.subheader("History")
    for entry in st.session_state.history[:5]:
        with st.expander(entry["operation"]):
            st.json(entry["payload"])


if __name__ == "__main__":
    main()
