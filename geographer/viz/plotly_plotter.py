"""Translate GeoGrapher plot element specs into Plotly figures."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import plotly.graph_objects as go

from geographer.core.utils import PlotElement


_COLOR_DEFAULT = "#1f77b4"


def build_figure(elements: Iterable[PlotElement], layout: dict | None = None) -> go.Figure:
    traces: List[go.BaseTraceType] = []
    layout_config = {
        "xaxis": {"zeroline": True, "showgrid": True, "scaleanchor": "y"},
        "yaxis": {"zeroline": True, "showgrid": True},
        "legend": {"orientation": "h"},
        "margin": dict(l=40, r=40, t=40, b=40),
    }
    if layout:
        layout_config.update(layout)

    for element in elements:
        handler = _HANDLERS.get(element.type)
        if handler:
            trace = handler(element)
            if isinstance(trace, list):
                traces.extend(trace)
            elif trace is not None:
                traces.append(trace)

    fig = go.Figure(data=traces)
    fig.update_layout(**layout_config)
    return fig


def _handle_point(element: PlotElement) -> go.Scatter:
    coords = element.data["coords"]
    label = element.data.get("label", "")
    color = _extract_color(element)
    return go.Scatter(
        x=[coords[0]],
        y=[coords[1]],
        mode="markers+text" if label else "markers",
        text=[label],
        textposition="top right",
        marker=dict(color=color, size=element.style.get("size", 10)),
        name=label or "point",
    )


def _handle_points(element: PlotElement) -> go.Scatter:
    coords = element.data["coords"]
    labels = element.data.get("labels")
    color = _extract_color(element)
    return go.Scatter(
        x=[p[0] for p in coords],
        y=[p[1] for p in coords],
        mode="markers+text" if labels else "markers",
        text=labels,
        textposition="top center",
        marker=dict(color=color, size=element.style.get("size", 9)),
        name=element.data.get("label", "points"),
    )


def _handle_segment(element: PlotElement) -> go.Scatter:
    start = element.data["from"]
    end = element.data["to"]
    color = _extract_color(element)
    return go.Scatter(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        mode="lines",
        line=dict(color=color, width=element.style.get("width", 2), dash=element.style.get("dash")),
        name=element.data.get("label", "segment"),
    )


def _handle_line(element: PlotElement) -> go.Scatter:
    start = element.data["from"]
    end = element.data["to"]
    color = _extract_color(element)
    return go.Scatter(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        mode="lines",
        line=dict(color=color, width=element.style.get("width", 2), dash=element.style.get("dash")),
        name=element.data.get("label", "line"),
    )


def _handle_polygon(element: PlotElement) -> List[go.BaseTraceType]:
    vertices = element.data["vertices"]
    x_points = [v[0] for v in vertices] + [vertices[0][0]]
    y_points = [v[1] for v in vertices] + [vertices[0][1]]
    fillcolor = element.style.get("fillcolor", "rgba(31,119,180,0.2)")
    line_style = element.style.get("line", {})
    return [
        go.Scatter(
            x=x_points,
            y=y_points,
            mode="lines",
            fill="toself",
            fillcolor=fillcolor,
            line=dict(color=line_style.get("color", _COLOR_DEFAULT)),
            name=element.data.get("label", "polygon"),
        )
    ]


def _handle_circle(element: PlotElement) -> List[go.BaseTraceType]:
    center = element.data["center"]
    radius = element.data["radius"]
    theta = np.linspace(0, 2 * np.pi, 200)
    xs = center[0] + radius * np.cos(theta)
    ys = center[1] + radius * np.sin(theta)
    line_style = element.style.get("line", {})
    return [
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=line_style.get("color", _COLOR_DEFAULT), dash=line_style.get("dash")),
            name=element.data.get("label", "circle"),
        )
    ]


def _handle_curve(element: PlotElement) -> go.Scatter:
    points = element.data["points"]
    label = element.data.get("label", "curve")
    color = _extract_color(element)
    return go.Scatter(
        x=[p[0] for p in points],
        y=[p[1] for p in points],
        mode="lines",
        line=dict(color=color),
        name=label,
    )


def _handle_area(element: PlotElement) -> go.Scatter:
    points = element.data["points"]
    baseline = element.data.get("baseline", 0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return go.Scatter(
        x=xs,
        y=ys,
        fill="tozeroy",
        fillcolor=element.style.get("color", "rgba(31,119,180,0.3)"),
        line=dict(color=_extract_color(element)),
        name="area",
    )


def _handle_vector(element: PlotElement) -> go.Scatter:
    points = element.data["points"]
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    return go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines+markers",
        line=dict(color=_extract_color(element)),
        marker=dict(size=10),
        name=element.data.get("label", "vector"),
    )


def _handle_lines(element: PlotElement) -> go.Scatter:
    points = element.data["points"]
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    return go.Scatter(
        x=x_values,
        y=y_values,
        mode="markers",
        marker=dict(color=_extract_color(element), size=6),
        name=element.data.get("label", "lines"),
    )


def _handle_angle(element: PlotElement) -> List[go.BaseTraceType]:
    vertex = element.data.get("vertex")
    if not vertex:
        return []
    value = element.data.get("value")
    label = f"{value:.1f}°" if value is not None else "Angle"
    return [
        go.Scatter(
            x=[vertex[0]],
            y=[vertex[1]],
            mode="text",
            text=[label],
            textposition="top right",
            name="angle",
        )
    ]


def _extract_color(element: PlotElement) -> str:
    style = element.style or {}
    if "color" in style:
        return style["color"]
    line_style = style.get("line")
    if isinstance(line_style, dict) and "color" in line_style:
        return line_style["color"]
    return _COLOR_DEFAULT


_HANDLERS = {
    "point": _handle_point,
    "points": _handle_points,
    "segment": _handle_segment,
    "line": _handle_line,
    "polygon": _handle_polygon,
    "circle": _handle_circle,
    "curve": _handle_curve,
    "area": _handle_area,
    "vector": _handle_vector,
    "lines": _handle_lines,
    "angle": _handle_angle,
}
